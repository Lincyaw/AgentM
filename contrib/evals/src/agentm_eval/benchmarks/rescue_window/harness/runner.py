"""Branch runner: fork a prefix, inject one treatment, roll out the actor.

The runner forks a persisted baseline session at a prefix (conversation fork =
exact checkpoint for the read-only RCA data plane, DESIGN §1), injects the
treatment's channel message, continues the fixed actor pi_A to completion, judges
the result, and returns one ``EvalUnit`` per rollout.

CONTINUE is special: a faithful no-message re-rollout needs a resume primitive
the runtime does not expose generically, so the CONTINUE reference is the
baseline trajectory's own recorded outcome — it literally continued from every
prefix on it. Intervention branches fork+prompt and carry sampling variance; the
asymmetry is what E0 validates (DESIGN §5).
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from loguru import logger

from agentm.core.abi import AgentMessage, AgentSessionConfig
from agentm.core.abi.loop import LoopConfig
from agentm.core.abi.session_store import SessionStore
from agentm.core.runtime import AgentSession
from agentm.core.runtime.session_bootstrap import make_default_session_store

from ..model import EvalUnit, PrefixPoint, Treatment
from .adapter import ScenarioAdapter, ScoredOutcome
from .corpus import TrajectoryRef, load_trajectory_messages

_ENV_SESSION_LOCK = asyncio.Lock()
_ENV_SESSION_HELD: ContextVar[bool] = ContextVar("rescue_window_env_session_held", default=False)


@dataclass(frozen=True, slots=True)
class RolloutConfig:
    """Runtime knobs shared by every branch at a prefix (budget fairness)."""

    max_turns: int | None = 60
    max_tool_calls: int | None = None
    idle_timeout: float = 120.0
    cwd: str | None = None


async def run_intervention_rollout(
    *,
    ref: TrajectoryRef,
    prefix: PrefixPoint,
    treatment: Treatment,
    seed: int,
    store: SessionStore,
    adapter: ScenarioAdapter,
    config: RolloutConfig,
    provider_override: tuple[str, dict[str, Any]] | None = None,
) -> EvalUnit:
    """Fork at ``prefix``, inject ``treatment``, roll out once, judge, return a row.

    ``provider_override`` replaces the baseline's stored provider for the
    continuation actor — needed when the recorded endpoint is unreachable (stale
    base_url). Keep the same model to preserve the fixed actor pi_A.
    """

    base = _base_unit(ref, prefix, treatment, seed)
    if not treatment.is_continue and not treatment.intervention.message:
        return _with_status(base, "skipped", error="empty intervention message")

    try:
        source = store.open(ref.trajectory_id)
        header = source.get_header()
        if header is None:
            raise RuntimeError("source session has no header")
        stored = dict(header.config or {})
        scenario = _stored_scenario(stored)
        provider = provider_override or _stored_provider(stored)
        cwd = config.cwd or header.cwd or os.getcwd()
        forked = store.fork(ref.trajectory_id, **prefix.fork_point.to_dict())
        env = _stored_agentm_env(stored)

        # Let the adapter prepare the execution environment (e.g. ARL
        # sandbox + replay for stateful benchmarks). No-op for read-only
        # data planes like RCA.
        source_messages = _source_messages_for_env(store, ref)
        env_handle = await adapter.setup_environment(
            ref, prefix.turn_index, source_messages,
        )
        atom_overrides: dict[str, dict[str, Any]] = {}
        if env_handle is not None:
            atom_overrides = env_handle.atom_config_overrides()

        async with env_session_scope():
            with _temporary_env(env):
                session = await AgentSession.create(
                    AgentSessionConfig(
                        cwd=cwd,
                        session_manager=cast(Any, forked),
                        scenario=scenario,
                        provider=provider,
                        atom_config_overrides=atom_overrides,
                        loop_config=LoopConfig(
                            max_turns=config.max_turns,
                            max_tool_calls=config.max_tool_calls,
                        ),
                        parent_session_id=ref.trajectory_id,
                        lineage={
                            "kind": "fork",
                            "entrypoint": "rescue_window.runner",
                            "source_session_id": ref.trajectory_id,
                            "fork_point": prefix.fork_point.to_dict(),
                        },
                        experiment=_experiment_payload(ref, prefix, treatment, seed),
                    )
                )
        messages: list[AgentMessage] = []
        try:
            if treatment.is_continue:
                await session.resume()
            else:
                await session.prompt(treatment.intervention.message)
            await session.idle(timeout=config.idle_timeout)
            messages = session.session_manager.get_messages()
        finally:
            await session.shutdown()
            if env_handle is not None:
                await env_handle.teardown()
        outcome = await adapter.judge(messages, ref)
        fork_id = forked.get_session_id()
        return _apply_outcome(
            base, outcome, status="succeeded", fork_session_id=fork_id, n_messages=len(messages)
        )
    except Exception as exc:  # noqa: BLE001 -- one rollout must not sink the batch
        logger.debug("Rollout failed for {}: {}", ref.trajectory_id, exc)
        return _with_error(base, f"{type(exc).__name__}: {exc}")


async def continue_outcome(
    ref: TrajectoryRef, *, store: SessionStore, adapter: ScenarioAdapter
) -> ScoredOutcome:
    """Judge the baseline trajectory's own final submission (the CONTINUE ref)."""

    messages = load_trajectory_messages(ref, store=store)
    return await adapter.judge(messages, ref)


# --- helpers ---------------------------------------------------------------


def _source_messages_for_env(store: SessionStore, ref: TrajectoryRef) -> list[AgentMessage]:
    """Load source messages for environment setup (best-effort)."""
    try:
        return load_trajectory_messages(ref, store=store)
    except Exception as exc:
        logger.debug("Failed to load source messages for {}: {}", ref.trajectory_id, exc)
        return []


def _base_unit(
    ref: TrajectoryRef, prefix: PrefixPoint, treatment: Treatment, seed: int
) -> EvalUnit:
    return EvalUnit(
        case_id=ref.case_id,
        repository_id=ref.repository_id,
        trajectory_id=ref.trajectory_id,
        prefix_id=prefix.prefix_id,
        fork_point=prefix.fork_point,
        progress=prefix.progress,
        treatment_id=treatment.treatment_id,
        content_level=treatment.content_level,
        action=treatment.action,
        intervention=treatment.intervention,
        rung=treatment.rung,
        branch_seed=seed,
        actor_id=ref.actor_id,
        remaining_budget=dict(prefix.remaining_budget),
        sampling_weight=prefix.weight,
        metadata={"stratum": prefix.stratum},
    )


def _apply_outcome(
    base: EvalUnit,
    outcome: ScoredOutcome,
    *,
    status: str,
    fork_session_id: str | None,
    n_messages: int,
) -> EvalUnit:
    from dataclasses import replace

    return replace(
        base,
        status=status,
        fork_session_id=fork_session_id,
        binary_success=outcome.binary_success,
        normalized_score=outcome.normalized_score,
        judge_detail=outcome.detail,
        error=outcome.error,
        cost={"final_message_count": n_messages},
    )


def _with_status(base: EvalUnit, status: str, *, error: str | None = None) -> EvalUnit:
    from dataclasses import replace

    return replace(base, status=status, error=error)


def _with_error(base: EvalUnit, error: str) -> EvalUnit:
    return _with_status(base, "failed", error=error)


def _experiment_payload(
    ref: TrajectoryRef, prefix: PrefixPoint, treatment: Treatment, seed: int
) -> dict[str, Any]:
    return {
        "kind": "rescue_window_rollout",
        "id": f"{prefix.prefix_id}:{treatment.treatment_id}:{seed}",
        "variant": treatment.treatment_id,
        "case_id": ref.case_id,
        "trajectory_id": ref.trajectory_id,
        "prefix_id": prefix.prefix_id,
        "treatment_id": treatment.treatment_id,
        "content_level": treatment.content_level.value,
        "branch_seed": seed,
        "insert_turn_index": prefix.turn_index,
        "reminder_text": treatment.intervention.message,
    }


@contextlib.asynccontextmanager
async def env_session_scope() -> Any:
    """Serialize code that may read or temporarily replace ``os.environ``."""

    if _ENV_SESSION_HELD.get():
        yield
        return
    async with _ENV_SESSION_LOCK:
        token = _ENV_SESSION_HELD.set(True)
        try:
            yield
        finally:
            _ENV_SESSION_HELD.reset(token)


def default_store(cwd: str | None = None) -> SessionStore:
    return make_default_session_store(cwd or str(Path.cwd()))


def _stored_scenario(stored: dict[str, Any]) -> str:
    scenario = stored.get("scenario")
    if not isinstance(scenario, str) or not scenario:
        raise RuntimeError("source session has no stored scenario")
    return scenario


def _stored_provider(stored: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    provider = stored.get("provider")
    if (
        not isinstance(provider, list)
        or len(provider) != 2
        or not isinstance(provider[0], str)
        or not isinstance(provider[1], dict)
    ):
        raise RuntimeError("source session has no stored provider")
    return provider[0], dict(provider[1])


def _stored_agentm_env(stored: dict[str, Any]) -> dict[str, str]:
    env = stored.get("env")
    if not isinstance(env, dict):
        return {}
    return {
        key: value
        for key, value in env.items()
        if key.startswith("AGENTM_") and isinstance(value, str)
    }


@contextlib.contextmanager
def _temporary_env(values: dict[str, str]) -> Any:
    previous: dict[str, str | None] = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, old_value in previous.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value
