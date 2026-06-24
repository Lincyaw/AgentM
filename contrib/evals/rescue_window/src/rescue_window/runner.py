"""Generic same-prefix branch runner for AgentM sessions."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from agentm.core.abi import AgentMessage, AgentSessionConfig
from agentm.core.abi.loop import LoopConfig
from agentm.core.abi.session_store import SessionStore
from agentm.core.runtime import AgentSession
from agentm.core.runtime.session_bootstrap import make_default_session_store

from .schema import BranchSpec, ForkPoint, InterventionDecision

_ENV_SESSION_LOCK = asyncio.Lock()
_ENV_SESSION_HELD: ContextVar[bool] = ContextVar(
    "rescue_window_env_session_held",
    default=False,
)


@dataclass(frozen=True)
class BranchRunConfig:
    """Runtime knobs for one branch rollout."""

    max_turns: int | None = None
    max_tool_calls: int | None = None
    cwd: str | None = None
    extra_extensions: tuple[tuple[str, dict[str, Any]], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BranchResult:
    """Generic branch rollout result; scenario scoring is added elsewhere."""

    branch_id: str
    source_session_id: str
    fork_point: dict[str, Any]
    policy_id: str
    intervention: dict[str, Any]
    status: str
    fork_session_id: str | None = None
    error: str | None = None
    final_message_count: int = 0
    baseline_session_id: str | None = None
    trajectory_id: str | None = None
    case_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "branch_id": self.branch_id,
            "source_session_id": self.source_session_id,
            "fork_point": self.fork_point,
            "policy_id": self.policy_id,
            "intervention": self.intervention,
            "status": self.status,
            "fork_session_id": self.fork_session_id,
            "error": self.error,
            "final_message_count": self.final_message_count,
            "baseline_session_id": self.baseline_session_id,
            "trajectory_id": self.trajectory_id,
            "case_id": self.case_id,
            "metadata": self.metadata,
        }


async def run_branch(
    *,
    source_session_id: str,
    fork_point: ForkPoint,
    decision: InterventionDecision,
    branch_id: str,
    store: SessionStore | None = None,
    config: BranchRunConfig | None = None,
    baseline_session_id: str | None = None,
    trajectory_id: str | None = None,
    case_id: str | None = None,
) -> tuple[BranchResult, list[AgentMessage]]:
    """Fork a persisted AgentM session, inject an intervention, and continue."""

    run_config = config or BranchRunConfig()
    fork_selector = fork_point.to_dict()
    if not decision.should_intervene or not decision.intervention.message:
        return (
            BranchResult(
                branch_id=branch_id,
                source_session_id=source_session_id,
                fork_point=fork_selector,
                policy_id=decision.policy_id,
                intervention=decision.intervention.to_dict(),
                status="skipped",
                baseline_session_id=baseline_session_id,
                trajectory_id=trajectory_id,
                case_id=case_id,
                metadata={**run_config.metadata, **decision.metadata},
            ),
            [],
        )

    try:
        async with env_session_scope():
            resolved_store = store or make_default_session_store(
                _store_cwd(run_config.cwd)
            )
            source = resolved_store.open(source_session_id)
            header = source.get_header()
            if header is None:
                raise RuntimeError("source session has no header")
            stored = dict(header.config or {})
            scenario = _stored_scenario(stored)
            provider = _stored_provider(stored)
            cwd = run_config.cwd or header.cwd or os.getcwd()
            forked = resolved_store.fork(source_session_id, **fork_selector)
            env = _stored_agentm_env(stored)
            with _temporary_env(env):
                experiment = _branch_experiment_payload(
                    branch_id=branch_id,
                    policy_id=decision.policy_id,
                    baseline_session_id=baseline_session_id,
                    trajectory_id=trajectory_id,
                    case_id=case_id,
                    fork_selector=fork_selector,
                    decision=decision,
                    run_config=run_config,
                )
                session = await AgentSession.create(
                    AgentSessionConfig(
                        cwd=cwd,
                        session_manager=cast(Any, forked),
                        scenario=scenario,
                        provider=provider,
                        extra_extensions=list(run_config.extra_extensions),
                        loop_config=LoopConfig(
                            max_turns=run_config.max_turns,
                            max_tool_calls=run_config.max_tool_calls,
                        ),
                        parent_session_id=source_session_id,
                        lineage={
                            "kind": "fork",
                            "entrypoint": "rescue_window.branch_runner",
                            "source_session_id": source_session_id,
                            "fork_point": fork_selector,
                        },
                        experiment=experiment,
                    ),
                )
                messages: list[AgentMessage] = []
                try:
                    await session.prompt(decision.intervention.message)
                    await session.idle(timeout=30)
                    messages = session.session_manager.get_messages()
                finally:
                    await session.shutdown()

        return (
            BranchResult(
                branch_id=branch_id,
                source_session_id=source_session_id,
                fork_point=fork_selector,
                policy_id=decision.policy_id,
                intervention=decision.intervention.to_dict(),
                status="succeeded",
                fork_session_id=forked.get_session_id(),
                final_message_count=len(messages),
                baseline_session_id=baseline_session_id,
                trajectory_id=trajectory_id,
                case_id=case_id,
                metadata={**run_config.metadata, **decision.metadata},
            ),
            messages,
        )
    except Exception as exc:  # noqa: BLE001 -- one branch must not sink a batch
        return (
            BranchResult(
                branch_id=branch_id,
                source_session_id=source_session_id,
                fork_point=fork_selector,
                policy_id=decision.policy_id,
                intervention=decision.intervention.to_dict(),
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
                baseline_session_id=baseline_session_id,
                trajectory_id=trajectory_id,
                case_id=case_id,
                metadata={**run_config.metadata, **decision.metadata},
            ),
            [],
        )


async def run_branch_spec(
    spec: BranchSpec,
    *,
    store: SessionStore | None = None,
) -> tuple[BranchResult, list[AgentMessage]]:
    """Run one explicit-intervention branch spec."""

    decision = InterventionDecision(
        policy_id=spec.policy_id,
        intervention=spec.intervention,
        should_intervene=bool(spec.intervention.message),
        reason="spec",
    )
    return await run_branch(
        source_session_id=spec.source_session_id,
        fork_point=spec.fork_point,
        decision=decision,
        branch_id=spec.branch_id,
        store=store,
        config=BranchRunConfig(
            max_turns=spec.max_turns,
            max_tool_calls=spec.max_tool_calls,
            cwd=spec.cwd,
            metadata=spec.metadata,
        ),
        baseline_session_id=spec.baseline_session_id,
        trajectory_id=spec.trajectory_id,
        case_id=spec.case_id,
    )


async def run_specs(
    specs: list[BranchSpec],
    *,
    out_jsonl: Path,
    concurrency: int = 1,
) -> list[BranchResult]:
    """Run branch specs and write one JSONL record per result."""

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if out_jsonl.exists():
        out_jsonl.unlink()
    sem = asyncio.Semaphore(max(1, concurrency))
    write_lock = asyncio.Lock()
    results: list[BranchResult] = []

    async def _run_one(spec: BranchSpec) -> None:
        async with sem:
            result, _ = await run_branch_spec(spec)
            async with write_lock:
                with out_jsonl.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(result.to_json_dict(), ensure_ascii=False) + "\n")
                results.append(result)

    await asyncio.gather(*[_run_one(spec) for spec in specs])
    return results


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


def _store_cwd(cwd: str | None) -> str:
    return str(Path(cwd).expanduser()) if cwd else str(Path.cwd())


def _branch_experiment_payload(
    *,
    branch_id: str,
    policy_id: str,
    baseline_session_id: str | None,
    trajectory_id: str | None,
    case_id: str | None,
    fork_selector: dict[str, Any],
    decision: InterventionDecision,
    run_config: BranchRunConfig,
) -> dict[str, Any]:
    metadata = {**run_config.metadata, **decision.metadata}
    payload: dict[str, Any] = {
        "kind": "rescue_window_branch",
        "id": branch_id,
        "branch_id": branch_id,
        "variant": branch_id,
        "policy_id": policy_id,
        "baseline_session_id": baseline_session_id,
        "trajectory_id": trajectory_id,
        "case_id": case_id,
        "reminder_id": branch_id,
        "reminder_text": decision.intervention.message,
        "intervention": decision.intervention.to_dict(),
        "metadata": metadata,
    }
    turn_index = fork_selector.get("turn_index")
    if isinstance(turn_index, int) and not isinstance(turn_index, bool):
        payload["insert_turn_index"] = turn_index
    message_id = fork_selector.get("message_id")
    if isinstance(message_id, str) and message_id:
        payload["insert_message_id"] = message_id
    return payload


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
