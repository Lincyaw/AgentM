"""Run intervention policies over persisted AgentM session prefixes."""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, cast

from agentm.core.abi import AgentMessage
from agentm.core.abi.session_store import SessionState, SessionStore
from agentm.core.runtime.session_bootstrap import make_default_session_store

from .policies import InterventionPolicy, PolicyContext
from .runner import BranchResult, BranchRunConfig, env_session_scope, run_branch
from .schema import ActionType, ForkPoint, Intervention, InterventionDecision


@dataclass(frozen=True)
class PolicyRunConfig:
    """Runtime knobs for policy-driven branch rollout."""

    max_turns: int | None = 60
    max_tool_calls: int | None = None
    cwd: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


ProgressCallback = Callable[[BranchResult, int, int], None]


async def run_policy_for_session(
    *,
    source_session_id: str,
    policy: InterventionPolicy,
    store: SessionStore | None = None,
    config: PolicyRunConfig | None = None,
) -> BranchResult:
    """Audit one source session prefix, fork at the surfaced point, and continue."""

    run_config = config or PolicyRunConfig()
    async with env_session_scope():
        resolved_store = store or make_default_session_store(str(Path.cwd()))
        try:
            source = resolved_store.open(source_session_id)
            header = source.get_header()
            stored = dict(header.config or {}) if header and header.config else {}
            cwd = run_config.cwd or (header.cwd if header else None) or os.getcwd()
            messages = _source_messages(source)
            case_id = _case_id_from_config(stored)
            decision = await policy.decide(
                PolicyContext(
                    source_session_id=source_session_id,
                    messages=messages,
                    cwd=cwd,
                    provider=_stored_provider(stored),
                    trajectory_id=_trajectory_id(stored),
                    metadata={**run_config.metadata, "case_id": case_id},
                )
            )
            fork_turn_index = _decision_surface_turn_index(decision)
            branch_id = _branch_id(
                source_session_id=source_session_id,
                policy_id=decision.policy_id,
                case_id=case_id,
                fork_turn_index=fork_turn_index,
            )
            if not decision.should_intervene or not decision.intervention.message:
                return BranchResult(
                    branch_id=branch_id,
                    source_session_id=source_session_id,
                    fork_point={},
                    policy_id=decision.policy_id,
                    intervention=decision.intervention.to_dict(),
                    status="skipped",
                    baseline_session_id=source_session_id,
                    trajectory_id=_trajectory_id(stored),
                    case_id=case_id,
                    metadata={
                        **run_config.metadata,
                        **decision.metadata,
                        "reason": decision.reason,
                    },
                )
            if fork_turn_index is None:
                return BranchResult(
                    branch_id=branch_id,
                    source_session_id=source_session_id,
                    fork_point={},
                    policy_id=decision.policy_id,
                    intervention=decision.intervention.to_dict(),
                    status="failed",
                    error="policy decision missing integer surface_turn_index metadata",
                    baseline_session_id=source_session_id,
                    trajectory_id=_trajectory_id(stored),
                    case_id=case_id,
                    metadata={**run_config.metadata, **decision.metadata},
                )
            result, _ = await run_branch(
                source_session_id=source_session_id,
                fork_point=ForkPoint(turn_index=fork_turn_index),
                decision=decision,
                branch_id=branch_id,
                store=resolved_store,
                config=BranchRunConfig(
                    max_turns=run_config.max_turns,
                    max_tool_calls=run_config.max_tool_calls,
                    cwd=run_config.cwd,
                    metadata=run_config.metadata,
                ),
                baseline_session_id=source_session_id,
                trajectory_id=_trajectory_id(stored),
                case_id=case_id,
            )
            return result
        except Exception as exc:  # noqa: BLE001 -- one source must not sink a batch
            return BranchResult(
                branch_id=_branch_id(
                    source_session_id=source_session_id,
                    policy_id=getattr(policy, "policy_id", "unknown_policy"),
                    case_id=None,
                    fork_turn_index=None,
                ),
                source_session_id=source_session_id,
                fork_point={},
                policy_id=getattr(policy, "policy_id", "unknown_policy"),
                intervention=_empty_intervention().to_dict(),
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
                baseline_session_id=source_session_id,
                metadata=run_config.metadata,
            )


async def run_policy_over_sessions(
    source_session_ids: list[str],
    *,
    policy: InterventionPolicy,
    out_jsonl: Path,
    store: SessionStore | None = None,
    config: PolicyRunConfig | None = None,
    concurrency: int = 1,
    append: bool = False,
    on_result: ProgressCallback | None = None,
) -> list[BranchResult]:
    """Run a policy over many source sessions and write branch result JSONL."""

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if out_jsonl.exists() and not append:
        out_jsonl.unlink()
    resolved_store = store or make_default_session_store(str(Path.cwd()))
    sem = asyncio.Semaphore(max(1, concurrency))
    write_lock = asyncio.Lock()
    results: list[BranchResult] = []
    total = len(source_session_ids)
    done = 0

    async def _run_one(source_session_id: str) -> BranchResult:
        async with sem:
            return await run_policy_for_session(
                source_session_id=source_session_id,
                policy=policy,
                store=resolved_store,
                config=config,
            )

    tasks = [asyncio.create_task(_run_one(sid)) for sid in source_session_ids]
    for fut in asyncio.as_completed(tasks):
        result = await fut
        async with write_lock:
            with out_jsonl.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(result.to_json_dict(), ensure_ascii=False) + "\n")
            results.append(result)
            done += 1
            if on_result is not None:
                on_result(result, done, total)
    return results


def _source_messages(source: SessionState) -> list[AgentMessage]:
    raw_getter = getattr(source, "get_raw_messages", None)
    if callable(raw_getter):
        return list(cast(list[AgentMessage], raw_getter()))
    return list(source.build_session_context().messages)


def _stored_provider(stored: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    provider = stored.get("provider")
    if (
        isinstance(provider, list)
        and len(provider) == 2
        and isinstance(provider[0], str)
        and isinstance(provider[1], dict)
    ):
        return provider[0], dict(provider[1])
    return None


def _trajectory_id(stored: dict[str, Any]) -> str | None:
    trace = stored.get("trace")
    if isinstance(trace, dict) and isinstance(trace.get("trace_id"), str):
        return trace["trace_id"]
    return None


def _case_id_from_config(stored: dict[str, Any]) -> str | None:
    experiment = stored.get("experiment")
    if isinstance(experiment, dict) and isinstance(experiment.get("case_id"), str):
        return experiment["case_id"]
    return None


def _decision_surface_turn_index(decision: InterventionDecision) -> int | None:
    for container in (decision.metadata, decision.intervention.metadata):
        value = container.get("surface_turn_index")
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _branch_id(
    *,
    source_session_id: str,
    policy_id: str,
    case_id: str | None,
    fork_turn_index: int | None,
) -> str:
    suffix = f"turn{fork_turn_index}" if fork_turn_index is not None else "no-fork"
    return "-".join(
        [
            _slug(case_id or source_session_id[:12]),
            _slug(policy_id),
            suffix,
        ]
    )


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return slug or "branch"


def _empty_intervention() -> Intervention:
    return Intervention(
        action=ActionType.CONTINUE,
        condition_id="POLICY_ERROR",
        content_level="ERROR",
    )
