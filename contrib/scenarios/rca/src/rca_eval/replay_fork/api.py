"""Programmatic API for replay-fork experiments.

Two entry points:

- ``replay_one()`` — replay-fork a single baseline session
- ``replay_batch()`` — run many sessions with concurrency control

Session config (scenario, provider, data_dir) is auto-restored from
the source session's stored config — callers only specify the harness
model and strategy parameters::

    summary = await replay_batch(
        session_ids=["abc123", "def456"],
        store=store,
        harness_provider=build_profile_provider("doubao"),
        auditor_prompt="trajectory_coverage",
        concurrency=10,
    )
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentm.core.abi import AgentMessage, AssistantMessage, ToolCallBlock
from loguru import logger

__all__ = [
    "ReplayForkAttempt",
    "ReplayResult",
    "ReplaySummary",
    "replay_batch",
    "replay_one",
]


@dataclass
class ReplayForkAttempt:
    generation: int
    source_session_id: str
    source_correct: bool | None = None
    source_submission_summary: dict[str, Any] | None = None
    surface_turn: int | None = None
    reminder: str | None = None
    forked_session_id: str | None = None
    fork_correct: bool | None = None
    fork_submission_summary: dict[str, Any] | None = None
    audit_firings: list[Any] = field(default_factory=list)
    error: str | None = None


@dataclass
class ReplayResult:
    case_id: str
    fired: bool
    control_correct: bool | None = None
    intervene_correct: bool | None = None
    control_submission_summary: dict[str, Any] | None = None
    intervene_submission_summary: dict[str, Any] | None = None
    surface_turn: int | None = None
    reminder: str | None = None
    forked_session_id: str | None = None
    fork_attempts: list[ReplayForkAttempt] = field(default_factory=list)
    audit_firings: list[Any] = field(default_factory=list)
    error: str | None = None

    @property
    def helped(self) -> bool:
        return self.control_correct is False and self.intervene_correct is True

    @property
    def harmed(self) -> bool:
        return self.control_correct is True and self.intervene_correct is False


@dataclass
class ReplaySummary:
    results: list[ReplayResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def fired(self) -> int:
        return sum(1 for r in self.results if r.fired)

    @property
    def errored(self) -> int:
        return sum(1 for r in self.results if r.error)

    @property
    def control_correct(self) -> int:
        return sum(1 for r in self.results if r.control_correct)

    @property
    def intervene_correct(self) -> int:
        return sum(1 for r in self.results if r.intervene_correct)

    @property
    def helped(self) -> int:
        return sum(1 for r in self.results if r.helped)

    @property
    def harmed(self) -> int:
        return sum(1 for r in self.results if r.harmed)

    def format(self) -> str:
        def pct(n: int) -> str:
            return f"{100.0 * n / self.total:.1f}%" if self.total else "n/a"

        return (
            f"cases={self.total} fired={self.fired} errored={self.errored}\n"
            f"control  correct: {self.control_correct} ({pct(self.control_correct)})\n"
            f"intervene correct: {self.intervene_correct} ({pct(self.intervene_correct)})\n"
            f"flips  W->R(helped): {self.helped}   R->W(harmed): {self.harmed}"
        )


def _extract_submission(messages: list[AgentMessage]) -> str | None:
    candidates: list[tuple[float, int, bool, str]] = []
    order = 0
    for msg in messages:
        if not isinstance(msg, AssistantMessage):
            continue
        for block in msg.content:
            if isinstance(block, ToolCallBlock) and block.name == "submit_final_report":
                text = block.arguments.get("text")
                if isinstance(text, str):
                    payload = text
                else:
                    payload = json.dumps(block.arguments, ensure_ascii=False, default=str)
                is_complete = (
                    "nodes" in block.arguments
                    and "root_causes" in block.arguments
                )
                candidates.append((msg.timestamp, order, is_complete, payload))
                order += 1
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    complete = [item for item in candidates if item[2]]
    return (complete or candidates)[-1][3]


def _summarize_submission(raw: str | None) -> dict[str, Any]:
    """Summarize an RCA final report without storing the full evidence text."""
    if raw is None:
        return {"present": False, "parse_error": "missing"}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {
            "present": True,
            "parse_error": f"invalid_json: {exc.msg}",
        }
    if not isinstance(payload, dict):
        return {
            "present": True,
            "parse_error": f"expected_object: {type(payload).__name__}",
        }

    missing = [
        key for key in ("nodes", "root_causes") if key not in payload
    ]
    nodes_raw = payload.get("nodes")
    edges_raw = payload.get("edges")
    roots_raw = payload.get("root_causes")
    nodes = nodes_raw if isinstance(nodes_raw, list) else []
    edges = edges_raw if isinstance(edges_raw, list) else []
    roots = roots_raw if isinstance(roots_raw, list) else []

    by_id: dict[str, dict[str, Any]] = {}
    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_id = node.get("id")
        if isinstance(node_id, str):
            by_id[node_id] = node

    root_summaries: list[dict[str, Any]] = []
    for root in roots:
        root_id = root.get("id") if isinstance(root, dict) else root
        if not isinstance(root_id, str):
            continue
        node = by_id.get(root_id, {})
        item: dict[str, Any] = {"id": root_id}
        for key in ("subject", "predicate", "confidence"):
            value = node.get(key)
            if isinstance(value, str | int | float | bool):
                item[key] = value
        interval = node.get("interval")
        if isinstance(interval, dict):
            item["interval"] = {
                k: v for k, v in interval.items() if isinstance(k, str) and isinstance(v, str)
            }
        root_summaries.append(item)

    return {
        "present": True,
        "parse_error": (
            f"missing_fields: {', '.join(missing)}" if missing else None
        ),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "root_count": len(roots),
        "root_causes": root_summaries,
    }


def _read_source_config(session_state: Any) -> dict[str, Any]:
    """Read stored session config from a session manager."""
    header = session_state.get_header()
    if header is None or header.config is None:
        return {}
    return dict(header.config)


def _fork_session_manager(source: Any, *, up_to: int) -> Any:
    """Fork an already-open session manager at a message prefix.

    ``SessionStore.fork()`` re-opens by session id. That is fine for the
    baseline, but multi-generation replay may fork from an in-memory child
    before the observability backend can be queried. Fork directly from the
    current manager so the chain is deterministic inside this process.
    """
    from agentm.core.runtime.session_manager import SessionManager

    source_id = source.get_session_id()
    cwd = source._cwd or os.getcwd()
    forked = SessionManager(cwd=cwd, persist=False, parent_session=source_id)
    header = source.get_header()
    if header is not None and header.config:
        forked.set_session_config(copy.deepcopy(header.config))
    for msg in source.get_raw_messages()[:up_to]:
        forked.append_message(msg)
    return forked


async def _judge_response(
    *,
    judge: Any,
    response: str | None,
    data_dir: str | None,
    case_id: str,
) -> Any:
    if data_dir:
        return await judge.judge(
            agent_output_json=response,
            data_dir=data_dir,
            case_id=case_id,
        )
    return type("_", (), {"correct": None, "detail": {}, "error": None})()


async def replay_one(
    *,
    session_id: str,
    store: Any,
    harness_provider: tuple[str, dict[str, Any]],
    extractor_interval: int = 5,
    audit_interval: int = 5,
    auditor_prompt: str = "minimal_index",
    max_turns: int = 60,
    data_dir_override: str | None = None,
    fork_live_harness: bool = False,
    max_forks: int = 1,
) -> ReplayResult:
    """Replay-fork one baseline session.

    Scenario, provider, and data_dir are auto-restored from the source
    session's stored config. Only the harness model (for extractor/auditor)
    and strategy parameters need to be specified.
    """
    from agentm.core.abi import AgentSessionConfig
    from agentm.core.abi.loop import LoopConfig
    from agentm.core.runtime import AgentSession, create_agent_session
    from llmharness import offline_audit

    from .judge import RcabenchJudge

    # 1. Open source session and read stored config
    try:
        source = store.open(session_id)
        messages = source.get_raw_messages()
    except Exception as exc:
        return ReplayResult(
            case_id=session_id, fired=False, error=f"open failed: {exc}"
        )

    stored = _read_source_config(source)
    scenario = stored.get("scenario")
    if not scenario:
        return ReplayResult(
            case_id=session_id, fired=False,
            error="source session has no stored scenario config",
        )

    stored_provider = stored.get("provider")
    if not stored_provider or not isinstance(stored_provider, list) or len(stored_provider) != 2:
        return ReplayResult(
            case_id=session_id, fired=False,
            error="source session has no stored provider config",
        )
    agent_provider = (stored_provider[0], stored_provider[1])

    cwd = source._cwd or os.getcwd()

    # Restore AGENTM_* env vars from stored config
    stored_env = stored.get("env")
    if isinstance(stored_env, dict):
        for k, v in stored_env.items():
            if k.startswith("AGENTM_") and isinstance(v, str):
                os.environ.setdefault(k, v)

    data_dir = data_dir_override or _find_data_dir_from_config(stored)
    if data_dir:
        os.environ["AGENTM_RCA_DATA_DIR"] = data_dir
    rca_case_id = Path(data_dir).name if data_dir else session_id

    # 2. Judge control
    judge = RcabenchJudge()
    control_response = _extract_submission(messages)
    ctrl = await _judge_response(
        judge=judge,
        response=control_response,
        data_dir=data_dir,
        case_id=session_id,
    )
    control_summary = _summarize_submission(control_response)

    attempts: list[ReplayForkAttempt] = []
    current_manager = source
    current_session_id = session_id
    current_messages = messages
    current_correct = ctrl.correct
    current_summary = control_summary
    first_surface_turn: int | None = None
    first_reminder: str | None = None
    first_audit_firings: list[Any] = []
    final_error: str | None = None

    for generation in range(1, max(1, max_forks) + 1):
        if current_correct is True:
            break

        try:
            audit_result = await offline_audit(
                current_messages,
                cwd=cwd,
                provider=harness_provider,
                extractor_interval=extractor_interval,
                audit_interval=audit_interval,
                auditor_prompt=auditor_prompt,
                stop_on_first_surface=True,
            )
            surfaces = audit_result.surfaces
            audit_firings = audit_result.firings
        except Exception as exc:
            logger.exception(f"offline_audit failed for {current_session_id}")
            final_error = f"fork {generation} audit failed: {exc}"
            attempts.append(
                ReplayForkAttempt(
                    generation=generation,
                    source_session_id=current_session_id,
                    source_correct=current_correct,
                    source_submission_summary=current_summary,
                    error=final_error,
                )
            )
            break

        if generation == 1:
            first_audit_firings = audit_firings

        if not surfaces:
            attempts.append(
                ReplayForkAttempt(
                    generation=generation,
                    source_session_id=current_session_id,
                    source_correct=current_correct,
                    source_submission_summary=current_summary,
                    audit_firings=audit_firings,
                    error="no surface",
                )
            )
            break

        s = surfaces[0]
        if first_surface_turn is None:
            first_surface_turn = s.turn_index
            first_reminder = s.reminder_text

        try:
            forked = _fork_session_manager(current_manager, up_to=s.turn_index)
            extra_extensions: list[tuple[str, dict[str, Any]]] = []
            if fork_live_harness:
                provider_module, provider_config = harness_provider
                extra_extensions.append(
                    (
                        "llmharness.atom",
                        {
                            "mode": "sync",
                            "extractor_interval_turns": extractor_interval,
                            "audit_interval_turns": audit_interval,
                            "enable_reminders": True,
                            "extractor_provider": {
                                "module": provider_module,
                                "config": dict(provider_config),
                            },
                            "auditor_provider": {
                                "module": provider_module,
                                "config": dict(provider_config),
                            },
                            "auditor_prompt": auditor_prompt,
                            "finalize_tool": "submit_final_report",
                        },
                    )
                )
            config = AgentSessionConfig(
                cwd=cwd,
                session_manager=forked,
                scenario=scenario,
                provider=agent_provider,
                extra_extensions=extra_extensions,
                loop_config=LoopConfig(max_turns=max_turns),
                parent_session_id=current_session_id,
                lineage={
                    "kind": "fork",
                    "entrypoint": "rca.replay_fork",
                    "source_session_id": current_session_id,
                    "fork_point": {"turn_index": s.turn_index},
                    "generation": generation,
                },
                experiment={
                    "kind": "reminder_injection",
                    "case_id": rca_case_id,
                    "baseline_session_id": session_id,
                    "source_session_id": current_session_id,
                    "generation": generation,
                    "insert_turn_index": s.turn_index,
                    "reminder_text": s.reminder_text,
                },
            )
            session = await create_agent_session(AgentSession, config)
            try:
                await session.prompt(s.reminder_text)
            finally:
                await session.shutdown()
            fork_messages = forked.get_raw_messages()
        except Exception as exc:
            logger.exception(f"fork continuation failed for {current_session_id}")
            final_error = f"fork {generation} failed: {exc}"
            attempts.append(
                ReplayForkAttempt(
                    generation=generation,
                    source_session_id=current_session_id,
                    source_correct=current_correct,
                    source_submission_summary=current_summary,
                    surface_turn=s.turn_index,
                    reminder=s.reminder_text,
                    audit_firings=audit_firings,
                    error=final_error,
                )
            )
            break

        fork_response = _extract_submission(fork_messages)
        fork_summary = _summarize_submission(fork_response)
        fork_outcome = await _judge_response(
            judge=judge,
            response=fork_response,
            data_dir=data_dir,
            case_id=f"{session_id}-fork-{generation}",
        )
        attempts.append(
            ReplayForkAttempt(
                generation=generation,
                source_session_id=current_session_id,
                source_correct=current_correct,
                source_submission_summary=current_summary,
                surface_turn=s.turn_index,
                reminder=s.reminder_text,
                forked_session_id=forked.get_session_id(),
                fork_correct=fork_outcome.correct,
                fork_submission_summary=fork_summary,
                audit_firings=audit_firings,
            )
        )

        current_manager = forked
        current_session_id = forked.get_session_id()
        current_messages = fork_messages
        current_correct = fork_outcome.correct
        current_summary = fork_summary

    return ReplayResult(
        case_id=session_id,
        fired=any(attempt.surface_turn is not None for attempt in attempts),
        surface_turn=first_surface_turn,
        reminder=first_reminder,
        control_correct=ctrl.correct,
        intervene_correct=current_correct,
        control_submission_summary=control_summary,
        intervene_submission_summary=current_summary,
        forked_session_id=(
            current_session_id if current_session_id != session_id else None
        ),
        fork_attempts=attempts,
        audit_firings=first_audit_firings,
        error=final_error,
    )


def _find_data_dir_from_config(stored: dict[str, Any]) -> str | None:
    """Extract data_dir from the stored session config's env snapshot."""
    env = stored.get("env")
    if isinstance(env, dict):
        val = env.get("AGENTM_RCA_DATA_DIR")
        if isinstance(val, str) and val:
            return val
    return os.environ.get("AGENTM_RCA_DATA_DIR")


async def replay_batch(
    session_ids: list[str],
    *,
    store: Any,
    harness_provider: tuple[str, dict[str, Any]],
    extractor_interval: int = 5,
    audit_interval: int = 5,
    auditor_prompt: str = "minimal_index",
    max_turns: int = 60,
    fork_live_harness: bool = False,
    max_forks: int = 1,
    concurrency: int = 1,
    on_result: Any | None = None,
) -> ReplaySummary:
    """Run replay-fork over many sessions.

    Only ``session_ids`` and ``harness_provider`` are required — scenario,
    agent provider, and data_dir are auto-restored from each session's
    stored config.

    ``on_result`` is an optional callback ``(result, index, total) -> None``
    for progress reporting.
    """
    summary = ReplaySummary()
    total = len(session_ids)
    sem = asyncio.Semaphore(max(1, concurrency))
    done_count = 0

    async def _run_one(sid: str) -> ReplayResult:
        nonlocal done_count
        async with sem:
            result = await replay_one(
                session_id=sid,
                store=store,
                harness_provider=harness_provider,
                extractor_interval=extractor_interval,
                audit_interval=audit_interval,
                auditor_prompt=auditor_prompt,
                max_turns=max_turns,
                fork_live_harness=fork_live_harness,
                max_forks=max_forks,
            )
            done_count += 1
            if on_result is not None:
                on_result(result, done_count, total)
            return result

    tasks = [asyncio.create_task(_run_one(sid)) for sid in session_ids]
    for fut in asyncio.as_completed(tasks):
        result = await fut
        summary.results.append(result)

    return summary
