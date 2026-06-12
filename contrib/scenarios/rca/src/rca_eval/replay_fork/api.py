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
import os
from dataclasses import dataclass, field
from typing import Any

from agentm.core.abi import AgentMessage, AssistantMessage, ToolCallBlock
from loguru import logger

__all__ = [
    "ReplayResult",
    "ReplaySummary",
    "replay_batch",
    "replay_one",
]


@dataclass
class ReplayResult:
    case_id: str
    fired: bool
    control_correct: bool | None = None
    intervene_correct: bool | None = None
    surface_turn: int | None = None
    reminder: str | None = None
    forked_session_id: str | None = None
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
    for msg in messages:
        if not isinstance(msg, AssistantMessage):
            continue
        for block in msg.content:
            if isinstance(block, ToolCallBlock) and block.name == "submit_final_report":
                return block.arguments.get("text")
    return None


def _read_source_config(session_state: Any) -> dict[str, Any]:
    """Read stored session config from a session manager."""
    header = session_state.get_header()
    if header is None or header.config is None:
        return {}
    return dict(header.config)


async def replay_one(
    *,
    session_id: str,
    store: Any,
    harness_provider: tuple[str, dict[str, Any]],
    extractor_interval: int = 5,
    audit_interval: int = 5,
    auditor_prompt: str = "minimal",
    max_turns: int = 60,
    data_dir_override: str | None = None,
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

    # 2. Judge control
    judge = RcabenchJudge()
    control_response = _extract_submission(messages)
    if data_dir:
        ctrl = await judge.judge(
            agent_output_json=control_response,
            data_dir=data_dir,
            case_id=session_id,
        )
    else:
        ctrl = type("_", (), {"correct": None})()

    # 3. Offline audit
    try:
        audit_result = await offline_audit(
            messages,
            cwd=cwd,
            provider=harness_provider,
            extractor_interval=extractor_interval,
            audit_interval=audit_interval,
            auditor_prompt=auditor_prompt,
        )
        surfaces = audit_result.surfaces
        audit_firings = audit_result.firings
    except Exception as exc:
        logger.exception(f"offline_audit failed for {session_id}")
        return ReplayResult(
            case_id=session_id,
            fired=False,
            control_correct=ctrl.correct,
            error=f"audit failed: {exc}",
        )

    if not surfaces:
        return ReplayResult(
            case_id=session_id,
            fired=False,
            control_correct=ctrl.correct,
            intervene_correct=ctrl.correct,
            audit_firings=audit_firings,
        )

    # 4. Fork at first surface and continue
    s = surfaces[0]
    try:
        forked = store.fork(session_id, up_to=s.turn_index)
        config = AgentSessionConfig(
            cwd=cwd,
            session_manager=forked,
            scenario=scenario,
            provider=agent_provider,
            loop_config=LoopConfig(max_turns=max_turns),
        )
        session = await create_agent_session(AgentSession, config)
        try:
            fork_messages = await session.prompt(s.reminder_text)
        finally:
            await session.shutdown()
    except Exception as exc:
        logger.exception(f"fork continuation failed for {session_id}")
        return ReplayResult(
            case_id=session_id,
            fired=True,
            surface_turn=s.turn_index,
            reminder=s.reminder_text,
            control_correct=ctrl.correct,
            error=f"fork failed: {exc}",
        )

    # 5. Judge fork
    fork_response = _extract_submission(fork_messages)
    if data_dir:
        fork_outcome = await judge.judge(
            agent_output_json=fork_response,
            data_dir=data_dir,
            case_id=f"{session_id}-fork",
        )
        fork_correct = fork_outcome.correct
    else:
        fork_correct = None

    return ReplayResult(
        case_id=session_id,
        fired=True,
        surface_turn=s.turn_index,
        reminder=s.reminder_text,
        control_correct=ctrl.correct,
        intervene_correct=fork_correct,
        forked_session_id=forked.get_session_id(),
        audit_firings=audit_firings,
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
    auditor_prompt: str = "minimal",
    max_turns: int = 60,
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
