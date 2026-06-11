"""Programmatic API for replay-fork experiments.

Two entry points:

- ``replay_one()`` — replay-fork a single baseline session
- ``replay_batch()`` — run many cases with concurrency control

Designed for eval scripts, notebooks, and strategy sweeps::

    results = await replay_batch(
        cases=[("sid1", "data/case1"), ("sid2", "data/case2")],
        store=store,
        harness_provider=harness_provider,
        agent_provider=agent_provider,
        auditor_prompt="trajectory_coverage",
        concurrency=4,
    )
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from agentm.core.abi import AgentMessage, AssistantMessage, ToolCallBlock

_log = logging.getLogger(__name__)

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


async def replay_one(
    *,
    session_id: str,
    data_dir: str,
    store: Any,
    harness_provider: tuple[str, dict[str, Any]],
    agent_provider: tuple[str, dict[str, Any]],
    scenario: str = "rca:baseline",
    cwd: str | None = None,
    extractor_interval: int = 5,
    audit_interval: int = 5,
    auditor_prompt: str = "minimal",
    max_turns: int = 60,
) -> ReplayResult:
    """Replay-fork one baseline session.

    1. Read baseline messages from ``store``
    2. Run ``offline_audit`` to find surface points
    3. Fork at first surface, continue with reminder, judge
    """
    import os

    from agentm.core.abi import AgentSessionConfig
    from agentm.core.abi.loop import LoopConfig
    from agentm.core.runtime import AgentSession, create_agent_session
    from llmharness import offline_audit

    from .judge import RcabenchJudge

    resolved_cwd = cwd or os.getcwd()
    os.environ["AGENTM_RCA_DATA_DIR"] = data_dir

    try:
        source = store.open(session_id)
        messages = source.get_raw_messages()
    except Exception as exc:
        return ReplayResult(
            case_id=session_id, fired=False, error=f"open failed: {exc}"
        )

    # Judge control
    judge = RcabenchJudge()
    control_response = _extract_submission(messages)
    ctrl = await judge.judge(
        agent_output_json=control_response,
        data_dir=data_dir,
        case_id=session_id,
    )

    # Offline audit
    try:
        surfaces = await offline_audit(
            messages,
            cwd=resolved_cwd,
            provider=harness_provider,
            extractor_interval=extractor_interval,
            audit_interval=audit_interval,
            auditor_prompt=auditor_prompt,
        )
    except Exception as exc:
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
        )

    # Fork at first surface
    s = surfaces[0]
    try:
        forked = store.fork(session_id, up_to=s.turn_index)
        config = AgentSessionConfig(
            cwd=resolved_cwd,
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
        return ReplayResult(
            case_id=session_id,
            fired=True,
            surface_turn=s.turn_index,
            reminder=s.reminder_text,
            control_correct=ctrl.correct,
            error=f"fork failed: {exc}",
        )

    fork_response = _extract_submission(fork_messages)
    fork_outcome = await judge.judge(
        agent_output_json=fork_response,
        data_dir=data_dir,
        case_id=f"{session_id}-fork",
    )

    return ReplayResult(
        case_id=session_id,
        fired=True,
        surface_turn=s.turn_index,
        reminder=s.reminder_text,
        control_correct=ctrl.correct,
        intervene_correct=fork_outcome.correct,
        forked_session_id=forked.get_session_id(),
    )


async def replay_batch(
    cases: list[tuple[str, str]],
    *,
    store: Any,
    harness_provider: tuple[str, dict[str, Any]],
    agent_provider: tuple[str, dict[str, Any]],
    scenario: str = "rca:baseline",
    cwd: str | None = None,
    extractor_interval: int = 5,
    audit_interval: int = 5,
    auditor_prompt: str = "minimal",
    max_turns: int = 60,
    concurrency: int = 1,
    on_result: Any | None = None,
) -> ReplaySummary:
    """Run replay-fork over many ``(session_id, data_dir)`` pairs.

    ``on_result`` is an optional callback ``(result, index, total) -> None``
    called after each case completes (for progress reporting / streaming
    to a JSONL sink).
    """
    summary = ReplaySummary()
    total = len(cases)
    sem = asyncio.Semaphore(max(1, concurrency))
    done_count = 0

    async def _run_one(sid: str, data_dir: str) -> ReplayResult:
        nonlocal done_count
        async with sem:
            result = await replay_one(
                session_id=sid,
                data_dir=data_dir,
                store=store,
                harness_provider=harness_provider,
                agent_provider=agent_provider,
                scenario=scenario,
                cwd=cwd,
                extractor_interval=extractor_interval,
                audit_interval=audit_interval,
                auditor_prompt=auditor_prompt,
                max_turns=max_turns,
            )
            done_count += 1
            if on_result is not None:
                on_result(result, done_count, total)
            return result

    tasks = [asyncio.create_task(_run_one(sid, dd)) for sid, dd in cases]
    for fut in asyncio.as_completed(tasks):
        result = await fut
        summary.results.append(result)

    return summary
