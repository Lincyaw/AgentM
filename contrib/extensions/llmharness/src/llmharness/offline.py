"""Offline audit primitive for recorded AgentM trajectories.

This module is deliberately a primitive, not an experiment runner. It audits a
visible message prefix with llmharness' auditor agent and returns surfaced
reminders. Branching, scoring, and case-study export live in eval packages such
as ``agentm-rescue-window``.
"""

from __future__ import annotations

import contextlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from agentm.core.abi import (
    AgentMessage,
    AgentSessionConfig,
    AssistantMessage,
    ToolCallBlock,
)
from agentm.core.runtime import AgentSession
from loguru import logger

from llmharness.agents.auditor.tools import SUBMIT_VERDICT_TOOL_NAME
from llmharness.schema import Reminder, Verdict
from llmharness.state import CumulativeAuditState

PhaseStatus = Literal["ok", "no_call", "spawn_error", "prompt_error"]


@dataclass(slots=True)
class PhaseResult:
    """Outcome of one standalone auditor phase invocation."""

    output: dict[str, Any] | None
    status: PhaseStatus
    error: str | None
    latency_ms: int
    messages: list[AgentMessage]


@dataclass(frozen=True, slots=True)
class AuditorSettings:
    """Minimal config needed for one offline auditor firing."""

    base_prompt: str | None = None
    tools: tuple[str, ...] | None = None

    @classmethod
    def default(cls) -> AuditorSettings:
        from llmharness.agents.auditor.context import load_auditor_prompt

        return cls(base_prompt=load_auditor_prompt("minimal_index"))


@dataclass(frozen=True, slots=True)
class SurfaceFiring:
    """One auditor firing that surfaced a reminder during offline audit."""

    turn_index: int
    reminder_text: str
    cumulative_snapshot: CumulativeAuditState


@dataclass(slots=True)
class OfflineRunResult:
    """Outcome of one offline audit invocation."""

    reminder: Reminder | None
    state: CumulativeAuditState
    all_step_results: list[dict[str, Any]] = field(default_factory=list)
    surfaces: list[SurfaceFiring] = field(default_factory=list)


class StandaloneChildRunner:
    """Spawns top-level auditor sessions for offline audit."""

    def __init__(
        self,
        cwd: str,
        *,
        parent_session_id: str | None = None,
        trace_id: str | None = None,
    ) -> None:
        self._cwd = cwd
        self._parent_session_id = parent_session_id
        self._trace_id = trace_id

    async def run_auditor(
        self,
        *,
        prompt_text: str,
        tools_config: dict[str, Any],
        provider: tuple[str, dict[str, Any]] | None,
        context_index: dict[str, Any] | None = None,
        recent_verdicts: list[dict[str, Any]] | None = None,
        continuation_notes_from_prior_firing: list[str] | None = None,
        trajectory: list[dict[str, Any]] | None = None,
        symbols: list[dict[str, Any]] | None = None,
        references: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        _auditor_tools = "llmharness.agents.auditor.tools"
        _index_tools = "llmharness.agents.auditor.index_tools"
        _observability = "agentm.extensions.builtin.observability"
        _operations = "agentm.extensions.builtin.operations"
        _system_prompt = "agentm.extensions.builtin.system_prompt"
        extensions: list[tuple[str, dict[str, Any]]] = [
            (_observability, {}),
            (_operations, {}),
            (_auditor_tools, dict(tools_config)),
            (
                _index_tools,
                {
                    "trajectory": trajectory or [],
                    "symbols": symbols or [],
                    "references": references or [],
                    "context_index": context_index or {},
                },
            ),
            (_system_prompt, {"prompt": prompt_text}),
        ]
        payload: dict[str, Any] = {
            "context_index": context_index,
            "recent_verdicts": list(recent_verdicts or []),
            "continuation_notes_from_prior_firing": list(
                continuation_notes_from_prior_firing or []
            ),
        }
        result = await run_phase_standalone(
            cwd=self._cwd,
            extensions=extensions,
            provider=provider,
            payload=payload,
            terminal_tool=SUBMIT_VERDICT_TOOL_NAME,
            purpose="cognitive_audit_auditor_offline",
            parent_session_id=self._parent_session_id,
            trace_id=self._trace_id,
        )
        raw_blocks = _flatten_assistant_blocks(result.messages)
        if result.status != "ok" or result.output is None:
            return {
                "verdict": None,
                "raw_blocks": raw_blocks,
                "error": result.error,
                "latency_ms": result.latency_ms,
            }
        try:
            verdict_raw = result.output.get("verdict") or result.output
            if isinstance(verdict_raw, dict):
                verdict = Verdict.from_dict(verdict_raw)
                return {
                    "verdict": verdict,
                    "raw_blocks": raw_blocks,
                    "error": None,
                    "latency_ms": result.latency_ms,
                }
        except (KeyError, TypeError, ValueError) as exc:
            return {
                "verdict": None,
                "raw_blocks": raw_blocks,
                "error": f"malformed: {exc}",
                "latency_ms": result.latency_ms,
            }
        return {
            "verdict": None,
            "raw_blocks": raw_blocks,
            "error": "no verdict in output",
            "latency_ms": result.latency_ms,
        }


def terminal_tool_arguments(
    messages: list[AgentMessage],
    tool_name: str,
) -> dict[str, Any] | None:
    """Extract the arguments of the last call to ``tool_name``."""

    for msg in reversed(messages):
        if not isinstance(msg, AssistantMessage):
            continue
        for block in reversed(msg.content):
            if isinstance(block, ToolCallBlock) and block.name == tool_name:
                return dict(block.arguments)
    return None


async def run_phase_standalone(
    *,
    cwd: str,
    extensions: list[tuple[str, dict[str, Any]]],
    provider: tuple[str, dict[str, Any]] | None,
    payload: dict[str, Any] | str,
    terminal_tool: str,
    purpose: str = "cognitive_audit_offline",
    parent_session_id: str | None = None,
    trace_id: str | None = None,
) -> PhaseResult:
    """Spawn a top-level auditor session and return terminal-tool arguments."""

    config = AgentSessionConfig(
        cwd=cwd,
        provider=provider,
        extensions=extensions,
        purpose=purpose,
        parent_session_id=parent_session_id,
        root_session_id=trace_id,
    )
    t0 = time.monotonic()
    try:
        session = await AgentSession.create(config)
    except Exception as exc:
        logger.debug("offline: caught exception: {}", exc)
        return PhaseResult(
            output=None,
            status="spawn_error",
            error=str(exc),
            latency_ms=int((time.monotonic() - t0) * 1000),
            messages=[],
        )

    try:
        if isinstance(payload, str):
            user_message = payload
        else:
            user_message = json.dumps(payload, ensure_ascii=False, default=str)
        messages = await session.prompt(user_message)
    except Exception as exc:
        logger.debug("offline: caught exception: {}", exc)
        with contextlib.suppress(Exception):
            await session.shutdown()
        return PhaseResult(
            output=None,
            status="prompt_error",
            error=str(exc),
            latency_ms=int((time.monotonic() - t0) * 1000),
            messages=[],
        )

    with contextlib.suppress(Exception):
        await session.shutdown()
    latency_ms = int((time.monotonic() - t0) * 1000)
    args = terminal_tool_arguments(messages, terminal_tool)
    if args is None:
        return PhaseResult(
            output=None,
            status="no_call",
            error=f"terminal tool {terminal_tool!r} was not called",
            latency_ms=latency_ms,
            messages=messages,
        )
    return PhaseResult(
        output=args,
        status="ok",
        error=None,
        latency_ms=latency_ms,
        messages=messages,
    )


async def audit_pipeline_over_trajectory(
    *,
    messages: list[AgentMessage],
    cwd: str,
    session_id: str,
    provider: tuple[str, dict[str, Any]] | None,
    auditor_settings: AuditorSettings,
    audit_interval: int = 5,
    enable_auditor: bool = True,
    stop_on_first_surface: bool = True,
    child: StandaloneChildRunner | None = None,
    seed_cumulative: CumulativeAuditState | None = None,
    start_turn: int = 1,
    symbols: list[dict[str, Any]] | None = None,
    references: list[dict[str, Any]] | None = None,
    trace_id: str | None = None,
) -> OfflineRunResult:
    """Run the cognitive-audit auditor over a captured trajectory."""

    resolved_trace_id = trace_id if trace_id is not None else session_id
    cumulative = seed_cumulative if seed_cumulative is not None else CumulativeAuditState.fresh()
    child_used = child if child is not None else StandaloneChildRunner(
        cwd,
        parent_session_id=session_id,
        trace_id=resolved_trace_id,
    )
    all_steps: list[dict[str, Any]] = []
    surfaces: list[SurfaceFiring] = []
    reminder: Reminder | None = None
    resolved_symbols = symbols or []
    resolved_references = references or []

    from llmharness.agents.auditor.context import build_auditor_system_prompt
    from llmharness.atom import _extract_loaded_skills, _serialize_trajectory
    from llmharness.context_index import build_context_index

    for turn_number, prefix_len in enumerate(_turn_end_prefix_lengths(messages), start=1):
        if prefix_len < start_turn:
            continue
        prefix = messages[:prefix_len]
        step_result: dict[str, Any] = {
            "turn_count": turn_number,
            "prefix_len": prefix_len,
            "surfaced_reminder": None,
            "auditor_record": None,
        }
        auditor_due = enable_auditor and (turn_number % audit_interval) == 0
        if auditor_due:
            trajectory = _serialize_trajectory(prefix)
            context_index = build_context_index(
                trajectory=trajectory,
                symbols=resolved_symbols,
                references=resolved_references,
            ).to_dict()
            loaded_skills = _extract_loaded_skills(prefix)
            prompt_text = (
                auditor_settings.base_prompt
                or "You are the cognitive-audit auditor."
            )
            aud_prompt = build_auditor_system_prompt(
                check_errors={},
                continuation_notes=list(cumulative.last_continuation_notes),
                base_prompt=prompt_text,
                methodology=loaded_skills,
                context_index=context_index,
            )
            aud_result = await child_used.run_auditor(
                prompt_text=aud_prompt,
                tools_config={
                    "tools": list(
                        auditor_settings.tools or (SUBMIT_VERDICT_TOOL_NAME,)
                    )
                },
                provider=provider,
                context_index=context_index,
                recent_verdicts=list(cumulative.recent_verdicts),
                continuation_notes_from_prior_firing=list(
                    cumulative.last_continuation_notes
                ),
                trajectory=trajectory,
                symbols=resolved_symbols,
                references=resolved_references,
            )
            verdict = aud_result.get("verdict")
            if verdict is not None:
                cumulative.absorb_auditor_verdict(verdict.to_dict())
                if verdict.surface_reminder and verdict.reminder_text:
                    surface = SurfaceFiring(
                        turn_index=prefix_len - 1,
                        reminder_text=verdict.reminder_text,
                        cumulative_snapshot=cumulative.snapshot(),
                    )
                    surfaces.append(surface)
                    if stop_on_first_surface:
                        reminder = Reminder(text=verdict.reminder_text)
                        step_result["surfaced_reminder"] = reminder
                        all_steps.append(step_result)
                        break
        all_steps.append(step_result)

    return OfflineRunResult(
        reminder=reminder,
        state=cumulative,
        all_step_results=all_steps,
        surfaces=surfaces,
    )


async def offline_audit(
    messages: list[Any],
    *,
    cwd: str,
    provider: tuple[str, dict[str, Any]],
    audit_interval: int = 5,
    auditor_prompt: str = "minimal_index",
    stop_on_first_surface: bool = False,
    min_surface_turn_index: int = 0,
    **kwargs: Any,
) -> OfflineRunResult:
    """Audit a recorded trajectory and return surfaced reminder candidates."""

    del kwargs
    from llmharness.agents.auditor.context import load_auditor_prompt

    typed_messages: list[AgentMessage] = list(messages)
    if not typed_messages:
        raise ValueError("offline_audit requires at least one message")

    base_prompt = load_auditor_prompt(auditor_prompt)
    auditor_settings = AuditorSettings(base_prompt=base_prompt)
    session_id = f"offline-audit-{id(messages)}"
    child = StandaloneChildRunner(cwd)
    logger.info(
        f"offline_audit: {len(typed_messages)} messages, "
        f"interval={audit_interval}, prompt={auditor_prompt}"
    )
    start_turn = 1 if min_surface_turn_index <= 0 else min_surface_turn_index + 2
    result = await audit_pipeline_over_trajectory(
        messages=typed_messages,
        cwd=cwd,
        session_id=session_id,
        provider=provider,
        auditor_settings=auditor_settings,
        audit_interval=audit_interval,
        enable_auditor=True,
        stop_on_first_surface=stop_on_first_surface,
        child=child,
        start_turn=start_turn,
    )
    if min_surface_turn_index > 0:
        result.surfaces = [
            surface
            for surface in result.surfaces
            if surface.turn_index > min_surface_turn_index
        ]
    return result


def _turn_end_prefix_lengths(messages: list[AgentMessage]) -> list[int]:
    assistant_idxs = [
        index for index, message in enumerate(messages) if isinstance(message, AssistantMessage)
    ]
    if not assistant_idxs:
        return [len(messages)] if messages else []
    bounds: list[int] = []
    for index, _ in enumerate(assistant_idxs):
        next_assistant = (
            assistant_idxs[index + 1]
            if index + 1 < len(assistant_idxs)
            else len(messages)
        )
        bounds.append(next_assistant)
    return bounds


def _flatten_assistant_blocks(messages: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, AssistantMessage):
            continue
        for block in msg.content:
            if isinstance(block, ToolCallBlock):
                out.append(
                    {
                        "type": "tool_call",
                        "name": block.name,
                        "arguments": dict(block.arguments),
                    }
                )
            elif hasattr(block, "text"):
                btype = getattr(block, "type", "text")
                out.append({"type": btype, "text": block.text})
    return out


__all__ = [
    "AuditorSettings",
    "OfflineRunResult",
    "PhaseResult",
    "StandaloneChildRunner",
    "SurfaceFiring",
    "audit_pipeline_over_trajectory",
    "offline_audit",
    "run_phase_standalone",
    "terminal_tool_arguments",
]
