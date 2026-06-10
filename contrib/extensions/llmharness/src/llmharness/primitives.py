"""Shared building blocks for the cognitive-audit pipeline.

* :class:`CumulativeAuditState` — event-sourced graph state across firings
* :func:`build_auditor_input` — cumulative state → prompt + tools config
* :func:`process_auditor_output` — raw terminal-tool args → Verdict
* Message serialization helpers used by atom.py for data preparation
"""

from __future__ import annotations

import collections
import contextlib
import copy
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from agentm.core.abi.messages import AgentMessage, AssistantMessage
from agentm.core.abi.session import SessionEntry

from . import schema as _et
from .agents.auditor.output import AuditorOutputError, RawVerdictOutput
from .agents.auditor.profiles import (
    TOOL_GET_EVENT_DETAIL,
    TOOL_GET_TURN,
    TOOL_SUBMIT_VERDICT,
)
from .agents.auditor.prompt import (
    build_auditor_system_prompt,
    build_auditor_trajectory_prompt,
    load_auditor_prompt,
)
from .graph.ops import GraphOp, parse_op
from .schema import Edge, Event, Finding, Phase, Verdict

_DEFAULT_RECENT_VERDICTS: Final[int] = _et.RECENT_VERDICTS_FOR_AUDITOR


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AuditorInput:
    """Everything needed to fire one auditor agent invocation."""

    prompt_text: str
    tools_config: dict[str, Any]


@dataclass(frozen=True)
class AuditorOutput:
    """Result of processing one auditor firing."""

    verdict: Verdict | None
    error: str | None = None


@dataclass(frozen=True)
class AuditorSettings:
    """Knobs for assembling an auditor firing."""

    base_prompt: str
    summary_threshold: int = 30
    tools: tuple[str, ...] = (TOOL_SUBMIT_VERDICT,)

    @classmethod
    def default(cls) -> AuditorSettings:
        return cls(base_prompt=load_auditor_prompt("minimal"))


# ---------------------------------------------------------------------------
# CumulativeAuditState
# ---------------------------------------------------------------------------


def _bool_safe_int(raw: Any) -> int | None:
    if isinstance(raw, int) and not isinstance(raw, bool):
        return raw
    return None


@dataclass
class CumulativeAuditState:
    """Event-sourced graph state + auditor side-channel state."""

    ops: list[GraphOp] = field(default_factory=list)
    cursor_last_turn_index: int = -1
    recent_verdicts: collections.deque[dict[str, Any]] = field(
        default_factory=lambda: collections.deque(maxlen=_DEFAULT_RECENT_VERDICTS)
    )
    last_continuation_notes: list[str] = field(default_factory=list)
    firing_id_counter: int = 0
    _cached_len: int = -1
    _cached_view: tuple[tuple[Event, ...], tuple[Edge, ...], tuple[Phase, ...]] | None = None
    _phases: list[Phase] = field(default_factory=list)

    def graph_view(self) -> tuple[tuple[Event, ...], tuple[Edge, ...], tuple[Phase, ...]]:
        if self._cached_view is not None and self._cached_len == len(self.ops):
            return self._cached_view
        from .graph.fold import fold_graph

        folded = fold_graph(self.ops)
        events = tuple(folded.nodes_list())
        edges = tuple(folded.edges_list())
        phases = tuple(self._phases)
        self._cached_view = (events, edges, phases)
        self._cached_len = len(self.ops)
        return self._cached_view

    def next_event_id(self) -> int:
        events, _edges, _phases = self.graph_view()
        return max((e.id for e in events), default=0) + 1

    def _invalidate_cache(self) -> None:
        self._cached_view = None
        self._cached_len = -1

    def absorb_extractor_firing(
        self,
        *,
        firing_ops: Sequence[GraphOp],
        firing_cursor: int,
        firing_id: int,
        firing_phases: Sequence[Phase] = (),
    ) -> None:
        self.ops.extend(firing_ops)
        self.cursor_last_turn_index = firing_cursor
        self._phases.extend(firing_phases)
        if firing_id >= self.firing_id_counter:
            self.firing_id_counter = firing_id + 1
        self._invalidate_cache()

    def absorb_auditor_verdict(self, verdict: dict[str, Any], *, is_silent: bool) -> None:
        del is_silent
        self.recent_verdicts.append(verdict)
        raw_notes = verdict.get("continuation_notes")
        if isinstance(raw_notes, list):
            self.last_continuation_notes = [n for n in raw_notes if isinstance(n, str)]

    @classmethod
    def fresh(cls) -> CumulativeAuditState:
        return cls()

    def snapshot(self) -> CumulativeAuditState:
        return copy.deepcopy(self)

    @classmethod
    def hydrate_from_session_log(cls, branch: list[SessionEntry]) -> CumulativeAuditState:
        ops: list[GraphOp] = []
        verdicts_all: list[dict[str, Any]] = []
        cursor_last_turn_index = -1

        for entry in branch:
            payload = entry.payload
            if not isinstance(payload, dict):
                continue
            if entry.type == _et.AUDIT_GRAPH_OP:
                try:
                    ops.append(parse_op(payload))
                except (KeyError, ValueError, TypeError):
                    continue
            elif entry.type == _et.VERDICT:
                verdicts_all.append(payload)
            elif entry.type == _et.EXTRACTOR_CURSOR:
                raw = _bool_safe_int(payload.get("last_turn_index"))
                if raw is not None:
                    cursor_last_turn_index = raw

        last_notes: list[str] = []
        if verdicts_all:
            raw_notes = verdicts_all[-1].get("continuation_notes")
            if isinstance(raw_notes, list):
                last_notes = [n for n in raw_notes if isinstance(n, str)]

        recent: collections.deque[dict[str, Any]] = collections.deque(
            maxlen=_DEFAULT_RECENT_VERDICTS
        )
        for v in verdicts_all[-_DEFAULT_RECENT_VERDICTS:]:
            recent.append(v)

        return cls(
            ops=ops,
            cursor_last_turn_index=cursor_last_turn_index,
            recent_verdicts=recent,
            last_continuation_notes=last_notes,
            firing_id_counter=0,
        )


# ---------------------------------------------------------------------------
# Message serialization helpers
# ---------------------------------------------------------------------------


def serialize_block(block: Any) -> dict[str, Any] | None:
    """Serialize one content block into a dict."""
    text = getattr(block, "text", None)
    if isinstance(text, str) and text:
        block_type = getattr(block, "type", None)
        return {
            "type": block_type if isinstance(block_type, str) and block_type else "text",
            "text": text,
        }

    name = getattr(block, "name", None)
    arguments = getattr(block, "arguments", None)
    if isinstance(name, str) and isinstance(arguments, dict):
        return {
            "type": "tool_call",
            "id": getattr(block, "id", None),
            "name": name,
            "arguments": dict(arguments),
        }

    tool_call_id = getattr(block, "tool_call_id", None)
    inner_content = getattr(block, "content", None)
    if isinstance(tool_call_id, str) and isinstance(inner_content, list):
        inner_blocks: list[dict[str, Any]] = []
        for inner in inner_content:
            inner_text = getattr(inner, "text", None)
            if isinstance(inner_text, str):
                inner_blocks.append({"type": "text", "text": inner_text})
            else:
                inner_blocks.append(
                    {
                        "type": getattr(inner, "type", inner.__class__.__name__),
                        "repr": repr(inner),
                    }
                )
        return {
            "type": "tool_result",
            "tool_call_id": tool_call_id,
            "content": inner_blocks,
            "is_error": bool(getattr(block, "is_error", False)),
        }

    return {"type": getattr(block, "type", block.__class__.__name__), "repr": repr(block)}


def flatten_assistant_blocks(messages: list[AgentMessage]) -> list[dict[str, Any]]:
    """Flatten every assistant message's content blocks into dicts."""
    blocks: list[dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, AssistantMessage):
            continue
        content = getattr(msg, "content", None)
        if not isinstance(content, list):
            continue
        for blk in content:
            serialized = serialize_block(blk)
            if serialized is not None:
                blocks.append(serialized)
    return blocks


def _render_message_text(msg: AgentMessage) -> str:
    parts: list[str] = []
    content = getattr(msg, "content", None)
    if isinstance(content, list):
        for block in content:
            text = getattr(block, "text", None)
            if isinstance(text, str) and text:
                parts.append(text)
                continue
            inner = getattr(block, "content", None)
            if isinstance(inner, list):
                for sub in inner:
                    sub_text = getattr(sub, "text", None)
                    if isinstance(sub_text, str) and sub_text:
                        parts.append(sub_text)
            args = getattr(block, "arguments", None)
            if isinstance(args, dict):
                with contextlib.suppress(TypeError, ValueError):
                    parts.append(json.dumps(args, ensure_ascii=False, default=str))
    return " ".join(parts)


def _serialize_message(msg: AgentMessage, *, index: int) -> dict[str, Any] | None:
    content = getattr(msg, "content", None)
    if not isinstance(content, list):
        return None
    blocks = [b for b in (serialize_block(blk) for blk in content) if b is not None]
    if not blocks:
        return None
    return {"index": index, "role": getattr(msg, "role", "unknown"), "content": blocks}


def serialize_full_trajectory(messages: list[AgentMessage]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, msg in enumerate(messages):
        serialized = _serialize_message(msg, index=i)
        if serialized is not None:
            out.append(serialized)
    return out


# ---------------------------------------------------------------------------
# Auditor primitives
# ---------------------------------------------------------------------------


def build_auditor_input(
    cumulative: CumulativeAuditState,
    settings: AuditorSettings,
    *,
    trajectory_snapshot: list[dict[str, Any]] | None = None,
    findings: list[Finding] | None = None,
    check_errors: dict[str, str] | None = None,
    skip_extractor: bool = False,
) -> AuditorInput:
    """Build prompt_text + tools_config for the auditor agent."""
    continuation_notes = list(cumulative.last_continuation_notes)

    if skip_extractor:
        framing = settings.base_prompt or load_auditor_prompt("trajectory")
        prompt_text = build_auditor_trajectory_prompt(
            trajectory=trajectory_snapshot or [],
            continuation_notes=continuation_notes,
            base_prompt=framing,
        )
        tools_config: dict[str, Any] = {"tools": [TOOL_SUBMIT_VERDICT]}
        return AuditorInput(prompt_text=prompt_text, tools_config=tools_config)

    events, edges, phases = cumulative.graph_view()

    prompt_text = build_auditor_system_prompt(
        events=events,
        edges=edges,
        phases=phases,
        findings=list(findings or []),
        check_errors=dict(check_errors or {}),
        continuation_notes=continuation_notes,
        summary_threshold=settings.summary_threshold,
        base_prompt=settings.base_prompt,
    )

    selected: list[str] = []
    if TOOL_SUBMIT_VERDICT in settings.tools:
        selected.append(TOOL_SUBMIT_VERDICT)
    if TOOL_GET_TURN in settings.tools and trajectory_snapshot is not None:
        selected.append(TOOL_GET_TURN)
    if TOOL_GET_EVENT_DETAIL in settings.tools:
        selected.append(TOOL_GET_EVENT_DETAIL)

    tools_config = {"tools": selected}
    if TOOL_GET_TURN in selected:
        tools_config["trajectory_snapshot"] = trajectory_snapshot
    if TOOL_GET_EVENT_DETAIL in selected:
        tools_config["events"] = list(events)
        tools_config["edges"] = list(edges)

    return AuditorInput(prompt_text=prompt_text, tools_config=tools_config)


def process_auditor_output(
    terminal_args: dict[str, Any] | None,
    cumulative: CumulativeAuditState | None = None,
) -> AuditorOutput:
    """Parse auditor terminal-tool output into a Verdict."""
    if terminal_args is None:
        return AuditorOutput(verdict=None, error="auditor did not call submit_verdict")

    try:
        raw = RawVerdictOutput.from_dict(terminal_args)
    except AuditorOutputError as exc:
        return AuditorOutput(verdict=None, error=f"malformed: {exc}")

    try:
        verdict = raw.to_verdict()
    except AuditorOutputError as exc:
        return AuditorOutput(verdict=None, error=f"malformed: {exc}")

    if cumulative is not None:
        cumulative.absorb_auditor_verdict(
            verdict.to_dict(), is_silent=not verdict.surface_reminder,
        )

    return AuditorOutput(verdict=verdict)


__all__ = [
    "AuditorInput",
    "AuditorOutput",
    "AuditorSettings",
    "CumulativeAuditState",
    "build_auditor_input",
    "flatten_assistant_blocks",
    "process_auditor_output",
    "serialize_block",
    "serialize_full_trajectory",
]
