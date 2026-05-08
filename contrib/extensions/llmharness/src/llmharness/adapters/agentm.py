"""Adapter: AgentM bus -> two-phase cognitive audit.

See `.claude/designs/llmharness-cognitive-audit.md` for the full design.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from agentm.core.abi import TurnEndEvent
from agentm.core.abi.messages import (
    AgentMessage,
    AssistantMessage,
    ToolCallBlock,
    ToolResultMessage,
)
from agentm.core.abi.session import SessionEntry
from agentm.extensions import ExtensionManifest
from agentm.harness.events import BeforeAgentStartEvent
from agentm.harness.extension import ExtensionAPI
from agentm.harness.session_config import AgentSessionConfig

from ..audit.auditor import (
    SUBMIT_VERDICT_TOOL_NAME,
    AuditorOutputError,
    RawVerdictOutput,
    compose_auditor_extensions,
)
from ..audit.extractor import (
    SUBMIT_EVENTS_TOOL_NAME,
    ExtractorOutputError,
    RawExtractorOutput,
    compose_extractor_extensions,
)
from ..schema import Event, Reminder, Verdict

MANIFEST = ExtensionManifest(
    name="agentm",
    description=(
        "Two-phase cognitive-audit adapter: per-turn extractor (Phase 1) and "
        "every-k-turns graph auditor (Phase 2); injects pending reminders on "
        "BeforeAgentStartEvent."
    ),
    registers=(
        "event:turn_end",
        "event:before_agent_start",
    ),
    config_schema={
        "type": "object",
        "properties": {
            "audit_interval_turns": {"type": "integer", "minimum": 1},
            "prompt_override_extractor": {"type": "string"},
            "prompt_override_auditor": {"type": "string"},
            "cards_tools_config": {"type": ["object", "null"]},
            "observability_config": {"type": ["object", "null"]},
        },
        "additionalProperties": False,
    },
    affects=(
        "event:turn_end",
        "event:before_agent_start",
    ),
    api_version=1,
    tier=1,
)


_DEFAULT_AUDIT_INTERVAL_TURNS = 3
_DEFAULT_RECENT_VERDICTS = 5
_RECENT_GRAPH_SLICE_FOR_EXTRACTOR = 20
_REMINDER_PREFIX = "\n\n[harness] "

_AUDIT_EVENT_ENTRY_TYPE = "llmharness.audit_event"
_VERDICT_ENTRY_TYPE = "llmharness.verdict"
_EXTRACTOR_CURSOR_ENTRY_TYPE = "llmharness.extractor_cursor"

_EXTRACTOR_NO_CALL_ENTRY = "llmharness.extractor_no_call"
_EXTRACTOR_ERROR_ENTRY = "llmharness.extractor_error"
_EXTRACTOR_EMPTY_ENTRY = "llmharness.extractor_empty"
_AUDIT_NO_CALL_ENTRY = "llmharness.audit_no_call"
_AUDIT_ERROR_ENTRY = "llmharness.audit_error"


# --- branch state -----------------------------------------------------------


@dataclass(frozen=True)
class _BranchState:
    """Snapshot of audit-relevant entries pulled from a single branch walk."""

    cursor_last_turn_index: int
    graph: list[Event]
    recent_verdicts: list[dict[str, Any]]


def _scan_branch(branch: list[SessionEntry], *, recent_verdicts_n: int) -> _BranchState:
    """Single-pass extraction of cursor + graph + recent verdicts."""
    cursor_last_turn_index = -1
    graph: list[Event] = []
    verdicts: list[dict[str, Any]] = []

    for entry in branch:
        payload = entry.payload
        if not isinstance(payload, dict):
            continue
        if entry.type == _AUDIT_EVENT_ENTRY_TYPE:
            try:
                graph.append(Event.from_dict(payload))
            except (KeyError, ValueError, TypeError):
                continue
        elif entry.type == _VERDICT_ENTRY_TYPE:
            verdicts.append(payload)
        elif entry.type == _EXTRACTOR_CURSOR_ENTRY_TYPE:
            raw = payload.get("last_turn_index")
            if isinstance(raw, int) and not isinstance(raw, bool):
                cursor_last_turn_index = raw

    return _BranchState(
        cursor_last_turn_index=cursor_last_turn_index,
        graph=graph,
        recent_verdicts=verdicts[-recent_verdicts_n:] if recent_verdicts_n > 0 else [],
    )


# --- failure recording ------------------------------------------------------


def _record_failure(
    api: ExtensionAPI, entry_type: str, payload: dict[str, Any]
) -> None:
    """Single chokepoint for typed failure entries. Append-only; never raises."""
    api.session.append_entry(entry_type, payload)


def _window_is_non_trivial(messages_slice: list[AgentMessage]) -> bool:
    """True iff the slice contains any AssistantMessage or ToolResultMessage.

    A pure user-only window does NOT count as non-trivial — extractor is not
    expected to produce events from user messages alone, so an empty events
    array on such a window is normal, not a failure (design §4).
    """
    return any(
        isinstance(msg, (AssistantMessage, ToolResultMessage)) for msg in messages_slice
    )


# --- install ----------------------------------------------------------------


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    k = int(config.get("audit_interval_turns", _DEFAULT_AUDIT_INTERVAL_TURNS))
    if k < 1:
        k = _DEFAULT_AUDIT_INTERVAL_TURNS

    cards_cfg_raw = config.get("cards_tools_config", {})
    obs_cfg_raw = config.get("observability_config", {})
    cards_cfg = cards_cfg_raw if isinstance(cards_cfg_raw, dict) else None
    obs_cfg = obs_cfg_raw if isinstance(obs_cfg_raw, dict) else None

    prompt_extractor_raw = config.get("prompt_override_extractor")
    prompt_auditor_raw = config.get("prompt_override_auditor")
    prompt_extractor = (
        prompt_extractor_raw if isinstance(prompt_extractor_raw, str) else None
    )
    prompt_auditor = (
        prompt_auditor_raw if isinstance(prompt_auditor_raw, str) else None
    )

    extractor_extensions = compose_extractor_extensions(
        prompt_override=prompt_extractor,
        cards_tools_config=cards_cfg,
        observability_config=obs_cfg,
    )
    auditor_extensions = compose_auditor_extensions(
        prompt_override=prompt_auditor,
        cards_tools_config=cards_cfg,
        observability_config=obs_cfg,
    )

    turn_count = 0
    pending: Reminder | None = None

    async def _on_turn_end(event: TurnEndEvent) -> None:
        nonlocal turn_count, pending

        turn_count += 1
        branch = api.session.get_branch()
        # Use the event's live snapshot, not ``api.session.get_messages()``
        # — the kernel persists messages to the SessionManager only after
        # ``prompt()`` returns, so a mid-loop session-view read is stale
        # (returns just the initial user message). The event carries the
        # authoritative trajectory up through this turn's assistant_msg.
        messages = list(event.messages)
        state = _scan_branch(branch, recent_verdicts_n=_DEFAULT_RECENT_VERDICTS)

        # --- Phase 1: extractor (always) ---
        new_events = await _run_extractor(
            api=api,
            extractor_extensions=extractor_extensions,
            messages=messages,
            cursor_last_turn_index=state.cursor_last_turn_index,
            recent_graph_slice=state.graph[-_RECENT_GRAPH_SLICE_FOR_EXTRACTOR:],
            next_event_id=_next_event_id(state.graph),
        )
        if new_events is None:
            return  # typed failure already recorded; cursor stays put

        for ev in new_events:
            api.session.append_entry(_AUDIT_EVENT_ENTRY_TYPE, ev.to_dict())

        api.session.append_entry(
            _EXTRACTOR_CURSOR_ENTRY_TYPE,
            {
                "last_turn_index": len(messages) - 1,
                "extraction_run_id": uuid.uuid4().hex,
            },
        )

        # --- Phase 2: auditor (every k turns) ---
        if (turn_count % k) != 0:
            return

        verdict = await _run_auditor(
            api=api,
            auditor_extensions=auditor_extensions,
            graph_events=state.graph + new_events,
            recent_verdicts=state.recent_verdicts,
        )
        if verdict is None:
            return

        api.session.append_entry(_VERDICT_ENTRY_TYPE, verdict.to_dict())

        if verdict.drift and verdict.reminder and verdict.type is not None:
            pending = Reminder(
                type=verdict.type,
                text=verdict.reminder,
            )

    def _on_before_agent_start(event: BeforeAgentStartEvent) -> None:
        nonlocal pending
        if pending is None:
            return
        reminder, pending = pending, None
        existing = event.system or ""
        event.system = existing + _REMINDER_PREFIX + reminder.text

    api.on(TurnEndEvent.CHANNEL, _on_turn_end)
    api.on(BeforeAgentStartEvent.CHANNEL, _on_before_agent_start)


def _next_event_id(prior_events: list[Event]) -> int:
    return max((e.id for e in prior_events), default=-1) + 1


# --- trajectory slicing -----------------------------------------------------


def _slice_new_turns(
    messages: list[AgentMessage], cursor_last_turn_index: int
) -> list[dict[str, Any]]:
    """Slice ``messages[cursor + 1 :]`` into a payload the extractor can consume.

    Keeps thinking blocks; tool-result content stays structured (list of inner
    blocks rather than a flattened string) so the extractor can distinguish
    error vs success and inspect tool extras (design §7.4).
    """
    start = max(cursor_last_turn_index + 1, 0)
    out: list[dict[str, Any]] = []
    for i, msg in enumerate(messages[start:], start=start):
        serialized = _serialize_message_for_extractor(msg, index=i)
        if serialized is not None:
            out.append(serialized)
    return out


def _serialize_message_for_extractor(
    msg: AgentMessage, *, index: int
) -> dict[str, Any] | None:
    """Best-effort dict view of one message.

    Walks blocks via duck-typing on public attrs so the serializer survives
    SDK additions (new content-block kinds) without code changes here.
    Unknown blocks fall through to a generic ``{type, repr}`` form rather than
    being silently dropped — visibility over precision.
    """
    content = getattr(msg, "content", None)
    if not isinstance(content, list):
        return None

    blocks = [b for b in (_serialize_block(blk) for blk in content) if b is not None]
    if not blocks:
        return None
    return {
        "index": index,
        "role": getattr(msg, "role", "unknown"),
        "content": blocks,
    }


def _serialize_block(block: Any) -> dict[str, Any] | None:
    """Serialize one content block; preserves thinking + tool-result structure."""
    text = getattr(block, "text", None)
    if isinstance(text, str) and text:
        block_type = getattr(block, "type", None)
        return {"type": block_type if isinstance(block_type, str) and block_type else "text", "text": text}

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


# --- phase runner -----------------------------------------------------------


_T = TypeVar("_T")


async def _run_phase(
    *,
    api: ExtensionAPI,
    extensions: list[tuple[str, dict[str, Any]]],
    purpose: str,
    payload: dict[str, Any],
    terminal_tool: str,
    coerce: Callable[[dict[str, Any]], _T],
    coerce_error: type[Exception],
    on_spawn_or_prompt_error: Callable[[str], None],
    on_no_call: Callable[[], None],
    on_malformed: Callable[[str], None],
) -> _T | None:
    """Spawn child, drive to terminal_tool, coerce, return result.

    Failure paths invoke the supplied callbacks (which write typed failure
    entries) and return None. The driver in ``_on_turn_end`` short-circuits
    on None without poisoning cursor / graph / pending state.
    """
    child_config = AgentSessionConfig(
        cwd=api.cwd,
        provider=None,
        extensions=extensions,
        purpose=purpose,
    )
    try:
        child = await api.spawn_child_session(child_config)
    except Exception as exc:
        on_spawn_or_prompt_error(str(exc))
        return None

    try:
        messages = await child.prompt(json.dumps(payload, ensure_ascii=False, default=str))
    except Exception as exc:
        on_spawn_or_prompt_error(str(exc))
        await _safe_shutdown(child)
        return None

    await _safe_shutdown(child)

    arguments = _find_terminal_tool_arguments(messages, terminal_tool)
    if arguments is None:
        on_no_call()
        return None

    try:
        return coerce(arguments)
    except coerce_error as exc:
        on_malformed(str(exc))
        return None


async def _run_extractor(
    *,
    api: ExtensionAPI,
    extractor_extensions: list[tuple[str, dict[str, Any]]],
    messages: list[AgentMessage],
    cursor_last_turn_index: int,
    recent_graph_slice: list[Event],
    next_event_id: int,
) -> list[Event] | None:
    new_turn_window = _slice_new_turns(messages, cursor_last_turn_index)
    if not new_turn_window:
        return []  # nothing to extract; clean no-op

    window_lo = max(cursor_last_turn_index + 1, 0)
    window_hi = len(messages) - 1
    window = [window_lo, window_hi]
    raw_window_slice = messages[window_lo:]

    payload = {
        "new_turns": new_turn_window,
        "recent_graph": [e.to_dict() for e in recent_graph_slice],
    }

    parsed = await _run_phase(
        api=api,
        extensions=extractor_extensions,
        purpose="cognitive_audit_extractor",
        payload=payload,
        terminal_tool=SUBMIT_EVENTS_TOOL_NAME,
        coerce=RawExtractorOutput.from_dict,
        coerce_error=ExtractorOutputError,
        on_spawn_or_prompt_error=lambda reason: _record_failure(
            api, _EXTRACTOR_ERROR_ENTRY, {"reason": reason, "turn_window": window}
        ),
        on_no_call=lambda: _record_failure(
            api,
            _EXTRACTOR_NO_CALL_ENTRY,
            {
                "reason": f"child returned without calling {SUBMIT_EVENTS_TOOL_NAME}",
                "turn_window": window,
            },
        ),
        on_malformed=lambda reason: _record_failure(
            api,
            _EXTRACTOR_ERROR_ENTRY,
            {"reason": f"malformed: {reason}", "turn_window": window},
        ),
    )
    if parsed is None:
        return None

    events = parsed.to_events(next_id=next_event_id)
    if not events and _window_is_non_trivial(raw_window_slice):
        _record_failure(api, _EXTRACTOR_EMPTY_ENTRY, {"turn_window": window})
        return None
    return events


async def _run_auditor(
    *,
    api: ExtensionAPI,
    auditor_extensions: list[tuple[str, dict[str, Any]]],
    graph_events: list[Event],
    recent_verdicts: list[dict[str, Any]],
) -> Verdict | None:
    payload = {
        "graph": [e.to_dict() for e in graph_events],
        "recent_verdicts": list(recent_verdicts),
    }
    raw = await _run_phase(
        api=api,
        extensions=auditor_extensions,
        purpose="cognitive_audit_auditor",
        payload=payload,
        terminal_tool=SUBMIT_VERDICT_TOOL_NAME,
        coerce=RawVerdictOutput.from_dict,
        coerce_error=AuditorOutputError,
        on_spawn_or_prompt_error=lambda reason: _record_failure(
            api, _AUDIT_ERROR_ENTRY, {"reason": reason}
        ),
        on_no_call=lambda: _record_failure(
            api,
            _AUDIT_NO_CALL_ENTRY,
            {"reason": f"child returned without calling {SUBMIT_VERDICT_TOOL_NAME}"},
        ),
        on_malformed=lambda reason: _record_failure(
            api, _AUDIT_ERROR_ENTRY, {"reason": f"malformed: {reason}"}
        ),
    )
    if raw is None:
        return None
    try:
        return raw.to_verdict()
    except AuditorOutputError as exc:
        _record_failure(api, _AUDIT_ERROR_ENTRY, {"reason": f"malformed: {exc}"})
        return None


def _find_terminal_tool_arguments(
    messages: list[AgentMessage], tool_name: str
) -> dict[str, Any] | None:
    for msg in reversed(messages):
        if not isinstance(msg, AssistantMessage):
            continue
        for block in reversed(msg.content):
            if isinstance(block, ToolCallBlock) and block.name == tool_name:
                return dict(block.arguments)
    return None


async def _safe_shutdown(child: Any) -> None:
    # Best-effort: audit pipeline must not crash the parent; child may already
    # be torn down by parent shutdown in some race conditions.
    try:
        await child.shutdown()
    except Exception:
        return


__all__ = [
    "MANIFEST",
    "install",
]
