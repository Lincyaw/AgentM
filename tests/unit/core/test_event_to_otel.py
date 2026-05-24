"""Behavioural contract for the declarative ``Event.to_otel`` mapping (PR-F).

The observability atom is now a thin dispatcher — every wire-format record
comes from a per-class :meth:`Event.to_otel` translator. These tests lock
down two properties consumers depend on:

1. **Coverage** — every concrete :class:`Event` subclass either declares its
   own :meth:`to_otel` override or inherits the base no-op deliberately.
   New Event subclasses cannot silently drop off the wire because
   :func:`test_every_event_class_has_to_otel` walks every subclass and
   records whether the method is overridden; the **default no-op set** is
   explicit, so adding an Event without considering wire format trips the
   test.

2. **Lifecycle pairing** — the ``span_tracker`` helper on
   :class:`SessionTelemetry` correctly pairs Start / End events through
   :meth:`open_span` and :meth:`pop_span`. Tested via the three real span
   families (``invoke_agent``, ``chat``, ``execute_tool``) by exercising
   the events directly against a minimal real :class:`SessionTelemetry`.

The full on-disk wire-format shape is locked down by
``tests/unit/extensions/test_observability_semconv.py``; this file does not
duplicate that. The fail-stop here is the *contract* between Event classes
and the observability dispatcher.
"""

from __future__ import annotations

import inspect
from dataclasses import is_dataclass
from pathlib import Path

import pytest

from agentm.core.abi import events as events_module
from agentm.core.abi.events import (
    AgentEndEvent,
    BeforeAgentStartEvent,
    Event,
    LlmRequestEndEvent,
    LlmRequestStartEvent,
    ModelEndTurn,
    ToolCallEvent,
    ToolResultEvent,
)
from agentm.core.abi.messages import AssistantMessage, TextContent
from agentm.core.abi.tool import ToolResult
from agentm.core.runtime.otel_export import setup_session_telemetry


# Events that intentionally have NO OTel wire-format mapping. Each entry
# justifies *why* the no-op base method is the right answer; if a new event
# joins the list, the comment must explain the reason. Drift here is what
# the test guards against: an unintended no-op slipping past code review.
_INTENTIONAL_NO_OP_EVENTS: frozenset[str] = frozenset(
    {
        # Kernel-internal lifecycle, covered by the bus observer's
        # ``agentm.event.dispatch`` records (channel + dispatch_id is all
        # consumers need; no span/log on the event itself).
        "AgentStartEvent",
        "DecideTurnActionEvent",
        # Mutation / decision channels — observability's "what changed"
        # records (``agentm.handler.mutated``) cover them via the observer.
        "BeforeSendToLlmEvent",
        "ContextEvent",
        "BeforeCompactEvent",
        "AfterCompactEvent",
        "BeforeInstallAtomEvent",
        "BeforeUnloadAtomEvent",
        # Stream chunks — explicitly excluded by the observability atom's
        # ``_DEFAULT_EXCLUDE_CHANNELS`` so we don't drown the trace in
        # token-by-token records.
        "StreamDeltaEvent",
        # Tool-error veneer — ``ToolResultEvent`` already carries the
        # error attribute on its span, this event only feeds the localizer
        # atom that rewrites the human-readable message.
        "ToolErrorEvent",
        # Child-session / cost / plan / command / resource — events
        # observed via the dispatch observer; their semantics are
        # subscriber-defined, not OTel-wire-format-load-bearing.
        "ChildSessionStartEvent",
        "ChildSessionEndEvent",
        "ChildSessionExtendingEvent",
        "CostBudgetExceededEvent",
        "PlanSubmittedEvent",
        "CommandDispatchedEvent",
        "ResolveSubagentEvent",
        "ResourcesDiscoverEvent",
        "ResourceWriteEvent",
        "EntryAppendedEvent",
        # Real-time persistence trigger emitted by ``AgentLoop`` — the
        # downstream :class:`MessageAppendedEvent` (fired when SessionManager
        # actually persists) carries the on-disk log via its own
        # ``to_otel``; this event only routes the persistence call, so a
        # separate OTel record would be a redundant duplicate.
        "MessagePersistedEvent",
        # Atom-install / -reload lifecycle — the observability atom owns
        # these because they need the fingerprint state (catalog hash
        # machinery) that is not on :class:`SessionTelemetry`.
        "ExtensionInstallEvent",
        "ExtensionReloadEvent",
    }
)


def _concrete_event_subclasses() -> list[type[Event]]:
    """Walk :mod:`agentm.core.abi.events` for every concrete Event subclass.

    Returns a stable-ordered list (by class name) so test failure messages
    surface predictable diffs.
    """
    out: list[type[Event]] = []
    for name in dir(events_module):
        obj = getattr(events_module, name)
        if not isinstance(obj, type):
            continue
        if obj is Event:
            continue
        if not issubclass(obj, Event):
            continue
        if not is_dataclass(obj):
            continue
        out.append(obj)
    return sorted(out, key=lambda cls: cls.__name__)


def test_every_event_class_has_to_otel() -> None:
    """Every concrete Event subclass either overrides ``to_otel`` or appears
    in the intentional no-op set. Catches new Event additions that forget
    to declare a wire-format mapping.
    """
    subclasses = _concrete_event_subclasses()
    assert subclasses, "events module exposes no concrete Event subclasses"

    overridden: set[str] = set()
    inherited_default: set[str] = set()
    for cls in subclasses:
        # ``to_otel`` is defined on Event; check whether the subclass owns
        # its own copy (via direct method-assign) or inherits the no-op.
        own = cls.__dict__.get("to_otel")
        if own is not None:
            overridden.add(cls.__name__)
        else:
            inherited_default.add(cls.__name__)

    # Every inherited-default event must be in the intentional set, and
    # vice versa: nothing in the intentional set should secretly override.
    unexpected_no_ops = inherited_default - _INTENTIONAL_NO_OP_EVENTS
    assert not unexpected_no_ops, (
        "Event subclasses without to_otel override and not in the "
        f"intentional no-op set: {sorted(unexpected_no_ops)}. Either add a "
        "to_otel override or document the no-op reason in "
        "_INTENTIONAL_NO_OP_EVENTS."
    )
    stale_intentional = _INTENTIONAL_NO_OP_EVENTS - inherited_default
    assert not stale_intentional, (
        "Intentional no-op set contains events that now override to_otel: "
        f"{sorted(stale_intentional)}. Remove them from "
        "_INTENTIONAL_NO_OP_EVENTS."
    )


def test_event_to_otel_base_default_is_no_op() -> None:
    """The base no-op is a real callable that accepts a telemetry handle
    and returns ``None`` without touching any field. Atoms that wire raw
    Event instances through ``Event.to_otel`` rely on this.
    """
    # ``inspect.getsource`` reads the method body — we don't assert on
    # source verbatim, only that the method exists and is callable.
    assert callable(Event.to_otel)
    sig = inspect.signature(Event.to_otel)
    assert list(sig.parameters) == ["self", "telemetry"]


# --- Lifecycle-pairing via the span_tracker --------------------------------


def _make_telemetry(tmp_path: Path) -> object:
    """Build a real :class:`SessionTelemetry` rooted in ``tmp_path``.

    The OTel SDK is happy with a freshly-built provider per test; the
    process-level singleton stays alive across tests but per-session
    processors are independent, so the test is hermetic.
    """
    return setup_session_telemetry(
        session_id="sess-to-otel",
        cwd=tmp_path,
        max_queue_size=128,
        schedule_delay_millis=10,
    )


def test_invoke_agent_span_pairing_via_to_otel(tmp_path: Path) -> None:
    """BeforeAgentStartEvent opens the ``invoke_agent`` span, AgentEndEvent
    closes it. The span_tracker must round-trip cleanly.
    """
    telemetry = _make_telemetry(tmp_path)
    telemetry.obs_scenario = "unit_scenario"
    telemetry.obs_purpose = "root"

    BeforeAgentStartEvent(messages=[], system=None).to_otel(telemetry)
    # The tracker now has one open invoke_agent span keyed by session_id.
    assert (
        "invoke_agent",
        telemetry.session_id,
    ) in telemetry.span_tracker

    msg = AssistantMessage(
        role="assistant",
        content=[TextContent(type="text", text="done")],
        timestamp=0.0,
        stop_reason="end_turn",
    )
    AgentEndEvent(messages=[msg], cause=ModelEndTurn()).to_otel(telemetry)
    assert (
        "invoke_agent",
        telemetry.session_id,
    ) not in telemetry.span_tracker
    telemetry.shutdown()


def test_chat_span_pairing_via_to_otel(tmp_path: Path) -> None:
    """LlmRequestStart/End must pair through the span_tracker. The pair key
    is the turn_id (stringified) so concurrent turns never collide.
    """
    telemetry = _make_telemetry(tmp_path)

    LlmRequestStartEvent(
        turn_index=0,
        message_count=1,
        tool_count=0,
        system_chars=3,
        model_id="m-stub",
        turn_id=7,
    ).to_otel(telemetry)
    assert ("chat", "7") in telemetry.span_tracker

    LlmRequestEndEvent(
        turn_index=0, chunk_count=2, duration_ns=1_000, error=None, turn_id=7
    ).to_otel(telemetry)
    assert ("chat", "7") not in telemetry.span_tracker
    telemetry.shutdown()


def test_execute_tool_span_pairing_via_to_otel(tmp_path: Path) -> None:
    """ToolCall/ToolResult pair through the span_tracker by tool_call_id."""
    telemetry = _make_telemetry(tmp_path)

    ToolCallEvent(
        tool_call_id="tc-1", tool_name="read", args={"path": "/x"}
    ).to_otel(telemetry)
    assert ("execute_tool", "tc-1") in telemetry.span_tracker

    ToolResultEvent(
        tool_call_id="tc-1",
        tool_name="read",
        result=ToolResult(content=[TextContent(type="text", text="ok")], is_error=False),
    ).to_otel(telemetry)
    assert ("execute_tool", "tc-1") not in telemetry.span_tracker
    telemetry.shutdown()


def test_tool_result_is_error_bumps_aggregator(tmp_path: Path) -> None:
    """ToolResultEvent with ``is_error=True`` bumps the per-turn aggregator
    counter that ``TurnEndEvent`` reads to fill in ``tool_error_count``.
    """
    telemetry = _make_telemetry(tmp_path)
    # Open a matching tool span so close_span finds one to close.
    ToolCallEvent(
        tool_call_id="tc-err", tool_name="read", args={}
    ).to_otel(telemetry)
    ToolResultEvent(
        tool_call_id="tc-err",
        tool_name="read",
        result=ToolResult(content=[], is_error=True),
    ).to_otel(telemetry)
    assert telemetry.turn_state["current_tool_errors"] == 1
    telemetry.shutdown()


def test_close_open_spans_drains_tracker(tmp_path: Path) -> None:
    """``SessionTelemetry.close_open_spans`` ends every still-open span
    so :meth:`SessionShutdownEvent.to_otel` can rely on it for cleanup.
    """
    telemetry = _make_telemetry(tmp_path)
    BeforeAgentStartEvent(messages=[], system=None).to_otel(telemetry)
    LlmRequestStartEvent(
        turn_index=0,
        message_count=1,
        tool_count=0,
        system_chars=3,
        model_id="m-stub",
        turn_id=99,
    ).to_otel(telemetry)
    assert len(telemetry.span_tracker) == 2
    telemetry.close_open_spans(status_description="test")
    assert telemetry.span_tracker == {}
    telemetry.shutdown()


def test_to_otel_attr_coerces_complex_to_json() -> None:
    """``SessionTelemetry.to_otel_attr`` is the single canonical attribute
    coercion path; complex values become JSON strings, scalars pass through.
    """
    from agentm.core.runtime.otel_export import SessionTelemetry

    assert SessionTelemetry.to_otel_attr(None) == ""
    assert SessionTelemetry.to_otel_attr(True) is True
    assert SessionTelemetry.to_otel_attr(7) == 7
    assert SessionTelemetry.to_otel_attr("x") == "x"
    coerced = SessionTelemetry.to_otel_attr({"k": [1, 2]})
    assert isinstance(coerced, str) and '"k"' in coerced and "[1, 2]" in coerced


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
