"""Otel tracing atom: exercise the install path with an in-memory exporter
and assert that the canonical span surface (session / turn / llm / tool /
event / handler) is recorded as the user expects.

Fail-stop position: this guards the "everything has a span" contract the
default scenario relies on. If a future refactor of the atom or the event
schema drops one of these spans, traces in the user's collector silently
lose entire categories.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agentm.core.abi import EventBus
from agentm.core.abi.events import (
    LlmRequestEndEvent,
    LlmRequestStartEvent,
    SessionShutdownEvent,
    ToolCallEvent,
    ToolResultEvent,
    TurnEndEvent,
    TurnStartEvent,
)
from agentm.core.abi.messages import AssistantMessage
from agentm.core.abi.tool import ToolResult
from agentm.core.runtime.extension import _ExtensionAPIImpl
from agentm.extensions.builtin import otel_tracing


class _SessionView:
    def get_messages(self) -> list[Any]:
        return []

    def get_branch(self) -> list[Any]:
        return []

    def get_leaf_id(self) -> str | None:
        return None

    def get_entry(self, entry_id: str) -> Any | None:
        del entry_id
        return None

    def get_loop_config(self) -> Any:
        return None

    def append_entry(self, type: str, payload: Any, parent_id: str | None = None) -> str:
        del type, payload, parent_id
        return "entry"


def _api(
    tmp_path: Path,
    *,
    session_id: str = "session-otel",
    root_session_id: str | None = None,
    parent_session_id: str | None = None,
) -> _ExtensionAPIImpl:
    from agentm.core.runtime.extension import build_extension_api_scope

    scope = build_extension_api_scope(
        bus=EventBus(),
        cwd=str(tmp_path),
        session_id=session_id,
        root_session_id=root_session_id,
        parent_session_id=parent_session_id,
        session=_SessionView(),
        tools=[],
        commands={},
        providers={},
        renderers={},
        pending_user_messages=[],
        model_getter=lambda: None,
        provider_getter=lambda: None,
    )
    return _ExtensionAPIImpl(scope)


@pytest.fixture
def exporter():
    pytest.importorskip("opentelemetry")
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exp = InMemorySpanExporter()
    # OTel only allows one TracerProvider per process; other tests in the
    # suite may have already installed one. Attach our processor to the
    # current provider when possible, otherwise install a fresh one.
    current = trace.get_tracer_provider()
    if hasattr(current, "add_span_processor"):
        current.add_span_processor(SimpleSpanProcessor(exp))
    else:
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exp))
        trace.set_tracer_provider(provider)
    # Pretend our atom already configured the provider so it doesn't try
    # to install an OTLP exporter pointing at localhost:4317.
    otel_tracing._provider_installed = True
    yield exp
    exp.clear()


@pytest.mark.asyncio
async def test_otel_spans_cover_session_turn_llm_tool_and_events(
    tmp_path: Path,
    exporter: Any,
) -> None:
    api = _api(tmp_path)
    otel_tracing.install(api, {})

    # Drive a representative slice of the event surface.
    await api.events.emit(TurnStartEvent.CHANNEL, TurnStartEvent(turn_index=0))
    await api.events.emit(
        LlmRequestStartEvent.CHANNEL,
        LlmRequestStartEvent(
            turn_index=0,
            message_count=1,
            tool_count=2,
            system_chars=10,
            model_id="m-1",
        ),
    )
    await api.events.emit(
        LlmRequestEndEvent.CHANNEL,
        LlmRequestEndEvent(turn_index=0, chunk_count=3, duration_ns=1_000),
    )
    await api.events.emit(
        ToolCallEvent.CHANNEL,
        ToolCallEvent(tool_call_id="tc-1", tool_name="bash", args={"cmd": "ls"}),
    )
    await api.events.emit(
        ToolResultEvent.CHANNEL,
        ToolResultEvent(
            tool_call_id="tc-1",
            tool_name="bash",
            result=ToolResult(content=[], is_error=False),
        ),
    )
    await api.events.emit(
        TurnEndEvent.CHANNEL,
        TurnEndEvent(
            turn_index=0,
            message=AssistantMessage(
                role="assistant",
                content=[],
                timestamp=0.0,
                stop_reason="end_turn",
            ),
        ),
    )
    await api.events.emit(
        SessionShutdownEvent.CHANNEL, SessionShutdownEvent(cwd=str(tmp_path))
    )

    spans = exporter.get_finished_spans()
    names = {s.name for s in spans}

    # Lifecycle / turn / llm / tool spans must be present.
    assert "agentm.session" in names
    assert "agentm.turn" in names
    assert "agentm.llm.request" in names
    assert "agentm.tool.execute" in names

    # Generic per-emit spans cover every non-excluded channel.
    assert "agentm.event:turn_start" in names
    assert "agentm.event:turn_end" in names
    assert "agentm.event:llm_request_start" in names
    assert "agentm.event:tool_call" in names

    # Tool span correlates to a single tool_call_id and carries the name.
    tool_spans = [s for s in spans if s.name == "agentm.tool.execute"]
    assert len(tool_spans) == 1
    assert tool_spans[0].attributes["agentm.tool.name"] == "bash"
    assert tool_spans[0].attributes["agentm.tool.call_id"] == "tc-1"
    assert tool_spans[0].attributes["agentm.tool.is_error"] is False

    # Session span carries the session_id so collector consumers can group
    # everything for a single agent run.
    session_spans = [s for s in spans if s.name == "agentm.session"]
    assert len(session_spans) == 1
    assert session_spans[0].attributes["agentm.session_id"] == "session-otel"


@pytest.mark.asyncio
async def test_w3c_traceparent_env_links_session_to_upstream_trace(
    tmp_path: Path,
    exporter: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cross-process linkage uses the standard ``opentelemetry.propagate``
    API on ``os.environ``. When the host sets the W3C ``TRACEPARENT`` env
    var (the canonical mechanism for OTel-instrumented hosts launching
    sub-processes), a fresh AgentM session must continue that trace
    instead of starting a new one.

    No AgentM-specific propagation: just stock W3C TraceContext.
    """
    parent_trace = "11112222333344445555666677778888"  # 32 hex / 16 bytes
    parent_span = "aaaabbbbccccdddd"  # 16 hex / 8 bytes
    monkeypatch.setenv(
        "TRACEPARENT", f"00-{parent_trace}-{parent_span}-01"
    )

    api = _api(tmp_path, session_id="downstream-session")
    otel_tracing.install(api, {})
    await api.events.emit(
        SessionShutdownEvent.CHANNEL, SessionShutdownEvent(cwd=str(tmp_path))
    )

    spans = exporter.get_finished_spans()
    session_spans = [s for s in spans if s.name == "agentm.session"]
    assert len(session_spans) == 1
    span = session_spans[0]
    # Trace continues the upstream W3C trace.
    assert format(span.get_span_context().trace_id, "032x") == parent_trace
    # Parent linkage points at the upstream span.
    assert span.parent is not None
    assert format(span.parent.span_id, "016x") == parent_span
    # AgentM's own session_id is preserved as a logical attribute.
    assert span.attributes["agentm.session_id"] == "downstream-session"


@pytest.mark.asyncio
async def test_top_level_session_starts_fresh_trace_when_no_traceparent(
    tmp_path: Path,
    exporter: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without an upstream ``TRACEPARENT``, a top-level session starts a
    fresh OTel trace — the SDK's IdGenerator picks the trace_id, AgentM
    doesn't try to influence it.
    """
    monkeypatch.delenv("TRACEPARENT", raising=False)
    api = _api(tmp_path, session_id="solo-session")
    otel_tracing.install(api, {})
    await api.events.emit(
        SessionShutdownEvent.CHANNEL, SessionShutdownEvent(cwd=str(tmp_path))
    )

    spans = exporter.get_finished_spans()
    session_spans = [s for s in spans if s.name == "agentm.session"]
    assert len(session_spans) == 1
    span = session_spans[0]
    # Top-level: no parent linkage.
    assert span.parent is None
    # SDK-generated trace_id is non-zero.
    assert span.get_span_context().trace_id != 0
