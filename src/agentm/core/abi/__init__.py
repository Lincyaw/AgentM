"""AgentM kernel: the langchain-free pluggable seed.

Public surface for the kernel layer described in
`.claude/designs/pluggable-architecture.md` §3. Everything exported here is
considered stable API for higher layers (runtime, scenarios, presenters).

The ABI is the type/Protocol foundation the other layers depend on. Its own
submodules import only stdlib; the package ``__init__`` additionally
re-exports a few concrete service *types* from ``core/_internal/`` and
``core/runtime/`` — ``FunctionTool``, ``SessionTelemetry``, ``TraceReader``
(+ its view dataclasses) — so atoms can name them for annotation without
importing ``agentm.core.runtime.*`` directly (the §11 atom contract forbids
that). Where the substrate owns the instance lifecycle (e.g.
``SessionTelemetry`` via ``api.get_session_telemetry()``), import the type to
annotate and obtain the live instance through ``ExtensionAPI`` — do not
construct it yourself. These re-exports stay within the recovery floor
(``core/abi`` + ``core/lib`` + ``core/runtime`` + stdlib), so they do not
weaken it.

It exposes:

- Message data model (``messages``)
- Bare tool contract (``tool``)
- Event bus + typed events (``events``)
- LLM stream boundary (``stream``)
- The minimal :class:`AgentLoop` (``loop``)
"""

from __future__ import annotations

from .events import (
    AgentEndEvent,
    AgentStartEvent,
    BeforeSendToLlmEvent,
    BudgetExhausted,
    BusPriority,
    ContextEvent,
    DecideTurnActionEvent,
    Event,
    EventBus,
    EventBusObserver,
    Handler,
    ObserverCallback,
    ObserverRegistration,
    Inject,
    LlmRequestEndEvent,
    LlmRequestStartEvent,
    LoopAction,
    MaxTurnsExhausted,
    ModelEndTurn,
    NoPendingInput,
    ProviderProtocolViolation,
    ProviderTruncated,
    SignalAborted,
    Step,
    Stop,
    StreamDeltaEvent,
    TerminationCause,
    ToolCallEvent,
    ToolErrorEvent,
    ToolResultEvent,
    ToolTerminated,
    TurnEndEvent,
    TurnObservation,
    TurnStartEvent,
)
from .loop import AgentLoop, LoopConfig
from .manifest import ExtensionManifest as ExtensionManifest
from .presenter import PHASE_GLYPHS, Phase
from .provider import ProviderConfig, ProviderManifest, ProviderResolver
from .retry import RetryPolicy
from .session_store import SessionState, SessionStore
from .messages import (
    AgentMessage,
    AssistantContent,
    AssistantMessage,
    ImageContent,
    TextContent,
    ThinkingBlock,
    ToolCallBlock,
    ToolResultBlock,
    ToolResultMessage,
    Usage,
    UserMessage,
    text_message,
    tool_result,
)
from .stream import (
    AssistantStreamEvent,
    MessageEnd,
    Model,
    StreamFn,
    TextDelta,
    ThinkingDelta,
    ToolCallArgsDelta,
    ToolCallArgsParseError,
    ToolCallEnd,
    ToolCallStart,
)
from .termination import (
    Aborted,
    EndTurn,
    MaxTokens,
    PauseTurn,
    ProviderError,
    TerminationHint,
    ToolUseExpected,
    VendorSpecific,
)
from .tool import Tool, ToolContinue, ToolOutcome, ToolResult, ToolTerminate

from agentm.core._internal.tools import FunctionTool  # noqa: E402

# Lazy re-exports from runtime (break circular import chain).
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentm.core.runtime.otel_export import SessionTelemetry as SessionTelemetry
    from agentm.core.runtime.trace_reader import (
        LogRecord as LogRecord,
        SessionIdentity as SessionIdentity,
        Span as Span,
        TraceReader as TraceReader,
        attr as attr,
    )


def __getattr__(name: str) -> object:
    if name == "SessionTelemetry":
        from agentm.core.runtime.otel_export import SessionTelemetry
        return SessionTelemetry
    if name in ("LogRecord", "SessionIdentity", "Span", "TraceReader", "attr"):
        from agentm.core.runtime import trace_reader as _tr
        return getattr(_tr, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "Aborted", "AgentEndEvent", "AgentLoop", "AgentMessage", "AgentStartEvent",
    "AssistantContent", "AssistantMessage", "AssistantStreamEvent",
    "BeforeSendToLlmEvent", "BudgetExhausted", "BusPriority",
    "ContextEvent", "DecideTurnActionEvent",
    "EndTurn", "Event", "EventBus", "EventBusObserver", "ExtensionManifest",
    "FunctionTool", "Handler",
    "ImageContent", "Inject",
    "LlmRequestEndEvent", "LlmRequestStartEvent",
    "LogRecord", "LoopAction", "LoopConfig",
    "MaxTokens", "MaxTurnsExhausted", "MessageEnd", "Model", "ModelEndTurn",
    "NoPendingInput",
    "ObserverCallback", "ObserverRegistration",
    "PHASE_GLYPHS", "PauseTurn", "Phase", "ProviderConfig", "ProviderError",
    "ProviderManifest", "ProviderProtocolViolation", "ProviderResolver",
    "ProviderTruncated",
    "RetryPolicy",
    "SessionIdentity", "SessionState", "SessionStore", "SessionTelemetry",
    "SignalAborted", "Span", "Step", "Stop", "StreamDeltaEvent", "StreamFn",
    "TerminationCause", "TerminationHint",
    "TextContent", "TextDelta", "ThinkingBlock", "ThinkingDelta",
    "Tool", "ToolCallArgsDelta", "ToolCallArgsParseError", "ToolCallBlock",
    "ToolCallEnd", "ToolCallEvent", "ToolCallStart",
    "ToolContinue", "ToolErrorEvent", "ToolOutcome", "ToolResult",
    "ToolResultBlock", "ToolResultEvent", "ToolResultMessage",
    "ToolTerminate", "ToolTerminated", "ToolUseExpected",
    "TraceReader", "TurnEndEvent", "TurnObservation", "TurnStartEvent",
    "Usage", "UserMessage", "VendorSpecific",
    "attr", "text_message", "tool_result",
]
