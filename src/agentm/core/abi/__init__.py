"""AgentM kernel: the langchain-free pluggable seed.

Public surface for the kernel layer described in
`.claude/designs/pluggable-architecture.md` §3. Everything exported here is
considered stable API for higher layers (runtime, scenarios, presenters).

The ABI is the type/Protocol foundation the other layers depend on. Its own
submodules import only stdlib + ``lib/``; the package ``__init__`` re-exports
concrete types so atoms can name them for annotation without importing
``agentm.core.runtime.*`` directly (the §11 atom contract forbids that).

It exposes:

- Message data model (``messages``)
- Bare tool contract (``tool``)
- Event bus + typed events (``events`` + ``bus``)
- LLM stream boundary (``stream``)
- The minimal :class:`AgentLoop` (``loop``)
"""

from __future__ import annotations

from .bus import (
    EventBus,
    EventBusObserver,
    Handler,
    ObserverCallback,
    ObserverRegistration,
)
from .events import (
    AgentEndEvent,
    AgentStartEvent,
    BeforeSendToLlmEvent,
    BudgetExhausted,
    BusPriority,
    ContextEvent,
    DecideTurnActionEvent,
    Event,
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
from .telemetry import SessionTelemetry
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
from .tool import FunctionTool, Tool, ToolContinue, ToolOutcome, ToolResult, ToolTerminate

# ``TraceReader`` is the canonical reader API for the OTLP/JSON event log.
# It now lives in ``lib/trace_reader`` (no runtime dependency); re-exported
# here so atoms can ``from agentm.core.abi import TraceReader`` per §11.
from agentm.core.lib.trace_reader import (  # noqa: E402
    LogRecord,
    SessionIdentity,
    Span,
    TraceReader,
    attr,
)

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
