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

# ``FunctionTool`` is a concrete adapter — it lives outside the ABI surface
# proper (under ``core/_internal/tools.py``) but is re-exported here for
# ergonomic access from atoms and tests. Importing _internal from the abi
# __init__ is allowed: this module is itself part of the constitution.
from agentm.core._internal.tools import FunctionTool  # noqa: E402

# ``SessionTelemetry`` is similarly a runtime-side service object (file
# exporters + batch processors live in ``core.runtime.otel_export``), but
# atoms need the *type* to annotate handler captures of
# ``api.get_session_telemetry()``. Re-exporting through the ABI is the
# established service-facade pattern (cf. ``FunctionTool`` above) and lets
# the §11 validator stay strict about atoms importing
# ``agentm.core.runtime.*`` directly.
#
# ``TraceReader`` is the canonical reader API for the OTLP/JSON event log
# (`.claude/designs/single-event-log.md` PR-G). Atoms cannot import
# ``core.runtime.*`` directly under §11, so we re-export the read-only
# surface — class + dataclass views + the ``attr`` helper — through the
# ABI. The runtime impl stays in ``core.runtime.trace_reader``.
#
# Both are imported lazily via ``__getattr__`` to break a circular import
# chain: ``agentm.extensions`` -> ``core.abi`` -> ``core.runtime.*`` ->
# ``agentm.extensions`` (see audit finding 1). TYPE_CHECKING keeps static
# analysers happy; __getattr__ resolves them at runtime on first access.
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
    # loop
    "AgentLoop",
    "LoopConfig",
    # extension manifest (atom-facing declaration record)
    "ExtensionManifest",
    # presenter view contract
    "PHASE_GLYPHS",
    "Phase",
    # provider
    "ProviderConfig",
    "ProviderManifest",
    "ProviderResolver",
    # retry
    "RetryPolicy",
    # session store
    "SessionState",
    "SessionStore",
    # telemetry (re-exported from core.runtime.otel_export — see comment
    # in __init__ body for the service-facade rationale).
    "SessionTelemetry",
    # trace reader (re-exported from core.runtime.trace_reader — atoms
    # use this to read the OTLP/JSON event log without violating §11).
    "LogRecord",
    "SessionIdentity",
    "Span",
    "TraceReader",
    "attr",
    # messages
    "AgentMessage",
    "AssistantContent",
    "AssistantMessage",
    "ImageContent",
    "TextContent",
    "ThinkingBlock",
    "ToolCallBlock",
    "ToolResultBlock",
    "ToolResultMessage",
    "Usage",
    "UserMessage",
    "text_message",
    "tool_result",
    # tool
    "FunctionTool",
    "Tool",
    "ToolContinue",
    "ToolOutcome",
    "ToolResult",
    "ToolTerminate",
    # events
    "AgentEndEvent",
    "AgentStartEvent",
    "BeforeSendToLlmEvent",
    "BudgetExhausted",
    "BusPriority",
    "ContextEvent",
    "DecideTurnActionEvent",
    "Event",
    "EventBus",
    "EventBusObserver",
    "Inject",
    "LlmRequestEndEvent",
    "LlmRequestStartEvent",
    "LoopAction",
    "Handler",
    "ObserverCallback",
    "ObserverRegistration",
    "MaxTurnsExhausted",
    "ModelEndTurn",
    "NoPendingInput",
    "ProviderProtocolViolation",
    "ProviderTruncated",
    "SignalAborted",
    "Step",
    "Stop",
    "StreamDeltaEvent",
    "TerminationCause",
    "ToolCallEvent",
    "ToolErrorEvent",
    "ToolResultEvent",
    "ToolTerminated",
    "TurnEndEvent",
    "TurnObservation",
    "TurnStartEvent",
    # stream
    "AssistantStreamEvent",
    "MessageEnd",
    "Model",
    "StreamFn",
    "TextDelta",
    "ThinkingDelta",
    "ToolCallArgsDelta",
    "ToolCallArgsParseError",
    "ToolCallEnd",
    "ToolCallStart",
    # termination
    "Aborted",
    "EndTurn",
    "MaxTokens",
    "PauseTurn",
    "ProviderError",
    "TerminationHint",
    "ToolUseExpected",
    "VendorSpecific",
]
