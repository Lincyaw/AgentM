"""AgentM kernel: the langchain-free pluggable seed.

Public surface for the kernel layer described in
`.claude/designs/pluggable-architecture.md` §3. Everything exported here is
considered stable API for higher layers (runtime, scenarios, presenters).

The kernel is the bottom layer: it does not import from any other AgentM
module. It exposes:

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

__all__ = [
    # loop
    "AgentLoop",
    "LoopConfig",
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
