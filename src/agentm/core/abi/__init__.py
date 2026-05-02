"""AgentM kernel: the langchain-free pluggable seed.

Public surface for the kernel layer described in
`.claude/designs/pluggable-architecture.md` §3. Everything exported here is
considered stable API for higher layers (harness, scenarios, presenters).

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
    BeforeAgentEndEvent,
    BeforeSendToLlmEvent,
    ContextEvent,
    Event,
    EventBus,
    EventBusObserver,
    Handler,
    LlmRequestEndEvent,
    LlmRequestStartEvent,
    StreamDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
    TurnEndEvent,
    TurnStartEvent,
)
from .loop import AgentLoop, LoopConfig
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
    ToolCallEnd,
    ToolCallStart,
)
from .tool import FunctionTool, Tool, ToolResult

__all__ = [
    # loop
    "AgentLoop",
    "LoopConfig",
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
    "ToolResult",
    # events
    "AgentEndEvent",
    "AgentStartEvent",
    "BeforeAgentEndEvent",
    "BeforeSendToLlmEvent",
    "ContextEvent",
    "Event",
    "EventBus",
    "EventBusObserver",
    "LlmRequestEndEvent",
    "LlmRequestStartEvent",
    "Handler",
    "StreamDeltaEvent",
    "ToolCallEvent",
    "ToolResultEvent",
    "TurnEndEvent",
    "TurnStartEvent",
    # stream
    "AssistantStreamEvent",
    "MessageEnd",
    "Model",
    "StreamFn",
    "TextDelta",
    "ThinkingDelta",
    "ToolCallArgsDelta",
    "ToolCallEnd",
    "ToolCallStart",
]
