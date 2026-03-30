"""Core data types for the Agent Harness SDK."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Literal, Protocol, TypeAlias, TypedDict

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain_core.messages import BaseMessage


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------

class ModelProtocol(Protocol):
    """Protocol for LLM model interface.

    Supports both plain models and models with bound tools.
    """

    async def ainvoke(self, messages: list[Message]) -> object:
        """Invoke the model with messages. Returns a response object."""
        ...

    def with_structured_output(self, schema: type, *, method: str = "function_calling") -> "ModelProtocol":
        """Return a model configured for structured output."""
        ...

    def bind_tools(self, tools: list[JsonDict]) -> "ModelProtocol":
        """Bind tools to the model."""
        ...


# ---------------------------------------------------------------------------
# Type aliases for common patterns
# ---------------------------------------------------------------------------

# JSON-compatible value
JsonValue: TypeAlias = "str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]"

# Message type - supports dict and LangChain BaseMessage
Message: TypeAlias = "dict[str, JsonValue] | BaseMessage"

# Tool result type - all tools return strings (JSON-encoded)
ToolResult: TypeAlias = str

# JSON-compatible dict
JsonDict: TypeAlias = "dict[str, JsonValue]"

# Tool callable type - async or sync function returning ToolResult
ToolCallable: TypeAlias = "Callable[..., ToolResult | Awaitable[ToolResult]]"


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class AgentStatus(Enum):
    """Lifecycle status of an agent."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


@dataclass
class RunConfig:
    """Per-run execution configuration."""

    max_steps: int | None = None
    timeout: float | None = None
    thread_id: str | None = None  # for checkpointing
    metadata: JsonDict = field(default_factory=dict)


@dataclass
class AgentResult:
    """Outcome of an agent loop execution."""

    agent_id: str
    status: AgentStatus
    output: JsonValue = None  # final response (str, dict, Pydantic model)
    error: str | None = None
    duration_seconds: float | None = None
    steps: int = 0  # total LLM call rounds
    tool_calls: int = 0  # total tool invocations
    metadata: JsonDict = field(default_factory=dict)


@dataclass
class AgentEvent:
    """Streaming event from an agent loop.

    Event types: llm_start, llm_end, tool_start, tool_end,
    inject, complete, error.
    """

    type: Literal["llm_start", "llm_end", "tool_start", "tool_end", "inject", "complete", "error"]
    agent_id: str
    data: JsonDict = field(default_factory=dict)
    step: int = 0
    timestamp: str = ""


@dataclass
class AgentInfo:
    """Runtime status snapshot of an agent."""

    agent_id: str
    status: AgentStatus
    parent_id: str | None = None
    current_step: int = 0
    started_at: str | None = None
    metadata: JsonDict = field(default_factory=dict)
    result: AgentResult | None = None


@dataclass(frozen=True)
class LoopContext:
    """Read-only context available to middleware during a loop iteration."""

    agent_id: str
    step: int
    max_steps: int | None
    tool_call_count: int
    metadata: JsonDict


# ---------------------------------------------------------------------------
# TypedDicts for AgentSystem input / output
# ---------------------------------------------------------------------------


class AgentInput(TypedDict, total=False):
    """Typed input for AgentSystem.execute() / stream()."""

    task_description: str
    messages: list[dict[str, str]]


class AgentOutput(TypedDict):
    """Typed output returned by AgentSystem.execute()."""

    output: object
    status: str
    steps: int
