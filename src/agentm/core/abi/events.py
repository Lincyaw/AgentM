"""v2 event taxonomy — frozen dataclasses for bus dispatch.

All events are frozen.  Handlers express intent through return values:

- ContextEvent handler returns ``list[AgentMessage] | None``
  (replacement message list, or None for no opinion)
- BeforeSendEvent handler returns a dict of field overrides or None
- DecideEvent handler returns a ``LoopAction | None``
- ToolCallEvent handler returns ``{"block": True, "reason": ...}``
  or ``{"rewrite": {...}}`` or None
- ToolResultEvent handler returns a replacement ToolResult or None
- Observation-only events: handler return is ignored
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

from agentm.core.abi.messages import AgentMessage, AssistantMessage
from agentm.core.abi.stream import AssistantStreamEvent, Model
from agentm.core.abi.tool import Tool, ToolOutcome, ToolResult
from agentm.core.abi.bus import Event
from agentm.core.abi.trajectory import Outcome, Turn, TurnMeta
from agentm.core.abi.trigger import Trigger


# --- Termination causes (reuse from v1, they're already frozen) -------------

@dataclass(frozen=True, slots=True)
class TerminationCause:
    final: ClassVar[bool] = False


@dataclass(frozen=True, slots=True)
class ModelEndTurn(TerminationCause):
    pass


@dataclass(frozen=True, slots=True)
class ToolTerminated(TerminationCause):
    tool_name: str = ""
    reason: str = ""


@dataclass(frozen=True, slots=True)
class MaxTurnsExhausted(TerminationCause):
    final: ClassVar[bool] = True


@dataclass(frozen=True, slots=True)
class SignalAborted(TerminationCause):
    final: ClassVar[bool] = True


@dataclass(frozen=True, slots=True)
class ProviderTruncated(TerminationCause):
    kind: Literal["max_tokens", "error"] = "max_tokens"


@dataclass(frozen=True, slots=True)
class BudgetExhausted(TerminationCause):
    final: ClassVar[bool] = True
    detail: str = ""


@dataclass(frozen=True, slots=True)
class NoPendingInput(TerminationCause):
    pass


# --- Loop actions -----------------------------------------------------------

@dataclass(frozen=True, slots=True)
class LoopAction:
    pass


@dataclass(frozen=True, slots=True)
class Step(LoopAction):
    pass


@dataclass(frozen=True, slots=True)
class Stop(LoopAction):
    cause: TerminationCause = field(default_factory=ModelEndTurn)


@dataclass(frozen=True, slots=True)
class Inject(LoopAction):
    messages: tuple[AgentMessage, ...] = ()


# --- Turn lifecycle events --------------------------------------------------

@dataclass(frozen=True, slots=True)
class TurnBeginEvent(Event):
    CHANNEL: ClassVar[str] = "turn_begin"
    index: int = 0
    trigger: Trigger | None = None


@dataclass(frozen=True, slots=True)
class TurnCommittedEvent(Event):
    CHANNEL: ClassVar[str] = "turn_committed"
    turn: Turn | None = None


# --- Context events ---------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ContextEvent(Event):
    """Handlers return ``list[AgentMessage] | None`` to replace messages."""
    CHANNEL: ClassVar[str] = "context"
    messages: tuple[AgentMessage, ...] = ()
    turn_index: int = 0


@dataclass(frozen=True, slots=True)
class BeforeSendEvent(Event):
    """Final preflight before LLM call.

    Handlers return a dict of overrides (``messages``, ``system``,
    ``tools``, ``model``) or None.  Last non-None wins per field.
    """
    CHANNEL: ClassVar[str] = "before_send"
    messages: tuple[AgentMessage, ...] = ()
    system: str | None = None
    tools: tuple[Tool, ...] = ()
    model: Model | None = None
    turn_index: int = 0


# --- Stream events ----------------------------------------------------------

@dataclass(frozen=True, slots=True)
class StreamDeltaEvent(Event):
    CHANNEL: ClassVar[str] = "stream_delta"
    turn_index: int = 0
    delta: AssistantStreamEvent | None = None


# --- Tool events ------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ToolCallEvent(Event):
    """Handlers return ``{"block": True, "reason": ...}`` or
    ``{"rewrite": {arg_overrides}}`` or None.
    """
    CHANNEL: ClassVar[str] = "tool_call"
    tool_call_id: str = ""
    tool_name: str = ""
    args: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolResultEvent(Event):
    """Handlers return a replacement ``ToolResult`` or None."""
    CHANNEL: ClassVar[str] = "tool_result"
    tool_call_id: str = ""
    tool_name: str = ""
    result: ToolResult | None = None


@dataclass(frozen=True, slots=True)
class ToolErrorEvent(Event):
    CHANNEL: ClassVar[str] = "tool_error"
    kind: str = ""
    tool_name: str = ""
    reason: str = ""
    exception: BaseException | None = None


# --- Decision events --------------------------------------------------------

@dataclass(frozen=True, slots=True)
class TurnObservation:
    turn_index: int = 0
    assistant_message: AssistantMessage | None = None
    tool_outcomes: tuple[tuple[str, ToolOutcome], ...] = ()
    default_action: LoopAction = field(default_factory=Step)


@dataclass(frozen=True, slots=True)
class DecideEvent(Event):
    """Handlers return a ``LoopAction`` or None (no opinion)."""
    CHANNEL: ClassVar[str] = "decide"
    observation: TurnObservation = field(default_factory=TurnObservation)


# --- Run lifecycle ----------------------------------------------------------

@dataclass(frozen=True, slots=True)
class BeforeRunEvent(Event):
    """Fired before the ReAct loop starts.

    Handlers return ``{"veto": TerminationCause}`` to abort, or
    ``{"messages": [...]}`` / ``{"system": "..."}`` to override.
    ``system`` override persists across all rounds; ``messages``
    override applies to round 0 only (subsequent rounds rebuild
    context from trajectory + policies).
    """
    CHANNEL: ClassVar[str] = "before_run"
    messages: tuple[AgentMessage, ...] = ()
    system: str | None = None


@dataclass(frozen=True, slots=True)
class RunEndEvent(Event):
    CHANNEL: ClassVar[str] = "run_end"
    outcome: Outcome | None = None
    meta: TurnMeta | None = None


# --- Session lifecycle ------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SessionReadyEvent(Event):
    CHANNEL: ClassVar[str] = "session_ready"
    session_id: str = ""
    tool_names: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SessionShutdownEvent(Event):
    CHANNEL: ClassVar[str] = "session_shutdown"


@dataclass(frozen=True, slots=True)
class ChildSessionStartEvent(Event):
    CHANNEL: ClassVar[str] = "child_session_start"
    child_session_id: str = ""
    parent_session_id: str = ""
    purpose: str = ""


@dataclass(frozen=True, slots=True)
class ChildSessionEndEvent(Event):
    CHANNEL: ClassVar[str] = "child_session_end"
    child_session_id: str = ""
    parent_session_id: str = ""


# --- Diagnostic -------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class DiagnosticEvent(Event):
    CHANNEL: ClassVar[str] = "diagnostic"
    level: Literal["info", "warning", "error"] = "info"
    source: str = ""
    message: str = ""


__all__ = [
    "BeforeRunEvent",
    "BeforeSendEvent",
    "BudgetExhausted",
    "ChildSessionEndEvent",
    "ChildSessionStartEvent",
    "ContextEvent",
    "DecideEvent",
    "DiagnosticEvent",
    "Inject",
    "LoopAction",
    "MaxTurnsExhausted",
    "ModelEndTurn",
    "NoPendingInput",
    "ProviderTruncated",
    "RunEndEvent",
    "SessionReadyEvent",
    "SessionShutdownEvent",
    "SignalAborted",
    "Step",
    "Stop",
    "StreamDeltaEvent",
    "TerminationCause",
    "ToolCallEvent",
    "ToolErrorEvent",
    "ToolResultEvent",
    "ToolTerminated",
    "TurnBeginEvent",
    "TurnCommittedEvent",
    "TurnObservation",
]
