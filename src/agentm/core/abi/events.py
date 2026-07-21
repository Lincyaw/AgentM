"""Event taxonomy for frozen bus-dispatch DTOs.

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
from typing import ClassVar, Literal

from agentm.core.abi.messages import AgentMessage, AssistantMessage, JsonValue
from agentm.core.abi.stream import AssistantStreamEvent, Model
from agentm.core.abi.termination import ModelEndTurn, TerminationCause
from agentm.core.abi.tool import Tool, ToolOutcome, ToolResult
from agentm.core.abi.bus import Event
from agentm.core.abi.trajectory import Outcome, Turn, TurnMeta
from agentm.core.abi.trigger import Trigger


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
    turn_index: int = 0
    turn_id: str = ""
    run_id: str = ""
    run_step: int = 0
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
    args: dict[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolResultEvent(Event):
    """Handlers return a replacement ``ToolResult`` or None."""

    CHANNEL: ClassVar[str] = "tool_result"
    tool_call_id: str = ""
    tool_name: str = ""
    result: ToolResult | None = None
    args: dict[str, JsonValue] = field(default_factory=dict)
    duration_ms: int | None = None
    exit_code: int | None = None
    result_content_hash: str | None = None


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
    """Fired when an external trigger starts a PromptRun.

    Handlers return ``{"veto": TerminationCause}`` to abort, or
    ``{"messages": [...]}`` / ``{"system": "..."}`` to override.
    ``system`` persists across continuation Turns; ``messages`` applies only
    to the initial Turn because later Turns rebuild context from committed
    trajectory state and policies.
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
    root_session_id: str = ""
    parent_session_id: str | None = None
    cwd: str = ""
    tool_names: tuple[str, ...] = ()
    command_names: tuple[str, ...] = ()
    extension_module_paths: tuple[str, ...] = ()
    model: Model | None = None


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
    final_message_count: int = 0
    error: str | None = None


# --- Diagnostic -------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DiagnosticEvent(Event):
    CHANNEL: ClassVar[str] = "diagnostic"
    level: Literal["info", "warning", "error"] = "info"
    source: str = ""
    message: str = ""
    phase: str = ""
    error_type: str | None = None
    error_detail: str | None = None
    turn_id: str | None = None
    turn_index: int | None = None
    checkpoint_id: str | None = None


# --- SDK domain events ------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExtensionInstallEvent(Event):
    CHANNEL: ClassVar[str] = "extension_install"
    name: str = ""
    module_path: str = ""
    phase: str = ""
    duration_ns: int = 0
    trigger: str = "session_start"
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ExtensionReloadEvent(Event):
    CHANNEL: ClassVar[str] = "extension_reload"
    name: str = ""
    trigger: str = ""
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ExtensionUnloadEvent(Event):
    CHANNEL: ClassVar[str] = "extension_unload"
    name: str = ""
    module_path: str = ""
    trigger: str = ""
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ResourcesDiscoverEvent(Event):
    CHANNEL: ClassVar[str] = "resources_discover"
    cwd: str = ""
    reason: str = ""


@dataclass(frozen=True, slots=True)
class ApiRegisterEvent(Event):
    CHANNEL: ClassVar[str] = "api_register"
    kind: str = ""
    name: str = ""
    extension: str = ""
    payload: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ApiSendUserMessageEvent(Event):
    CHANNEL: ClassVar[str] = "api_send_user_message"
    content: object = None
    extension: str = ""


@dataclass(frozen=True, slots=True)
class LlmRequestStartEvent(Event):
    CHANNEL: ClassVar[str] = "llm_request_start"
    turn_index: int = 0
    turn_id: str = ""
    model_id: str = ""
    message_count: int = 0
    tool_count: int = 0
    system_chars: int = 0
    system_text: str | None = None


@dataclass(frozen=True, slots=True)
class LlmRequestEndEvent(Event):
    CHANNEL: ClassVar[str] = "llm_request_end"
    turn_index: int = 0
    turn_id: str = ""
    chunk_count: int = 0
    duration_ns: int = 0
    error: str | None = None


__all__ = [
    "ApiRegisterEvent",
    "ApiSendUserMessageEvent",
    "BeforeRunEvent",
    "BeforeSendEvent",
    "ChildSessionEndEvent",
    "ChildSessionStartEvent",
    "ContextEvent",
    "DecideEvent",
    "DiagnosticEvent",
    "ExtensionInstallEvent",
    "ExtensionReloadEvent",
    "ExtensionUnloadEvent",
    "Inject",
    "LlmRequestEndEvent",
    "LlmRequestStartEvent",
    "LoopAction",
    "ResourcesDiscoverEvent",
    "RunEndEvent",
    "SessionReadyEvent",
    "SessionShutdownEvent",
    "Step",
    "Stop",
    "StreamDeltaEvent",
    "ToolCallEvent",
    "ToolErrorEvent",
    "ToolResultEvent",
    "TurnBeginEvent",
    "TurnCommittedEvent",
    "TurnObservation",
]
