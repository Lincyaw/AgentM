"""Backward-compatible stubs for v1 types deleted during v2 migration.

These allow atoms written against the v1 API surface to import without
error. The types are intentionally minimal: they exist so that
``from agentm.core.abi import ExtensionAPI`` resolves, and so that
event CHANNEL class vars remain available for bus subscriptions.

Atom migration to the v2 API is tracked per-atom; once all atoms are
migrated, this file is deleted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol, runtime_checkable

from agentm.core.abi.bus import Event


# --- ExtensionAPI (v1 atom-facing API) ------------------------------------

@runtime_checkable
class ExtensionAPI(Protocol):
    """Stub of the v1 ExtensionAPI.  Atoms type-hint install(api, config)
    against this.  At runtime, a compat adapter or the v2 Session is passed.
    """

    @property
    def session_id(self) -> str: ...

    @property
    def root_session_id(self) -> str: ...

    @property
    def parent_session_id(self) -> str | None: ...

    @property
    def cwd(self) -> str: ...

    @property
    def scenario_dir(self) -> str | None: ...

    @property
    def purpose(self) -> str: ...

    @property
    def scenario(self) -> str | None: ...

    @property
    def tools(self) -> list[Any]: ...

    @property
    def model(self) -> Any: ...

    @property
    def events(self) -> Any: ...

    @property
    def provider(self) -> Any: ...

    def on(self, channel: str, handler: Any, *, priority: int = 500) -> Any: ...

    def register_tool(self, tool: Any) -> None: ...

    def register_command(self, name: str, spec: Any) -> None: ...

    def register_provider(self, name: str, config: Any) -> None: ...

    def has_provider(self, name: str) -> bool: ...

    def register_operations(self, *, bash: Any) -> None: ...

    def register_message_renderer(self, custom_type: str, renderer: Any) -> None: ...

    def register_tool_renderer(self, tool_name: str, renderer: Any) -> None: ...

    def register_resource_writer(self, writer: Any) -> None: ...

    def post_inbox(self, *, source: str, payload: Any, dedup_key: str | None = None, terminal: bool = False) -> None: ...

    def track_background(self) -> Any: ...

    def send_user_message(self, content: str | list[Any]) -> None: ...

    def set_service(self, name: str, obj: Any) -> None: ...

    def get_service(self, name: str) -> Any: ...

    def get_operations(self) -> Any: ...

    def get_project_layout(self) -> Any: ...

    def get_resource_writer(self) -> Any: ...

    def get_session_telemetry(self) -> Any: ...

    async def spawn_child_session(self, config: Any) -> Any: ...

    def reload_atom(self, name: str, new_source: str, *, agent_initiated: bool = True, rationale: str | None = None) -> Any: ...

    def install_atom(self, *, name: str, source: str, target_path: str | None = None, config: dict[str, Any] | None = None, rationale: str | None = None, agent_initiated: bool = True) -> Any: ...

    def unload_atom(self, name: str, *, agent_initiated: bool = True) -> Any: ...

    def list_atoms(self) -> list[Any]: ...

    def freeze_current(self, name: str) -> str: ...

    def is_constitution_path(self, path: str) -> bool: ...

    @property
    def lineage(self) -> dict[str, Any] | None: ...

    @property
    def experiment(self) -> dict[str, Any] | None: ...

    @property
    def catalog(self) -> Any: ...

    @property
    def session(self) -> Any: ...

    async def wait_inbox_nonempty(self, sources: frozenset[str] | None = None) -> bool: ...


class ExtensionLoadError(Exception):
    def __init__(self, module_path: str, cause: BaseException | None = None) -> None:
        self.module_path = module_path
        self.cause = cause
        super().__init__(f"failed to load extension {module_path!r}: {cause}")


class ExtensionStaleError(RuntimeError):
    pass


class CommandSpec:
    """Stub for v1 command registration."""
    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


class CommandDispatcher(Protocol):
    def dispatch(self, command: str, args: str) -> Any: ...


class InstallAtomResult:
    pass

class ReloadResult:
    pass

class UnloadAtomResult:
    pass

class AtomInfo:
    pass

Renderer = Any


# --- v1 event types that don't exist in v2 --------------------------------
# These are frozen stubs with CHANNEL class vars so atoms that subscribe
# to them via api.on(SomeEvent.CHANNEL, handler) still work at import time.
# The handlers simply won't fire since the v2 driver doesn't emit these.


@dataclass(frozen=True, slots=True)
class AgentStartEvent(Event):
    CHANNEL: ClassVar[str] = "agent_start"


@dataclass(frozen=True, slots=True)
class AgentEndEvent(Event):
    CHANNEL: ClassVar[str] = "agent_end"


@dataclass(frozen=True, slots=True)
class BeforeAgentStartEvent(Event):
    CHANNEL: ClassVar[str] = "before_agent_start"


@dataclass(frozen=True, slots=True)
class BeforeSendToLlmEvent(Event):
    """v1 compat — v2 uses BeforeSendEvent with CHANNEL='before_send'."""
    CHANNEL: ClassVar[str] = "before_send_to_llm"
    messages: list[Any] = field(default_factory=list)
    system: str | None = None
    tools: list[Any] = field(default_factory=list)
    model: Any = None


@dataclass(frozen=True, slots=True)
class DecideTurnActionEvent(Event):
    """v1 compat — v2 uses DecideEvent with CHANNEL='decide'."""
    CHANNEL: ClassVar[str] = "decide_turn_action"
    observation: Any = None


@dataclass(frozen=True, slots=True)
class TurnStartEvent(Event):
    CHANNEL: ClassVar[str] = "turn_start"
    index: int = 0


@dataclass(frozen=True, slots=True)
class TurnEndEvent(Event):
    CHANNEL: ClassVar[str] = "turn_end"
    index: int = 0


@dataclass(frozen=True, slots=True)
class LlmRequestStartEvent(Event):
    CHANNEL: ClassVar[str] = "llm_request_start"


@dataclass(frozen=True, slots=True)
class LlmRequestEndEvent(Event):
    CHANNEL: ClassVar[str] = "llm_request_end"
    usage: Any = None


@dataclass(frozen=True, slots=True)
class InputEvent(Event):
    CHANNEL: ClassVar[str] = "input"


@dataclass(frozen=True, slots=True)
class BackgroundActivityEvent(Event):
    CHANNEL: ClassVar[str] = "background_activity"


@dataclass(frozen=True, slots=True)
class BeforeCompactEvent(Event):
    CHANNEL: ClassVar[str] = "before_compact"


@dataclass(frozen=True, slots=True)
class AfterCompactEvent(Event):
    CHANNEL: ClassVar[str] = "after_compact"


@dataclass(frozen=True, slots=True)
class ChildSessionExtendingEvent(Event):
    CHANNEL: ClassVar[str] = "child_session_extending"


@dataclass(frozen=True, slots=True)
class CostBudgetExceededEvent(Event):
    CHANNEL: ClassVar[str] = "cost_budget_exceeded"


@dataclass(frozen=True, slots=True)
class PlanSubmittedEvent(Event):
    CHANNEL: ClassVar[str] = "plan_submitted"


@dataclass(frozen=True, slots=True)
class MessagePersistedEvent(Event):
    CHANNEL: ClassVar[str] = "message_persisted"


@dataclass(frozen=True, slots=True)
class MessageAppendedEvent(Event):
    CHANNEL: ClassVar[str] = "message_appended"


@dataclass(frozen=True, slots=True)
class SessionHeaderEmittedEvent(Event):
    CHANNEL: ClassVar[str] = "session_header_emitted"


@dataclass(frozen=True, slots=True)
class EntryAppendedEvent(Event):
    CHANNEL: ClassVar[str] = "entry_appended"


@dataclass(frozen=True, slots=True)
class ResolveSubagentEvent(Event):
    CHANNEL: ClassVar[str] = "resolve_subagent"


@dataclass(frozen=True, slots=True)
class ExtensionInstallEvent(Event):
    CHANNEL: ClassVar[str] = "extension_install"
    extension: str = ""


@dataclass(frozen=True, slots=True)
class ExtensionReloadEvent(Event):
    CHANNEL: ClassVar[str] = "extension_reload"


@dataclass(frozen=True, slots=True)
class BeforeInstallAtomEvent(Event):
    CHANNEL: ClassVar[str] = "before_install_atom"


@dataclass(frozen=True, slots=True)
class BeforeUnloadAtomEvent(Event):
    CHANNEL: ClassVar[str] = "before_unload_atom"


@dataclass(frozen=True, slots=True)
class CommandDispatchedEvent(Event):
    CHANNEL: ClassVar[str] = "command_dispatched"


@dataclass(frozen=True, slots=True)
class ExtensionUnloadEvent(Event):
    CHANNEL: ClassVar[str] = "extension_unload"


@dataclass(frozen=True, slots=True)
class ApiRegisterEvent(Event):
    CHANNEL: ClassVar[str] = "api_register"
    kind: str = ""
    name: str = ""
    extension: str = ""
    payload: Any = None


@dataclass(frozen=True, slots=True)
class ApiSendUserMessageEvent(Event):
    CHANNEL: ClassVar[str] = "api_send_user_message"
    extension: str = ""
    content: Any = None


@dataclass(frozen=True, slots=True)
class ResourcesDiscoverEvent(Event):
    CHANNEL: ClassVar[str] = "resources_discover"


@dataclass(frozen=True, slots=True)
class ResourceWriteEvent(Event):
    CHANNEL: ClassVar[str] = "resource_write"


# --- v1 loop types --------------------------------------------------------

@dataclass(frozen=True, slots=True)
class LoopConfig:
    """Stub for v1 LoopConfig."""
    max_turns: int = 200


class AgentLoop:
    """Stub for v1 AgentLoop."""
    pass


# --- v1 session types -----------------------------------------------------

@dataclass(frozen=True, slots=True)
class AgentSessionConfig:
    """Stub for v1 AgentSessionConfig."""
    pass


class SessionState:
    """Stub for v1 SessionState."""
    pass


class SessionStore(Protocol):
    """Stub for v1 SessionStore."""
    pass


# --- v1 session entry types -----------------------------------------------

ENTRY_TYPE_MESSAGE = "message"
ENTRY_TYPE_COMPACTION = "compaction"
ENTRY_TYPE_BRANCH_SUMMARY = "branch_summary"
ENTRY_MATERIALIZERS: dict[str, Any] = {}


@dataclass(slots=True)
class SessionEntry:
    """Stub for v1 SessionEntry."""
    entry_type: str = ""
    data: Any = None


# --- v1 compaction types --------------------------------------------------

@dataclass(frozen=True, slots=True)
class CompactionDetails:
    pass

@dataclass(frozen=True, slots=True)
class CompactionResult:
    pass

class CompactionSettings:
    pass

@dataclass(frozen=True, slots=True)
class CompactionPrompts:
    summarization_system: str = ""
    update_summarization: str = ""
    system: str = ""
    summarize: str = ""
    update: str = ""
    branch: str = ""
    branch_preamble: str = ""

@dataclass(frozen=True, slots=True)
class ContextUsageSnapshot:
    pass

PROMPT_SUMMARIZATION = ""
PROMPT_SUMMARIZATION_SYSTEM = ""
PROMPT_UPDATE_SUMMARIZATION = ""
PROMPT_BRANCH_SUMMARY = ""
PROMPT_BRANCH_SUMMARY_PREAMBLE = ""


# --- v1 event bus types ---------------------------------------------------

class EventBusObserver:
    """Stub for v1 EventBusObserver."""
    pass

class ObserverCallback:
    """Stub for v1 ObserverCallback."""
    pass

class ObserverRegistration:
    """Stub for v1 ObserverRegistration."""
    pass


# --- v1 event action types (re-exported from v2 with aliases) -------------

# HookContract is a v1 concept — stub it
class HookContract:
    pass


# MUTABLE_EVENT_FIELDS_BY_TYPE is a v1 concept — no-op dict
MUTABLE_EVENT_FIELDS_BY_TYPE: dict[type, frozenset[str]] = {}


__all__ = [
    # ExtensionAPI surface
    "ExtensionAPI",
    "ExtensionLoadError",
    "ExtensionStaleError",
    "CommandSpec",
    "CommandDispatcher",
    "InstallAtomResult",
    "ReloadResult",
    "UnloadAtomResult",
    "AtomInfo",
    "Renderer",
    # v1 event types
    "AgentStartEvent",
    "AgentEndEvent",
    "BeforeAgentStartEvent",
    "BeforeSendToLlmEvent",
    "DecideTurnActionEvent",
    "TurnStartEvent",
    "TurnEndEvent",
    "LlmRequestStartEvent",
    "LlmRequestEndEvent",
    "InputEvent",
    "BackgroundActivityEvent",
    "BeforeCompactEvent",
    "AfterCompactEvent",
    "ChildSessionExtendingEvent",
    "CostBudgetExceededEvent",
    "PlanSubmittedEvent",
    "MessagePersistedEvent",
    "MessageAppendedEvent",
    "SessionHeaderEmittedEvent",
    "EntryAppendedEvent",
    "ResolveSubagentEvent",
    "ExtensionInstallEvent",
    "ExtensionReloadEvent",
    "BeforeInstallAtomEvent",
    "BeforeUnloadAtomEvent",
    "CommandDispatchedEvent",
    "ExtensionUnloadEvent",
    "ApiRegisterEvent",
    "ApiSendUserMessageEvent",
    "ResourcesDiscoverEvent",
    "ResourceWriteEvent",
    # v1 loop types
    "LoopConfig",
    "AgentLoop",
    # v1 session types
    "AgentSessionConfig",
    "SessionState",
    "SessionStore",
    # v1 session entry types
    "ENTRY_TYPE_MESSAGE",
    "ENTRY_TYPE_COMPACTION",
    "ENTRY_TYPE_BRANCH_SUMMARY",
    "ENTRY_MATERIALIZERS",
    "SessionEntry",
    # v1 compaction types
    "CompactionDetails",
    "CompactionResult",
    "CompactionSettings",
    "CompactionPrompts",
    "ContextUsageSnapshot",
    "PROMPT_SUMMARIZATION",
    "PROMPT_SUMMARIZATION_SYSTEM",
    "PROMPT_UPDATE_SUMMARIZATION",
    "PROMPT_BRANCH_SUMMARY",
    "PROMPT_BRANCH_SUMMARY_PREAMBLE",
    # v1 bus types
    "EventBusObserver",
    "ObserverCallback",
    "ObserverRegistration",
    "HookContract",
    "MUTABLE_EVENT_FIELDS_BY_TYPE",
]
