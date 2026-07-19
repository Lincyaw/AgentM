"""AtomAPI — the complete surface atoms interact with.

15 methods with clear type contracts and no ``Any`` leaks.

SessionContext propagates through the session graph like Go's
context.Context — every child session inherits the parent's context
with updated depth/identity.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

from agentm.core.abi.messages import AgentMessage
from agentm.core.abi.stream import Model
from agentm.core.abi.tool import Tool
from agentm.core.abi.bus import EventBus, Handler
from agentm.core.abi.context import ContextPolicy
from agentm.core.abi.lifecycle import LifecycleHook
from agentm.core.abi.services import ServiceRegistry
from agentm.core.abi.store import TrajectoryStore
from agentm.core.abi.trajectory import Turn
from agentm.core.abi.codec import TriggerCodec
from agentm.core.abi.trigger import Trigger, TriggerRenderer

Unsubscribe = Callable[[], None]


@dataclass(frozen=True, slots=True)
class LoopConfig:
    """Driver loop budget — max turns and tool calls per session."""

    max_turns: int | None = None
    max_tool_calls: int | None = None


@dataclass(slots=True)
class AgentSessionConfig:
    """Configuration for spawning a child session.

    Atoms like ``sub_agent`` and ``workflow`` construct this to pass
    structured spawn parameters through ``spawn_child_session``.
    Fields are atom-facing and JSON-safe — no runtime types leak here.

    Two extension modes (mutually exclusive in intent):
    - ``extensions``: explicit full list — replaces scenario's list.
    - ``extra_extensions``: additions appended to the scenario's list.
    """

    cwd: str = ""
    scenario: str | None = None
    extensions: list[tuple[str, dict[str, Any]]] | None = None
    extra_extensions: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    extra_tools: list[Tool] = field(default_factory=list)
    atom_config_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    provider: tuple[str, dict[str, Any]] | None = None
    bus: EventBus | None = None
    store: TrajectoryStore | None = None
    initial_turns: list[Turn] = field(default_factory=list)
    tool_allowlist: list[str] | None = None
    purpose: str = "subagent"
    lineage: dict[str, Any] = field(default_factory=dict)
    loop_config: LoopConfig | None = None
    task_id: str | None = None
    persona: str | None = None
    experiment: dict[str, Any] | None = None
    trace_label: str | None = None
    session_id: str | None = None
    root_session_id: str | None = None
    parent_session_id: str | None = None


@dataclass(frozen=True, slots=True)
class SessionContext:
    """Propagating identity + config context for the session graph.

    Every child session gets a derived context with updated identity.
    Atoms read ``api.ctx`` to know who they are and where they sit in
    the graph.
    """

    session_id: str = ""
    root_session_id: str = ""
    parent_session_id: str | None = None
    depth: int = 0
    cwd: str = ""
    purpose: str = "root"
    scenario: str | None = None
    scenario_dir: str | None = None

    def child(
        self,
        *,
        session_id: str,
        purpose: str = "subagent",
        cwd: str | None = None,
        scenario: str | None = None,
    ) -> SessionContext:
        """Derive a child context with inherited + overridden fields."""
        return SessionContext(
            session_id=session_id,
            root_session_id=self.root_session_id,
            parent_session_id=self.session_id,
            depth=self.depth + 1,
            cwd=cwd or self.cwd,
            purpose=purpose,
            scenario=scenario or self.scenario,
            scenario_dir=self.scenario_dir,
        )


@runtime_checkable
class AtomAPI(Protocol):
    """The complete surface atoms interact with.

    Atoms receive this at ``install(api, config)`` time.  Every method
    is typed — no ``Any`` in the signatures.
    """

    # --- Context (identity + graph position) ---------------------------------

    @property
    def ctx(self) -> SessionContext:
        """Session identity, depth, cwd, purpose, scenario."""
        ...

    # --- Bus (event subscribe + emit) ----------------------------------------

    def on(
        self,
        channel: str,
        handler: Handler,
        *,
        priority: int = 500,
    ) -> Unsubscribe:
        """Subscribe to a bus channel.  Returns unsubscribe function."""
        ...

    @property
    def bus(self) -> EventBus:
        """Direct bus access for emit/emit_sync."""
        ...

    # --- Tool / Policy registration ------------------------------------------

    def register_tool(self, tool: Tool) -> None: ...

    def register_context_policy(
        self, policy: ContextPolicy, *, priority: int = 500
    ) -> None: ...

    def register_trigger_renderer(
        self, source: str, renderer: TriggerRenderer
    ) -> None: ...

    def register_trigger_codec(self, source: str, codec: TriggerCodec) -> None:
        """Register a codec for custom trigger persistence."""
        ...

    def register_lifecycle_hook(self, hook: LifecycleHook) -> None:
        """Register a hook for fork/resume/replay/abandon events."""
        ...

    # --- Trigger (unified input) ---------------------------------------------

    def push_trigger(self, trigger: Trigger) -> None:
        """Push a trigger into the session's queue."""
        ...

    def track_background(self) -> AbstractContextManager[None]:
        """Bracket a background unit so idle() waits for it."""
        ...

    # --- Session data (read-only) --------------------------------------------

    def get_messages(self) -> list[AgentMessage]:
        """Message list from committed turns (sync, no policies)."""
        ...

    def get_turns(self) -> Sequence[Turn]:
        """Committed turns (read-only)."""
        ...

    # --- Services (typed DI) -------------------------------------------------

    @property
    def services(self) -> ServiceRegistry: ...

    # --- Child session -------------------------------------------------------

    async def spawn(
        self,
        *,
        purpose: str = "subagent",
        tools: list[Tool] | None = None,
        system: str | None = None,
        model: Model | None = None,
        scenario: str | None = None,
        cwd: str | None = None,
        max_turns: int | None = None,
        extra_services: ServiceRegistry | None = None,
    ) -> "SpawnedSession":
        """Spawn a lightweight child inheriting parent's config.

        Only override what you need — everything else inherits from
        the parent (tools, model, system, policies, graph, store).
        """
        ...

    async def spawn_child_session(
        self,
        config: AgentSessionConfig,
    ) -> "SpawnedSession":
        """Spawn a fully-constructed child session from config.

        Goes through the factory pipeline: resolves scenario, loads
        extensions, installs atoms.  Use this when the child needs a
        different scenario or extension set from the parent.
        """
        ...

    # --- Model access --------------------------------------------------------

    @property
    def model(self) -> Model | None: ...

    @property
    def experiment(self) -> dict[str, Any] | None: ...


@runtime_checkable
class SpawnedSession(Protocol):
    """Handle to a spawned child session."""

    @property
    def session_id(self) -> str: ...

    async def prompt(self, text: str) -> None: ...

    async def run(self, text: str) -> list[AgentMessage]:
        """Start, prompt, wait, return messages (blocking convenience)."""
        ...

    def push_trigger(self, trigger: Trigger) -> None: ...

    def interrupt(self) -> None: ...

    async def idle(self, timeout: float | None = None) -> bool: ...

    async def shutdown(self) -> None: ...

    def get_messages(self) -> list[AgentMessage]: ...

    def status(self) -> dict[str, str | int | list[str]]: ...


__all__ = [
    "AgentSessionConfig",
    "AtomAPI",
    "LoopConfig",
    "SessionContext",
    "SpawnedSession",
    "Unsubscribe",
]
