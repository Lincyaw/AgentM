"""AtomAPI — the complete surface atoms interact with.

This is the v2 equivalent of v1's ``ExtensionAPI``.  14 methods instead
of 37, with clear type contracts and no ``Any`` leaks.

SessionContext propagates through the session graph like Go's
context.Context — every child session inherits the parent's context
with updated depth/identity.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Protocol, runtime_checkable

from agentm.core.abi.messages import AgentMessage
from agentm.core.abi.stream import Model
from agentm.core.abi.tool import Tool
from agentm.core.v2.abi.bus import EventBus, Handler
from agentm.core.v2.abi.context import ContextPolicy
from agentm.core.v2.abi.lifecycle import LifecycleHook
from agentm.core.v2.abi.services import ServiceRegistry
from agentm.core.v2.abi.trajectory import Turn
from agentm.core.v2.abi.trigger import Trigger, TriggerRenderer

if TYPE_CHECKING:
    pass

Unsubscribe = Callable[[], None]


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
        extra_services: dict[str, object] | None = None,
    ) -> "SpawnedSession":
        """Spawn a child session inheriting parent's config.

        Only override what you need — everything else inherits from
        the parent (tools, model, system, policies, graph, store).

        Returns a SpawnedSession handle for interaction.
        """
        ...

    # --- Model access --------------------------------------------------------

    @property
    def model(self) -> Model | None: ...


@runtime_checkable
class SpawnedSession(Protocol):
    """Handle to a spawned child session."""

    @property
    def session_id(self) -> str: ...

    async def prompt(self, text: str) -> None: ...

    def push_trigger(self, trigger: Trigger) -> None: ...

    def interrupt(self) -> None: ...

    async def idle(self, timeout: float | None = None) -> bool: ...

    async def shutdown(self) -> None: ...

    def get_messages(self) -> list[AgentMessage]: ...

    def status(self) -> dict[str, str | int | list[str]]: ...


__all__ = [
    "AtomAPI",
    "SessionContext",
    "SpawnedSession",
    "Unsubscribe",
]
