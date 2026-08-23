"""A session as data: what one is built from, and what one can be rebuilt as.

Two DTOs and no behaviour.  ``SessionRuntimeConfig`` is the inbound shape --
everything the factory has resolved by the time a runtime is constructed --
and ``CompositionSnapshot`` is the outbound one, the formal surface a spawn,
fork or child build reads instead of reaching into session privates.

They live beside ``session_core`` rather than in it because they are the only
part of it that says nothing about a running session, and because a module
that answers "what is a session made of" is easier to read without the
thousand lines that answer "what does one do".  ``session_core`` re-exports
both, so nothing importing them has to know they moved.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from agentm.core.abi.bus import EventBus
from agentm.core.abi.cancel import CancelSignal
from agentm.core.abi.codec import CodecRegistry
from agentm.core.abi.context import ContextPolicy
from agentm.core.abi.provider import ProviderSessionIdentity
from agentm.core.abi.services import ServiceRegistry
from agentm.core.abi.session_api import ExtensionSpec, SessionContext
from agentm.core.abi.store import TrajectoryStore
from agentm.core.abi.stream import Model, StreamFn, ThinkingLevel
from agentm.core.abi.tool import Tool
from agentm.core.abi.tree import SessionGraphProtocol
from agentm.core.abi.trigger import TriggerRenderer
from agentm.core.runtime.trajectory import Trajectory


@dataclass(slots=True)
class SessionRuntimeConfig:
    """Low-level runtime dependencies after factory composition is resolved."""

    ctx: SessionContext | None = None
    session_id: str | None = None
    trajectory: Trajectory | None = None
    bus: EventBus | None = None
    store: TrajectoryStore | None = None
    graph: SessionGraphProtocol | None = None
    stream_fn: StreamFn | None = None
    model: Model | None = None
    tools: list[Tool] = field(default_factory=list)
    system: str | None = None
    context_policies: list[ContextPolicy] = field(default_factory=list)
    trigger_renderers: dict[str, TriggerRenderer] = field(default_factory=dict)
    codec: CodecRegistry | None = None
    max_turns: int | None = None
    max_tool_calls: int | None = None
    tool_allowlist: Sequence[str] | None = None
    thinking: ThinkingLevel = "off"
    cancel_signal: CancelSignal | None = None
    provider_identity: ProviderSessionIdentity | None = None
    services: ServiceRegistry | None = None
    cwd: str = ""
    purpose: str = "root"

    # Capability boundaries (resource ports, tool execution, permission,
    # effect scope, catalogs, provider resolver) are NOT fields here: they are
    # bound into ``services`` by the factory before construction. The registry
    # is the single representation of a session's boundaries.


@dataclass(frozen=True, slots=True)
class CompositionSnapshot:
    """Typed view of a session's rebuildable composition.

    The formal surface for spawn/fork/child construction — factories consume
    this instead of reading session privates.
    """

    extensions: tuple[ExtensionSpec, ...]
    external_tools: tuple[Tool, ...]
    external_context_policies: tuple[ContextPolicy, ...]
    external_trigger_renderers: dict[str, TriggerRenderer]
    codec: CodecRegistry
    stream_fn: StreamFn | None
    model: Model | None
    system: str | None
    max_turns: int | None
    max_tool_calls: int | None
    tool_allowlist: tuple[str, ...] | None
    thinking: ThinkingLevel
    lineage_cancel: CancelSignal


__all__ = ["CompositionSnapshot", "SessionRuntimeConfig"]
