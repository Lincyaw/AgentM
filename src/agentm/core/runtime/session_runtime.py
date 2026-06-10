"""Session-scoped runtime bundle passed into ``AgentSession.__init__``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agentm.core.abi import AgentLoop, EventBus, Tool
from agentm.core.lib.ref import Ref
from agentm.core.runtime.atom_reloader import AtomReloader
from agentm.core.runtime.extension import (
    CommandSpec,
    ProviderConfig,
    Renderer,
    _ExtensionAPIImpl,
)
from agentm.core.runtime.resource_loader import ResourceLoader
from agentm.core.runtime.session_inbox import SessionInbox
from agentm.core.runtime.session_manager import SessionManager


@dataclass(slots=True)
class SessionRuntime:
    bus: EventBus
    session_manager: SessionManager
    resource_loader: ResourceLoader
    loop: AgentLoop
    active_provider_ref: Ref[ProviderConfig | None]
    tools: list[Tool]
    commands: dict[str, CommandSpec]
    providers: dict[str, ProviderConfig]
    renderers: dict[str, Renderer]
    apis: dict[str, _ExtensionAPIImpl]
    services: dict[str, Any]
    reloader: AtomReloader
    inbox: SessionInbox
