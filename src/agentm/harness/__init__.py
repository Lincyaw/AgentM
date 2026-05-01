"""AgentM v2 harness public surface.

Phase 2.5 removed the legacy runtime layer; the only supported entry
points are listed below.
"""

from __future__ import annotations

from agentm.harness import events
from agentm.harness.extension import (
    CommandSpec,
    ExtensionAPI,
    ExtensionLoadError,
    ProviderConfig,
    ReadonlySession,
    UnknownCommandError,
    load_extension,
)
from agentm.harness.resource_loader import (
    DefaultResourceLoader,
    InMemoryResourceLoader,
    ResourceLoader,
)
from agentm.harness.session import AgentSession, AgentSessionConfig
from agentm.harness.session_manager import (
    InMemorySessionManager,
    JsonlSessionManager,
    SessionEntry,
    SessionManager,
)

__all__ = [
    "AgentSession",
    "AgentSessionConfig",
    "CommandSpec",
    "DefaultResourceLoader",
    "ExtensionAPI",
    "ExtensionLoadError",
    "InMemoryResourceLoader",
    "InMemorySessionManager",
    "JsonlSessionManager",
    "ProviderConfig",
    "ReadonlySession",
    "ResourceLoader",
    "SessionEntry",
    "SessionManager",
    "UnknownCommandError",
    "events",
    "load_extension",
]
