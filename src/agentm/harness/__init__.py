"""AgentM v2 harness public surface.

Phase 2.5 removed the legacy runtime layer; the only supported entry
points are listed below.
"""

from __future__ import annotations

from agentm.core.abi import FunctionTool
from agentm.harness import events
from agentm.harness.events import (
    ApiRegisterEvent,
    ApiSendUserMessageEvent,
    ChildSessionEndEvent,
    ChildSessionStartEvent,
    CostBudgetExceededEvent,
    ExtensionInstallEvent,
    ExtensionReloadEvent,
)
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
from agentm.harness.resource_writer import (
    GitBackedResourceWriter,
    ResourceWriter,
    WriteResult,
)
from agentm.harness.session_cwd import MissingSessionCwdError
from agentm.harness.session import AgentSession, AgentSessionConfig
from agentm.harness.session_manager import (
    InMemorySessionManager,
    JsonlSessionManager,
    JsonlSessionStore,
    SessionContext,
    SessionEntry,
    SessionHeader,
    SessionManager,
    SessionTreeNode,
)
from agentm.harness.session_bootstrap import (
    make_default_session_store,
    resolve_session_state,
)
from agentm.harness.session_runtime import AgentSessionRuntime
from agentm.harness.session_services import (
    AgentSessionRuntimeDiagnostic,
    AgentSessionServices,
    CreateAgentSessionFromServicesOptions,
    CreateAgentSessionServicesOptions,
    create_agent_session_from_services,
    create_agent_session_services,
)

__all__ = [
    "AgentSession",
    "ApiRegisterEvent",
    "ApiSendUserMessageEvent",
    "ChildSessionEndEvent",
    "ChildSessionStartEvent",
    "CostBudgetExceededEvent",
    "AgentSessionConfig",
    "CommandSpec",
    "DefaultResourceLoader",
    "ExtensionAPI",
    "ExtensionInstallEvent",
    "ExtensionLoadError",
    "ExtensionReloadEvent",
    "FunctionTool",
    "InMemoryResourceLoader",
    "InMemorySessionManager",
    "JsonlSessionManager",
    "JsonlSessionStore",
    "MissingSessionCwdError",
    "ProviderConfig",
    "ReadonlySession",
    "ResourceLoader",
    "ResourceWriter",
    "SessionContext",
    "SessionEntry",
    "SessionHeader",
    "SessionManager",
    "SessionTreeNode",
    "UnknownCommandError",
    "AgentSessionRuntime",
    "AgentSessionRuntimeDiagnostic",
    "AgentSessionServices",
    "CreateAgentSessionFromServicesOptions",
    "CreateAgentSessionServicesOptions",
    "create_agent_session_from_services",
    "create_agent_session_services",
    "events",
    "GitBackedResourceWriter",
    "load_extension",
    "make_default_session_store",
    "resolve_session_state",
    "WriteResult",
]
