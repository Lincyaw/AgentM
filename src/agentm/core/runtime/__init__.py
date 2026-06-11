"""Runtime substrate — sessions, extensions, catalog, resources.

Holds the stateful side of the SDK: `AgentSession`, the extension
loader, catalog/freeze machinery, default session/resource backends.
Modules here may touch the filesystem at runtime but perform no side
effects at module import time (`agentm.core` stays Jupyter-clean to
import). Atoms reach this layer only through `agentm.core.abi.*`
Protocols and `api.*` hooks; direct atom imports of
`agentm.core.runtime.*` are validator-rejected.
"""

from __future__ import annotations

from agentm.core.abi import FunctionTool
from agentm.core.abi import events
import agentm.core.observability.event_otel as _event_otel  # noqa: F401 — populates OTel registry
from agentm.core.abi.events import (
    ApiRegisterEvent,
    ApiSendUserMessageEvent,
    ChildSessionEndEvent,
    ChildSessionStartEvent,
    CostBudgetExceededEvent,
    ExtensionInstallEvent,
    ExtensionReloadEvent,
)
from agentm.core.runtime.extension import (
    CommandSpec,
    ExtensionAPI,
    ExtensionLoadError,
    ProviderConfig,
    ReadonlySession,
    UnknownCommandError,
    load_extension,
)
from agentm.core.runtime.resource_loader import (
    DefaultResourceLoader,
    InMemoryResourceLoader,
    ResourceLoader,
)
from agentm.core.runtime.resource_writer import (
    GitBackedResourceWriter,
    ResourceWriter,
    WriteResult,
)
from agentm.core.runtime.session_cwd import MissingSessionCwdError
from agentm.core.runtime.session import AgentSession, AgentSessionConfig
from agentm.core.runtime.session_manager import (
    InMemorySessionManager,
    JsonlSessionManager,
    JsonlSessionStore,
    SessionContext,
    SessionEntry,
    SessionHeader,
    SessionManager,
    SessionTreeNode,
)
from agentm.core.runtime.session_bootstrap import (
    make_default_session_store,
    resolve_session_state,
)
from agentm.core.runtime.session_factory import create_agent_session
from agentm.core.runtime.session_inbox import SessionInbox

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
    "create_agent_session",
    "events",
    "GitBackedResourceWriter",
    "load_extension",
    "make_default_session_store",
    "resolve_session_state",
    "SessionInbox",
    "WriteResult",
]
