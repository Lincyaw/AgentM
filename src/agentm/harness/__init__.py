"""AgentM v2 harness public surface."""

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
    "events",
    "GitBackedResourceWriter",
    "load_extension",
    "make_default_session_store",
    "resolve_session_state",
    "WriteResult",
]
