"""AgentM SDK public package surface."""

from __future__ import annotations

from agentm.core.abi.catalog import (
    ActiveSetFingerprint,
    AtomActivation,
    AtomCatalog,
    CatalogActiveSetInput,
)
from agentm.core.abi.lifecycle import (
    EffectScope,
    EffectTxn,
    EnvironmentRestorePolicy,
    EnvironmentRestoreStatus,
)
from agentm.core.abi.provider import (
    ProviderConfig,
    ProviderPromptCacheAdapter,
    ProviderPromptCacheRequest,
    ProviderPromptCacheResult,
    ProviderRegistry,
    ProviderResolver,
    ProviderSessionIdentity,
)
from agentm.core.abi.resource import (
    ResourceReader,
    ResourceMutation,
    ResourceRef,
    ResourceTxn,
    ResourceWriter,
    TransactionalResourceWriter,
    WriteResult,
)
from agentm.core.abi.session_config import (
    AgentSessionConfig,
    LoopConfig,
    ResolvedSessionSpec,
    SessionSpecResolver,
)
from agentm.core.abi.session_api import ExtensionSpec, ScenarioLoader, ScenarioSpec
from agentm.core.abi.stream import Model, StreamFn
from agentm.core.abi.store import TrajectoryStore
from agentm.core.abi.tool_executor import ToolExecutionRequirements, ToolExecutor
from agentm.core.abi.trajectory import SessionConfigChange
from agentm.core.runtime.session import Session
from agentm.scenarios import builtin_scenario_loader, packaged_scenario_names

AgentSession = Session

__all__ = [
    "ActiveSetFingerprint",
    "AgentSession",
    "AgentSessionConfig",
    "AtomActivation",
    "AtomCatalog",
    "CatalogActiveSetInput",
    "EffectScope",
    "EffectTxn",
    "EnvironmentRestorePolicy",
    "EnvironmentRestoreStatus",
    "ExtensionSpec",
    "LoopConfig",
    "Model",
    "ProviderConfig",
    "ProviderPromptCacheAdapter",
    "ProviderPromptCacheRequest",
    "ProviderPromptCacheResult",
    "ProviderRegistry",
    "ProviderResolver",
    "ProviderSessionIdentity",
    "ResolvedSessionSpec",
    "ResourceWriter",
    "ResourceReader",
    "ResourceMutation",
    "ResourceRef",
    "ResourceTxn",
    "ScenarioLoader",
    "ScenarioSpec",
    "Session",
    "SessionConfigChange",
    "SessionSpecResolver",
    "StreamFn",
    "ToolExecutionRequirements",
    "ToolExecutor",
    "TrajectoryStore",
    "TransactionalResourceWriter",
    "WriteResult",
    "builtin_scenario_loader",
    "packaged_scenario_names",
]
