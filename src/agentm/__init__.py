"""AgentM SDK public package surface."""

from __future__ import annotations

from agentm.core.abi.catalog import (
    ActiveSetFingerprint,
    AtomActivation,
    AtomCatalog,
)
from agentm.core.abi.lifecycle import (
    EffectScope,
    EffectTxn,
)
from agentm.core.abi.provider import ProviderConfig, ProviderRegistry, ProviderResolver
from agentm.core.abi.resource import (
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
from agentm.core.runtime.session import Session

__all__ = [
    "ActiveSetFingerprint",
    "AgentSessionConfig",
    "AtomActivation",
    "AtomCatalog",
    "EffectScope",
    "EffectTxn",
    "ExtensionSpec",
    "LoopConfig",
    "Model",
    "ProviderConfig",
    "ProviderRegistry",
    "ProviderResolver",
    "ResolvedSessionSpec",
    "ResourceWriter",
    "ResourceMutation",
    "ResourceRef",
    "ResourceTxn",
    "ScenarioLoader",
    "ScenarioSpec",
    "Session",
    "SessionSpecResolver",
    "StreamFn",
    "ToolExecutionRequirements",
    "ToolExecutor",
    "TrajectoryStore",
    "TransactionalResourceWriter",
    "WriteResult",
]
