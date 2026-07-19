"""AgentM SDK public package surface."""

from __future__ import annotations

from agentm.core.abi.catalog import (
    ActiveSetFingerprint,
    AtomActivation,
    AtomCatalog,
    CatalogActiveSetInput,
)
from agentm.core.abi.cancel import (
    CancelReason,
    CancelSignal,
    CancelSource,
    CompositeCancelSignal,
)
from agentm.core.abi.lifecycle import (
    EffectScope,
    EffectTxn,
    EnvironmentCheckpoint,
    EnvironmentFork,
    EnvironmentRestoreError,
    EnvironmentRestoreFailureHandler,
    EnvironmentRestoreState,
    EnvironmentRestoreStatus,
    EnvironmentSnapshot,
    EnvironmentSnapshotter,
)
from agentm.core.abi.messages import InterruptionMessagePolicy
from agentm.core.abi.messages import JsonValue
from agentm.core.abi.query import (
    ObservabilityQueryStore,
    TraceQueryStore,
    TrajectoryQueryStore,
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
    EnvironmentForkableResourceWriter,
    ResourceMutation,
    ResourceReader,
    ResourceRecoveryContext,
    ResourceRef,
    ResourceStore,
    ResourceTxn,
    ResourceTxnContext,
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
from agentm.core.abi.session_api import (
    ChildCancellationMode,
    ExtensionSpec,
    ScenarioLoader,
    ScenarioSpec,
)
from agentm.core.abi.stream import Model, StreamFn
from agentm.core.abi.store import (
    TrajectoryNodeQuery,
    TrajectoryNodeSort,
    TrajectoryNodeStore,
    TrajectoryStore,
)
from agentm.core.abi.tool_executor import ToolExecutionRequirements, ToolExecutor
from agentm.core.abi.trigger import (
    BackgroundCompletion,
    ContinueTrigger,
    Injection,
    MonitorFire,
    SubagentResult,
    Trigger,
    TriggerMetadata,
    TriggerPriority,
    UserInput,
)
from agentm.core.runtime.session import Session
from agentm.scenarios import builtin_scenario_loader, packaged_scenario_names

AgentSession = Session

__all__ = [
    "ActiveSetFingerprint",
    "AgentSession",
    "AgentSessionConfig",
    "AtomActivation",
    "AtomCatalog",
    "CancelReason",
    "CancelSignal",
    "CancelSource",
    "CatalogActiveSetInput",
    "ChildCancellationMode",
    "CompositeCancelSignal",
    "EffectScope",
    "EffectTxn",
    "EnvironmentCheckpoint",
    "EnvironmentFork",
    "EnvironmentForkableResourceWriter",
    "EnvironmentRestoreError",
    "EnvironmentRestoreFailureHandler",
    "EnvironmentRestoreState",
    "EnvironmentRestoreStatus",
    "EnvironmentSnapshot",
    "EnvironmentSnapshotter",
    "ExtensionSpec",
    "InterruptionMessagePolicy",
    "JsonValue",
    "LoopConfig",
    "Model",
    "ObservabilityQueryStore",
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
    "ResourceRecoveryContext",
    "ResourceRef",
    "ResourceStore",
    "ResourceTxn",
    "ResourceTxnContext",
    "ScenarioLoader",
    "ScenarioSpec",
    "Session",
    "SessionSpecResolver",
    "StreamFn",
    "ToolExecutionRequirements",
    "ToolExecutor",
    "TraceQueryStore",
    "TrajectoryNodeQuery",
    "TrajectoryNodeSort",
    "TrajectoryNodeStore",
    "TrajectoryQueryStore",
    "TrajectoryStore",
    "Trigger",
    "TriggerMetadata",
    "TriggerPriority",
    "UserInput",
    "ContinueTrigger",
    "Injection",
    "BackgroundCompletion",
    "MonitorFire",
    "SubagentResult",
    "TransactionalResourceWriter",
    "WriteResult",
    "builtin_scenario_loader",
    "packaged_scenario_names",
]
