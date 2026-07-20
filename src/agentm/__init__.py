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
    EnvironmentForkLease,
    EnvironmentRestoreError,
    EnvironmentRestoreFailureHandler,
    EnvironmentRestoreState,
    EnvironmentRestoreStatus,
    EnvironmentSnapshot,
    EnvironmentSnapshotter,
)
from agentm.core.abi.messages import InterruptionMessagePolicy
from agentm.core.abi.messages import JsonValue
from agentm.core.abi.operations import (
    BashOperations,
    EnvironmentOperations,
    EnvironmentRef,
    ExecResult,
)
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
    ResourceTransactionRef,
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
    ExtensionInput,
    ExtensionSource,
    ExtensionSpec,
    ScenarioLoader,
    ScenarioSpec,
    normalize_extension_spec,
)
from agentm.core.abi.stream import Model, StreamFn
from agentm.core.abi.store import (
    TrajectoryCommit,
    TrajectoryNodeQuery,
    TrajectoryNodeSort,
    TrajectoryStore,
)
from agentm.core.abi.termination import ProviderRequestFailed
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
from agentm.sdk import AgentSession
from agentm.scenarios import (
    builtin_scenario_loader,
    load_scenario_manifest,
    packaged_scenario_names,
)

__all__ = [
    "ActiveSetFingerprint",
    "AgentSession",
    "AgentSessionConfig",
    "AtomActivation",
    "AtomCatalog",
    "BashOperations",
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
    "EnvironmentForkLease",
    "EnvironmentForkableResourceWriter",
    "EnvironmentOperations",
    "EnvironmentRef",
    "EnvironmentRestoreError",
    "EnvironmentRestoreFailureHandler",
    "EnvironmentRestoreState",
    "EnvironmentRestoreStatus",
    "EnvironmentSnapshot",
    "EnvironmentSnapshotter",
    "ExtensionInput",
    "ExtensionSource",
    "ExtensionSpec",
    "ExecResult",
    "InterruptionMessagePolicy",
    "JsonValue",
    "LoopConfig",
    "Model",
    "ObservabilityQueryStore",
    "ProviderConfig",
    "ProviderPromptCacheAdapter",
    "ProviderPromptCacheRequest",
    "ProviderPromptCacheResult",
    "ProviderRequestFailed",
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
    "ResourceTransactionRef",
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
    "TrajectoryCommit",
    "TrajectoryNodeQuery",
    "TrajectoryNodeSort",
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
    "load_scenario_manifest",
    "normalize_extension_spec",
    "packaged_scenario_names",
]
