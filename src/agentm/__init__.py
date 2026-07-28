"""AgentM SDK public package surface."""

from __future__ import annotations

from agentm.core.abi.cancel import (
    CancelReason,
    CancelSignal,
    CancelSource,
    CompositeCancelSignal,
)
from agentm.core.abi.catalog import (
    ActiveSetFingerprint,
    AtomActivation,
    AtomCatalog,
    CatalogActiveSetInput,
)
from agentm.core.abi.compaction import (
    CompactionPublisher,
    CompactionRequest,
    CompactionResult,
    CompactionSourceAnchor,
    SessionCompactor,
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
from agentm.core.abi.messages import InterruptionMessagePolicy, JsonValue
from agentm.core.abi.operations import (
    BashOperations,
    EnvironmentOperations,
    EnvironmentRef,
    ExecResult,
)
from agentm.core.abi.provider import (
    ProviderConfig,
    ProviderRegistry,
    ProviderResolver,
    ProviderSessionIdentity,
)
from agentm.core.abi.query import (
    ObservabilityQueryStore,
    TraceQueryStore,
    TrajectoryQueryStore,
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
from agentm.core.abi.session_api import (
    ChildCancellationMode,
    ExtensionInput,
    ExtensionSource,
    ExtensionSpec,
    ScenarioLoader,
    ScenarioSpec,
    normalize_extension_spec,
)
from agentm.core.abi.session_config import (
    AgentSessionConfig,
    LoopConfig,
    ResolvedSessionSpec,
    SessionSpecResolver,
)
from agentm.core.abi.store import (
    TrajectoryCommit,
    TrajectoryDiagnostic,
    TrajectoryNodeQuery,
    TrajectoryNodeSort,
    TrajectoryStore,
)
from agentm.core.abi.stream import Model, StreamFn
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
from agentm.scenarios import (
    builtin_scenario_loader,
    load_scenario_manifest,
    packaged_scenario_names,
)
from agentm.sdk import AgentSession

__all__ = [
    "ActiveSetFingerprint",
    "AgentSession",
    "AgentSessionConfig",
    "AtomActivation",
    "AtomCatalog",
    "BackgroundCompletion",
    "BashOperations",
    "CancelReason",
    "CancelSignal",
    "CancelSource",
    "CatalogActiveSetInput",
    "ChildCancellationMode",
    "CompactionPublisher",
    "CompactionRequest",
    "CompactionResult",
    "CompactionSourceAnchor",
    "CompositeCancelSignal",
    "ContinueTrigger",
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
    "ExecResult",
    "ExtensionInput",
    "ExtensionSource",
    "ExtensionSpec",
    "Injection",
    "InterruptionMessagePolicy",
    "JsonValue",
    "LoopConfig",
    "Model",
    "MonitorFire",
    "ObservabilityQueryStore",
    "ProviderConfig",
    "ProviderRegistry",
    "ProviderRequestFailed",
    "ProviderResolver",
    "ProviderSessionIdentity",
    "ResolvedSessionSpec",
    "ResourceMutation",
    "ResourceReader",
    "ResourceRecoveryContext",
    "ResourceRef",
    "ResourceStore",
    "ResourceTransactionRef",
    "ResourceTxn",
    "ResourceTxnContext",
    "ResourceWriter",
    "ScenarioLoader",
    "ScenarioSpec",
    "Session",
    "SessionCompactor",
    "SessionSpecResolver",
    "StreamFn",
    "SubagentResult",
    "ToolExecutionRequirements",
    "ToolExecutor",
    "TraceQueryStore",
    "TrajectoryCommit",
    "TrajectoryDiagnostic",
    "TrajectoryNodeQuery",
    "TrajectoryNodeSort",
    "TrajectoryQueryStore",
    "TrajectoryStore",
    "TransactionalResourceWriter",
    "Trigger",
    "TriggerMetadata",
    "TriggerPriority",
    "UserInput",
    "WriteResult",
    "builtin_scenario_loader",
    "load_scenario_manifest",
    "normalize_extension_spec",
    "packaged_scenario_names",
]
