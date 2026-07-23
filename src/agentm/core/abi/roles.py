"""Cross-atom service keys and role descriptors used by the minimal SDK.

String constants name every well-known service. Cross-layer boundaries
additionally get a ``ServiceRole`` descriptor — the single source for the
boundary's key, Protocol, and canonical scope — consumed through
``ServiceRegistry.bind`` / ``get_role`` / ``require_role``. Atom-local or
peer-to-peer services may keep using plain ``register``/``get`` with the
string key.
"""

from __future__ import annotations

from typing import Final

from agentm.core.abi.catalog import (
    ActiveSetFingerprint,
    AtomCatalog,
    AtomCatalogQuery,
    VersionedResourceStore,
)
from agentm.core.abi.compaction import (
    CompactionPublisher,
    ContextCompactionService,
    SessionCompactor,
)
from agentm.core.abi.lifecycle import (
    EffectScope,
    EnvironmentRestoreFailureHandler,
    EnvironmentRestoreStatus,
)
from agentm.core.abi.operations import BashOperations, EnvironmentOperations
from agentm.core.abi.permission import PermissionPolicy
from agentm.core.abi.provider import ProviderResolver, ProviderSessionIdentity
from agentm.core.abi.resource import (
    ResourceReader,
    ResourceStore,
    ResourceTxn,
    ResourceWriter,
)
from agentm.core.abi.services import ServiceRegistry, ServiceRole
from agentm.core.abi.store import TrajectoryStore
from agentm.core.abi.telemetry import SessionTelemetry
from agentm.core.abi.tool_executor import ToolExecutor
from agentm.core.abi.tool_orchestration import ToolOrchestrator

LOOP_BUDGET_SERVICE: Final = "loop_budget"
"""Service key for a session loop budget config."""

RETRY_POLICY_SERVICE: Final = "retry_policy"
"""Service key for the provider retry policy callable."""

PROVIDER_RESOLVER_SERVICE: Final = "provider_resolver"
"""Service key for selecting the active provider registration."""

PROVIDER_SESSION_IDENTITY_SERVICE: Final = "provider_session_identity"
"""Service key for the provider/model identity bound to a session history."""

PROVIDER_PROMPT_CACHE_ADAPTER_SERVICE: Final = "provider_prompt_cache_adapter"
"""Service key for provider-specific prompt-cache materialization."""

INTERRUPTION_MESSAGE_POLICY_SERVICE: Final = "interruption_message_policy"
"""Service key for provider-facing interrupted-turn message construction."""

RESOURCE_WRITER_SERVICE: Final = "resource_writer"
"""Service key for the host-provided resource mutation port."""

RESOURCE_READER_SERVICE: Final = "resource_reader"
"""Service key for backend-neutral ResourceRef reads."""

RESOURCE_STORE_SERVICE: Final = "resource_store"
"""Service key for durable logical ResourceRef reads and mutations."""

RESOURCE_TXN_SERVICE: Final = "resource_txn"
"""Service key for the active turn-scoped resource transaction."""

ENVIRONMENT_OPERATIONS_SERVICE: Final = "operations:environment"
"""Service key for the active environment operations backend."""

BASH_OPERATIONS_SERVICE: Final = "operations:bash"
"""Service key for shell execution operations."""

HOST_BASH_OPERATIONS_SERVICE: Final = "operations:bash:host"
"""Service key for host-local shell execution, independent of session environment."""

TOOL_EXECUTOR_SERVICE: Final = "tool_executor"
"""Service key for the host-provided tool execution boundary."""

TOOL_ORCHESTRATOR_SERVICE: Final = "tool_orchestrator"
"""Service key for the host-provided batch tool orchestration boundary."""

PERMISSION_POLICY_SERVICE: Final = "permission_policy"
"""Service key for the host-provided permission decision boundary."""

TRAJECTORY_STORE_SERVICE: Final = "trajectory_store"
"""Service key for the selected trajectory persistence/query backend."""

TRAJECTORY_QUERY_STORE_SERVICE: Final = "trajectory_query_store"
"""Service key for session/turn trajectory query."""

CATALOG_QUERY_SERVICE: Final = "catalog_query"
"""Service key for indexed catalog active-set query."""

RESOLVED_SESSION_SPEC_SERVICE: Final = "resolved_session_spec"
"""Service key for the resolved composition/config used by this session."""

CONTEXT_PROJECTION_SERVICE: Final = "context_projection"
"""Service key for host/session context projection policy."""

CONTEXT_COMPACTION_SERVICE: Final = "context_compaction"
"""Service key for step-boundary context compaction requests."""

SESSION_COMPACTOR_SERVICE: Final = "session_compactor"
"""Service key for store-driven compaction artifact generation."""

COMPACTION_PUBLISHER_SERVICE: Final = "compaction_publisher"
"""Service key for publishing compaction artifacts to context projection."""

EFFECT_SCOPE_SERVICE: Final = "effect_scope"
"""Service key for the host-provided world-effect lifecycle port."""

ENVIRONMENT_RESTORE_FAILURE_HANDLER_SERVICE: Final = (
    "environment_restore_failure_handler"
)
"""Service key for host-enforced degraded read-only restore handling."""

ENVIRONMENT_RESTORE_STATUS_SERVICE: Final = "environment_restore_status"
"""Service key for the last resume-time environment restore status."""

VERSIONED_RESOURCE_STORE_SERVICE: Final = "versioned_resource_store"
"""Service key for versioned SDK resources such as atom identity payloads."""

ATOM_CATALOG_SERVICE: Final = "atom_catalog"
"""Service key for resolved atom composition identity."""

ACTIVE_SET_FINGERPRINT_SERVICE: Final = "active_set_fingerprint"
"""Service key for the active atom-set fingerprint for this session."""

SCENARIO_LOADER_SERVICE: Final = "scenario_loader"
"""Service key for a host-provided scenario resolver."""

EXPERIMENT_SERVICE: Final = "experiment"
"""Service key for the frozen experiment config mapping."""

TOOL_ALLOWLIST_SERVICE: Final = "tool_allowlist"
"""Service key for the session tool allowlist."""

SESSION_TELEMETRY_SERVICE: Final = "session_telemetry"
"""Service key for the session telemetry sink."""


# --- Cross-layer boundary roles (key + Protocol + canonical scope) ----------

RESOURCE_WRITER: Final[ServiceRole[ResourceWriter]] = ServiceRole(
    RESOURCE_WRITER_SERVICE, ResourceWriter, "tree"
)
RESOURCE_READER: Final[ServiceRole[ResourceReader]] = ServiceRole(
    RESOURCE_READER_SERVICE, ResourceReader, "tree"
)
RESOURCE_STORE: Final[ServiceRole[ResourceStore]] = ServiceRole(
    RESOURCE_STORE_SERVICE, ResourceStore, "tree"
)
RESOURCE_TXN: Final[ServiceRole[ResourceTxn]] = ServiceRole(
    RESOURCE_TXN_SERVICE, ResourceTxn, "session"
)
TOOL_EXECUTOR: Final[ServiceRole[ToolExecutor]] = ServiceRole(
    TOOL_EXECUTOR_SERVICE, ToolExecutor, "tree"
)
TOOL_ORCHESTRATOR: Final[ServiceRole[ToolOrchestrator]] = ServiceRole(
    TOOL_ORCHESTRATOR_SERVICE, ToolOrchestrator, "tree"
)
PERMISSION_POLICY_ROLE: Final[ServiceRole[PermissionPolicy]] = ServiceRole(
    PERMISSION_POLICY_SERVICE, PermissionPolicy, "tree"
)
EFFECT_SCOPE_ROLE: Final[ServiceRole[EffectScope]] = ServiceRole(
    EFFECT_SCOPE_SERVICE, EffectScope, "tree"
)
VERSIONED_RESOURCE_STORE_ROLE: Final[ServiceRole[VersionedResourceStore]] = ServiceRole(
    VERSIONED_RESOURCE_STORE_SERVICE, VersionedResourceStore, "tree"
)
ATOM_CATALOG_ROLE: Final[ServiceRole[AtomCatalog]] = ServiceRole(
    ATOM_CATALOG_SERVICE, AtomCatalog, "tree"
)
CATALOG_QUERY: Final[ServiceRole[AtomCatalogQuery]] = ServiceRole(
    CATALOG_QUERY_SERVICE, AtomCatalogQuery, "tree"
)
ENVIRONMENT_OPERATIONS: Final[ServiceRole[EnvironmentOperations]] = ServiceRole(
    ENVIRONMENT_OPERATIONS_SERVICE, EnvironmentOperations, "session"
)
BASH_OPERATIONS_ROLE: Final[ServiceRole[BashOperations]] = ServiceRole(
    BASH_OPERATIONS_SERVICE, BashOperations, "session"
)
HOST_BASH_OPERATIONS: Final[ServiceRole[BashOperations]] = ServiceRole(
    HOST_BASH_OPERATIONS_SERVICE, BashOperations, "tree"
)
TRAJECTORY_STORE_ROLE: Final[ServiceRole[TrajectoryStore]] = ServiceRole(
    TRAJECTORY_STORE_SERVICE, TrajectoryStore, "tree"
)
TRAJECTORY_QUERY_STORE: Final[ServiceRole[object]] = ServiceRole(
    TRAJECTORY_QUERY_STORE_SERVICE, None, "tree"
)
SESSION_COMPACTOR: Final[ServiceRole[SessionCompactor]] = ServiceRole(
    SESSION_COMPACTOR_SERVICE, SessionCompactor, "tree"
)
COMPACTION_PUBLISHER_ROLE: Final[ServiceRole[CompactionPublisher]] = ServiceRole(
    COMPACTION_PUBLISHER_SERVICE, CompactionPublisher, "tree"
)
CONTEXT_COMPACTION: Final[ServiceRole[ContextCompactionService]] = ServiceRole(
    CONTEXT_COMPACTION_SERVICE, ContextCompactionService, "session"
)
ENVIRONMENT_RESTORE_FAILURE_HANDLER: Final[
    ServiceRole[EnvironmentRestoreFailureHandler]
] = ServiceRole(
    ENVIRONMENT_RESTORE_FAILURE_HANDLER_SERVICE,
    EnvironmentRestoreFailureHandler,
    "tree",
)
ENVIRONMENT_RESTORE_STATUS_ROLE: Final[ServiceRole[EnvironmentRestoreStatus]] = (
    ServiceRole(
        ENVIRONMENT_RESTORE_STATUS_SERVICE,
        EnvironmentRestoreStatus,
        "session",
    )
)
PROVIDER_RESOLVER_ROLE: Final[ServiceRole[ProviderResolver]] = ServiceRole(
    PROVIDER_RESOLVER_SERVICE, ProviderResolver, "tree"
)
PROVIDER_SESSION_IDENTITY: Final[ServiceRole[ProviderSessionIdentity]] = ServiceRole(
    PROVIDER_SESSION_IDENTITY_SERVICE, ProviderSessionIdentity, "session"
)
ACTIVE_SET_FINGERPRINT_ROLE: Final[ServiceRole[ActiveSetFingerprint]] = ServiceRole(
    ACTIVE_SET_FINGERPRINT_SERVICE, ActiveSetFingerprint, "session"
)
SCENARIO_LOADER_ROLE: Final[ServiceRole[object]] = ServiceRole(
    SCENARIO_LOADER_SERVICE, None, "tree"
)
SESSION_TELEMETRY_ROLE: Final[ServiceRole[SessionTelemetry]] = ServiceRole(
    SESSION_TELEMETRY_SERVICE, SessionTelemetry, "session"
)


def bind_resource_store(
    services: ServiceRegistry,
    store: ResourceStore,
    *,
    replace: bool = False,
) -> None:
    """Bind a ResourceStore and keep the read side consistent.

    A durable store is also a reader: when no reader is bound (or the bound
    reader is the store being replaced), the store becomes the reader too.
    """

    previous = services.get_role(RESOURCE_STORE)
    services.bind(RESOURCE_STORE, store, replace=replace)
    reader = services.get(RESOURCE_READER_SERVICE)
    if reader is None:
        services.bind(RESOURCE_READER, store)
    elif replace and reader is previous:
        services.bind(RESOURCE_READER, store, replace=True)


def bind_atom_catalog(
    services: ServiceRegistry,
    catalog: AtomCatalog,
    *,
    replace: bool = False,
) -> None:
    """Bind an AtomCatalog and expose its query surface when present."""

    services.bind(ATOM_CATALOG_ROLE, catalog, replace=replace)
    if isinstance(catalog, AtomCatalogQuery):  # code-health: ignore[AM025]
        services.bind(CATALOG_QUERY, catalog, replace=replace)


__all__ = [
    "ACTIVE_SET_FINGERPRINT_ROLE",
    "ACTIVE_SET_FINGERPRINT_SERVICE",
    "ATOM_CATALOG_ROLE",
    "ATOM_CATALOG_SERVICE",
    "BASH_OPERATIONS_ROLE",
    "BASH_OPERATIONS_SERVICE",
    "CATALOG_QUERY",
    "CATALOG_QUERY_SERVICE",
    "COMPACTION_PUBLISHER_ROLE",
    "COMPACTION_PUBLISHER_SERVICE",
    "CONTEXT_COMPACTION",
    "CONTEXT_COMPACTION_SERVICE",
    "CONTEXT_PROJECTION_SERVICE",
    "EFFECT_SCOPE_ROLE",
    "EFFECT_SCOPE_SERVICE",
    "ENVIRONMENT_OPERATIONS",
    "ENVIRONMENT_OPERATIONS_SERVICE",
    "ENVIRONMENT_RESTORE_FAILURE_HANDLER",
    "ENVIRONMENT_RESTORE_FAILURE_HANDLER_SERVICE",
    "ENVIRONMENT_RESTORE_STATUS_ROLE",
    "ENVIRONMENT_RESTORE_STATUS_SERVICE",
    "EXPERIMENT_SERVICE",
    "HOST_BASH_OPERATIONS",
    "HOST_BASH_OPERATIONS_SERVICE",
    "INTERRUPTION_MESSAGE_POLICY_SERVICE",
    "LOOP_BUDGET_SERVICE",
    "PERMISSION_POLICY_ROLE",
    "PERMISSION_POLICY_SERVICE",
    "PROVIDER_RESOLVER_ROLE",
    "PROVIDER_RESOLVER_SERVICE",
    "PROVIDER_PROMPT_CACHE_ADAPTER_SERVICE",
    "PROVIDER_SESSION_IDENTITY",
    "PROVIDER_SESSION_IDENTITY_SERVICE",
    "RETRY_POLICY_SERVICE",
    "RESOURCE_READER",
    "RESOURCE_READER_SERVICE",
    "RESOURCE_STORE",
    "RESOURCE_STORE_SERVICE",
    "RESOURCE_TXN",
    "RESOURCE_TXN_SERVICE",
    "RESOURCE_WRITER",
    "RESOURCE_WRITER_SERVICE",
    "RESOLVED_SESSION_SPEC_SERVICE",
    "SCENARIO_LOADER_ROLE",
    "SCENARIO_LOADER_SERVICE",
    "SESSION_COMPACTOR",
    "SESSION_COMPACTOR_SERVICE",
    "SESSION_TELEMETRY_ROLE",
    "SESSION_TELEMETRY_SERVICE",
    "TOOL_ALLOWLIST_SERVICE",
    "TOOL_EXECUTOR",
    "TOOL_EXECUTOR_SERVICE",
    "TOOL_ORCHESTRATOR",
    "TOOL_ORCHESTRATOR_SERVICE",
    "TRAJECTORY_STORE_ROLE",
    "TRAJECTORY_STORE_SERVICE",
    "TRAJECTORY_QUERY_STORE",
    "TRAJECTORY_QUERY_STORE_SERVICE",
    "VERSIONED_RESOURCE_STORE_ROLE",
    "VERSIONED_RESOURCE_STORE_SERVICE",
    "bind_atom_catalog",
    "bind_resource_store",
]
