"""Cross-atom service keys used by the minimal SDK."""

from __future__ import annotations

from typing import Final

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

__all__ = [
    "ACTIVE_SET_FINGERPRINT_SERVICE",
    "ATOM_CATALOG_SERVICE",
    "BASH_OPERATIONS_SERVICE",
    "HOST_BASH_OPERATIONS_SERVICE",
    "CATALOG_QUERY_SERVICE",
    "COMPACTION_PUBLISHER_SERVICE",
    "CONTEXT_COMPACTION_SERVICE",
    "CONTEXT_PROJECTION_SERVICE",
    "EFFECT_SCOPE_SERVICE",
    "ENVIRONMENT_RESTORE_FAILURE_HANDLER_SERVICE",
    "ENVIRONMENT_RESTORE_STATUS_SERVICE",
    "ENVIRONMENT_OPERATIONS_SERVICE",
    "INTERRUPTION_MESSAGE_POLICY_SERVICE",
    "LOOP_BUDGET_SERVICE",
    "PERMISSION_POLICY_SERVICE",
    "PROVIDER_RESOLVER_SERVICE",
    "PROVIDER_PROMPT_CACHE_ADAPTER_SERVICE",
    "PROVIDER_SESSION_IDENTITY_SERVICE",
    "RETRY_POLICY_SERVICE",
    "RESOURCE_READER_SERVICE",
    "RESOURCE_STORE_SERVICE",
    "RESOURCE_WRITER_SERVICE",
    "RESOURCE_TXN_SERVICE",
    "RESOLVED_SESSION_SPEC_SERVICE",
    "SCENARIO_LOADER_SERVICE",
    "SESSION_COMPACTOR_SERVICE",
    "TOOL_EXECUTOR_SERVICE",
    "TOOL_ORCHESTRATOR_SERVICE",
    "TRAJECTORY_STORE_SERVICE",
    "TRAJECTORY_QUERY_STORE_SERVICE",
    "VERSIONED_RESOURCE_STORE_SERVICE",
]
