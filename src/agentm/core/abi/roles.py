"""Cross-atom service keys used by the minimal SDK."""

from __future__ import annotations

from typing import Final

LOOP_BUDGET_SERVICE: Final = "loop_budget"
"""Service key for a session loop budget config."""

RETRY_POLICY_SERVICE: Final = "retry_policy"
"""Service key for the provider retry policy callable."""

PROVIDER_RESOLVER_SERVICE: Final = "provider_resolver"
"""Service key for selecting the active provider registration."""

RESOURCE_WRITER_SERVICE: Final = "resource_writer"
"""Service key for the host-provided resource mutation port."""

RESOURCE_TXN_SERVICE: Final = "resource_txn"
"""Service key for the active turn-scoped resource transaction."""

OPERATIONS_SERVICE: Final = "operations"
"""Service key for the active Operations bundle."""

ENVIRONMENT_OPERATIONS_SERVICE: Final = "operations:environment"
"""Service key for the active environment operations backend."""

BASH_OPERATIONS_SERVICE: Final = "operations:bash"
"""Service key for shell execution operations."""

TOOL_EXECUTOR_SERVICE: Final = "tool_executor"
"""Service key for the host-provided tool execution boundary."""

TOOL_ORCHESTRATOR_SERVICE: Final = "tool_orchestrator"
"""Service key for the host-provided batch tool orchestration boundary."""

PERMISSION_POLICY_SERVICE: Final = "permission_policy"
"""Service key for the host-provided permission decision boundary."""

TRAJECTORY_NODE_STORE_SERVICE: Final = "trajectory_node_store"
"""Service key for message-level trajectory node persistence/query."""

TRAJECTORY_QUERY_STORE_SERVICE: Final = "trajectory_query_store"
"""Service key for session/turn/span/event trajectory query."""

SESSION_SPEC_RESOLVER_SERVICE: Final = "session_spec_resolver"
"""Service key for host-owned session config resolution."""

RESOLVED_SESSION_SPEC_SERVICE: Final = "resolved_session_spec"
"""Service key for the resolved composition/config used by this session."""

CONTEXT_PROJECTION_SERVICE: Final = "context_projection"
"""Service key for host/session context projection policy."""

EFFECT_SCOPE_SERVICE: Final = "effect_scope"
"""Service key for the host-provided world-effect lifecycle port."""

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
    "CONTEXT_PROJECTION_SERVICE",
    "EFFECT_SCOPE_SERVICE",
    "ENVIRONMENT_OPERATIONS_SERVICE",
    "LOOP_BUDGET_SERVICE",
    "OPERATIONS_SERVICE",
    "PERMISSION_POLICY_SERVICE",
    "PROVIDER_RESOLVER_SERVICE",
    "RETRY_POLICY_SERVICE",
    "RESOURCE_WRITER_SERVICE",
    "RESOURCE_TXN_SERVICE",
    "RESOLVED_SESSION_SPEC_SERVICE",
    "SCENARIO_LOADER_SERVICE",
    "SESSION_SPEC_RESOLVER_SERVICE",
    "TOOL_EXECUTOR_SERVICE",
    "TOOL_ORCHESTRATOR_SERVICE",
    "TRAJECTORY_NODE_STORE_SERVICE",
    "TRAJECTORY_QUERY_STORE_SERVICE",
    "VERSIONED_RESOURCE_STORE_SERVICE",
]
