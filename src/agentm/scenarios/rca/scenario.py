"""RCA Scenario implementation using the Scenario protocol.

Encapsulates all RCA-specific wiring: hypothesis/profile stores, tools,
context formatting, answer schemas, output schema, and hooks.
"""
from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, Literal, Optional

from agentm.harness.tool import tool_from_function

if TYPE_CHECKING:
    from agentm.harness.scenario import ScenarioWiring, SetupContext


# ---------------------------------------------------------------------------
# Shared profile tool logic — single implementation, two call modes
# ---------------------------------------------------------------------------

def _do_query(
    store: Any,
    service_names: str,
    anomalous_only: bool,
) -> str:
    """Shared query logic for profile tools.

    *service_names* may be a single name, comma-separated names, or empty.
    When empty, returns all profiles grouped by anomalous/healthy status.
    """
    if service_names:
        names = [n.strip() for n in service_names.split(",") if n.strip()]
        if len(names) == 1:
            return store.format_profile(names[0])
        return "\n\n".join(store.format_profile(n) for n in names)
    if anomalous_only:
        profiles = store.query(anomalous_only=True)
        if not profiles:
            return "No anomalous services recorded yet."
        return "\n".join(store.format_profile(p.service_name) for p in profiles)
    return store.format_for_llm() or "No service profiles recorded yet."


def _update_service_profile(
    profile_store: Any,
    service_name: str,
    is_anomalous: bool,
    anomaly_summary: str = "",
    upstream_services: Optional[list[str]] = None,
    downstream_services: Optional[list[str]] = None,
    data_sources_queried: Optional[list[str]] = None,
    key_observation: str = "",
    source_agent_id: str = "",
    source_task_type: str = "scout",
    related_hypothesis_id: Optional[str] = None,
) -> str:
    """Update the shared Service Profile for a service.

    Keep all inputs SHORT -- a profile is a quick-reference card, not a report.
    - anomaly_summary: one terse line, e.g., "p99 60s vs 4s, 45% errors"
    - key_observation: one sentence max, factual, no reasoning

    TIP: Call query_service_profile first to check if another agent has
    already recorded findings for this service. Only add NEW information.
    """
    existing = profile_store.get(service_name)
    existing_obs_count = len(existing.observations) if existing else 0

    profile = profile_store.update(
        service_name,
        agent_id=source_agent_id,
        task_type=source_task_type,
        is_anomalous=is_anomalous,
        anomaly_summary=anomaly_summary,
        upstream_services=upstream_services,
        downstream_services=downstream_services,
        data_sources_queried=data_sources_queried,
        key_observation=key_observation,
        related_hypothesis_id=related_hypothesis_id,
    )
    content = profile_store.format_profile(profile.service_name)

    if existing_obs_count > 0:
        content = (
            f"NOTE: This profile already had {existing_obs_count} observation(s) "
            f"from other agents. Use query_service_profile to review "
            f"existing data before adding more updates.\n\n"
        ) + content
    return content


def _query_service_profile(
    profile_store: Any,
    request: str,
    service_names: str = "",
    anomalous_only: bool = False,
) -> str:
    """Query the shared Service Profile store.

    Args:
        request: A short description of what you want to look up.
        service_names: One or more service names, comma-separated.
            Empty string returns ALL profiles grouped by anomalous/healthy.
        anomalous_only: If True, return only anomalous services.
    """
    return _do_query(profile_store, service_names, anomalous_only)


# ---------------------------------------------------------------------------
# Tool builders
# ---------------------------------------------------------------------------

def _build_profile_tools(profile_store: Any) -> list[Any]:
    """Build service profile tools (shared by orchestrator and worker)."""
    if profile_store is None:
        return []

    update_fn = partial(_update_service_profile, profile_store)
    query_fn = partial(_query_service_profile, profile_store)
    return [
        tool_from_function(
            update_fn,
            name="update_service_profile",
            description=(_update_service_profile.__doc__ or "").strip(),
        ),
        tool_from_function(
            query_fn,
            name="query_service_profile",
            description=(_query_service_profile.__doc__ or "").strip(),
        ),
    ]


def _build_rca_orchestrator_tools(
    trajectory: Any | None,
    hypothesis_store: Any,
    profile_store: Any,
) -> list[Any]:
    """Build orchestrator-side RCA tools as SDK Tool instances."""
    from agentm.harness.tool import Tool

    tools: list[Tool] = []

    # -- Hypothesis tools --

    async def update_hypothesis(
        id: str,
        description: str,
        status: Literal[
            "formed",
            "investigating",
            "confirmed",
            "rejected",
            "refined",
            "inconclusive",
        ] = "formed",
        evidence_summary: Optional[str] = None,
        parent_id: Optional[str] = None,
    ) -> str:
        """Create or update a hypothesis in the investigation."""
        if hypothesis_store is not None:
            hypothesis_store.update(
                id=id,
                description=description,
                status=status,
                evidence_summary=evidence_summary,
                parent_id=parent_id,
            )

        content = f"Hypothesis {id} updated: {status} — {description}"

        if trajectory is not None:
            await trajectory.record(
                event_type="hypothesis_update",
                agent_path=["orchestrator"],
                data={
                    "hypothesis_id": id,
                    "status": status,
                    "description": description,
                    "evidence_summary": evidence_summary,
                    "parent_id": parent_id,
                },
                metadata={"hypothesis_id": id},
            )

        return content

    async def remove_hypothesis(
        id: str,
    ) -> str:
        """Remove a hypothesis from the investigation."""
        if hypothesis_store is not None:
            hypothesis_store.remove(id)

        if trajectory is not None:
            await trajectory.record(
                event_type="hypothesis_update",
                agent_path=["orchestrator"],
                data={
                    "hypothesis_id": id,
                    "status": "removed",
                    "description": "",
                },
                metadata={"hypothesis_id": id},
            )

        return f"Hypothesis {id} removed"

    tools.append(tool_from_function(update_hypothesis))
    tools.append(tool_from_function(remove_hypothesis))

    # -- Service profile tools (same as worker — sync, handled by ainvoke) --
    tools.extend(_build_profile_tools(profile_store))

    return tools


def _build_rca_worker_tools(profile_store: Any) -> list[Any]:
    """Build worker-side RCA tools as SDK Tool instances."""
    return _build_profile_tools(profile_store)


# ---------------------------------------------------------------------------
# Scenario class
# ---------------------------------------------------------------------------

class RCAScenario:
    """Scenario implementation for hypothesis-driven Root Cause Analysis."""

    @property
    def name(self) -> str:
        return "hypothesis_driven"

    def setup(self, ctx: SetupContext) -> ScenarioWiring:
        """Wire up the RCA scenario: stores, tools, context, schemas, hooks."""
        from agentm.harness.scenario import ScenarioWiring
        from agentm.models.data import OrchestratorHooks
        from agentm.scenarios.rca.answer_schemas import (
            DeepAnalyzeAnswer,
            ScoutAnswer,
            VerifyAnswer,
        )
        from agentm.scenarios.rca.formatters import format_rca_context
        from agentm.scenarios.rca.hypothesis_store import HypothesisStore
        from agentm.scenarios.rca.output import CausalGraph
        from agentm.scenarios.rca.service_profile import ServiceProfileStore

        hypothesis_store = HypothesisStore()
        profile_store = ServiceProfileStore()

        orch_tools = _build_rca_orchestrator_tools(
            ctx.trajectory, hypothesis_store, profile_store,
        )
        worker_tools = _build_rca_worker_tools(profile_store)

        format_fn = partial(
            format_rca_context,
            profile_store=profile_store,
            hypothesis_store=hypothesis_store,
        )

        hooks = OrchestratorHooks(
            think_stall_enabled=True,
        )

        return ScenarioWiring(
            orchestrator_tools=orch_tools,
            worker_tools=worker_tools,
            format_context=format_fn,
            answer_schemas={
                "scout": ScoutAnswer,
                "deep_analyze": DeepAnalyzeAnswer,
                "verify": VerifyAnswer,
            },
            output_schema=CausalGraph,
            hooks=hooks,
        )
