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
# Shared query logic (used by both orchestrator and worker profile tools)
# ---------------------------------------------------------------------------

def _do_query(
    store: Any,
    service_names: str,
    anomalous_only: bool,
) -> str:
    """Shared query logic for both orchestrator and worker versions.

    *service_names* may be a single name, comma-separated names, or empty.
    When empty, returns all profiles grouped by anomalous/healthy status.
    """
    if service_names:
        names = [n.strip() for n in service_names.split(",") if n.strip()]
        if len(names) == 1:
            return store.format_profile(names[0])
        # Multiple names -- batch query
        parts: list[str] = []
        for name in names:
            parts.append(store.format_profile(name))
        return "\n\n".join(parts)
    # No names specified -- return all profiles
    if anomalous_only:
        profiles = store.query(anomalous_only=True)
        if not profiles:
            return "No anomalous services recorded yet."
        lines = [store.format_profile(p.service_name) for p in profiles]
        return "\n".join(lines)
    return store.format_for_llm() or "No service profiles recorded yet."


# ---------------------------------------------------------------------------
# Tool builders
# ---------------------------------------------------------------------------

def _build_rca_orchestrator_tools(
    trajectory: Any | None,
    hypothesis_store: Any,
    profile_store: Any,
) -> list[Any]:
    """Build orchestrator-side RCA tools as SDK Tool instances.

    Tools capture stores and trajectory via closure.
    """
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

    # -- Service profile tools (orchestrator versions) --

    if profile_store is not None:

        async def update_service_profile(
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

            NOTE: Workers also update profiles during investigation. Before calling
            this, use query_service_profile to check if workers have already recorded
            the information you intend to add. Only call this to add genuinely NEW
            information not already captured by workers.
            """
            # Check existing profile before update to generate hint
            existing = profile_store.get(service_name)
            worker_obs = (
                [
                    o
                    for o in existing.observations
                    if o.source_agent_id != "orchestrator"
                ]
                if existing
                else []
            )

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

            # Hint: if workers already contributed, remind orchestrator
            if worker_obs:
                hint = (
                    f"NOTE: This profile already had {len(worker_obs)} observation(s) "
                    f"from worker agents. Consider using query_service_profile first "
                    f"to review existing data before adding updates.\n\n"
                )
                content = hint + content

            return content

        async def query_service_profile(
            request: str,
            service_names: str = "",
            anomalous_only: bool = False,
        ) -> str:
            """Query the shared Service Profile store.

            Args:
                request: A short description of what you want to look up.
                service_names: One or more service names, comma-separated.
                    E.g. "serviceA" or "serviceA,serviceB,serviceC".
                    Empty string returns ALL profiles grouped by anomalous/healthy.
                anomalous_only: If True, return only anomalous services (ignored
                    when service_names is non-empty).
            """
            return _do_query(profile_store, service_names, anomalous_only)

        tools.append(tool_from_function(update_service_profile))
        tools.append(tool_from_function(query_service_profile))

    return tools


def _build_rca_worker_tools(profile_store: Any) -> list[Any]:
    """Build worker-side RCA tools as SDK Tool instances.

    Worker tools are sync (no trajectory recording) and use the same
    LLM-facing names as the orchestrator versions.
    """
    from agentm.harness.tool import Tool

    if profile_store is None:
        return []

    tools: list[Tool] = []

    def update_service_profile_worker(
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
        # Check existing profile before update to generate hint
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
            hint = (
                f"NOTE: This profile already had {existing_obs_count} observation(s) "
                f"from other agents. Use query_service_profile to review existing "
                f"data before adding more updates.\n\n"
            )
            content = hint + content

        return content

    def query_service_profile_worker(
        request: str,
        service_names: str = "",
        anomalous_only: bool = False,
    ) -> str:
        """Query the shared Service Profile store.

        Args:
            request: A short description of what you want to look up.
            service_names: One or more service names, comma-separated.
                E.g. "serviceA" or "serviceA,serviceB,serviceC".
                Empty string returns ALL profiles grouped by anomalous/healthy.
            anomalous_only: If True, return only anomalous services (ignored
                when service_names is non-empty).
        """
        return _do_query(profile_store, service_names, anomalous_only)

    # Use LLM-facing names (no _worker suffix)
    tools.append(tool_from_function(
        update_service_profile_worker,
        name="update_service_profile",
        description=update_service_profile_worker.__doc__ or "",
    ))
    tools.append(tool_from_function(
        query_service_profile_worker,
        name="query_service_profile",
        description=query_service_profile_worker.__doc__ or "",
    ))

    return tools


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

        # 1. Create domain stores
        hypothesis_store = HypothesisStore()
        profile_store = ServiceProfileStore()

        # 2. Create tools
        orch_tools = _build_rca_orchestrator_tools(
            ctx.trajectory, hypothesis_store, profile_store,
        )
        worker_tools = _build_rca_worker_tools(profile_store)

        # 3. format_context -- zero-arg callable via partial
        format_fn = partial(
            format_rca_context,
            profile_store=profile_store,
            hypothesis_store=hypothesis_store,
        )

        # 4. Hooks -- same as HypothesisDrivenStrategy.orchestrator_hooks()
        hooks = OrchestratorHooks(
            think_stall_enabled=True,
            think_stall_limit=3,
            think_stall_warning=(
                "THINK-STALL WARNING: You have called only `think` for the "
                "last {streak} rounds without taking any action. "
                "Thinking does not advance the investigation.\n\n"
                "You MUST call an action tool NOW — dispatch_agent, "
                "update_hypothesis, or finalize with "
                "<decision>finalize</decision>.\n"
                "Do NOT call think again until you have taken an action."
            ),
            skip_context_on_think_only=True,
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
