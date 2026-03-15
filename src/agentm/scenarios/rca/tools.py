"""RCA-specific orchestrator tools (hypothesis + service profile management)."""

from __future__ import annotations

from typing import Annotated, Any, Callable, Literal, Optional

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, InjectedToolCallId, StructuredTool
from langgraph.types import Command

from agentm.scenarios.rca.enums import HypothesisStatus
from agentm.scenarios.rca.service_profile import ServiceProfileStore


def create_rca_tools(
    trajectory: Any | None = None,
    profile_store: ServiceProfileStore | None = None,
) -> dict[str, Any]:
    """Create RCA-specific orchestrator tools.

    Returns a dict of tool name -> callable for hypothesis management
    and service profile management.  The special key ``_worker_profile_tools``
    contains a list of ``BaseTool`` instances for injection into workers.
    """

    # ------------------------------------------------------------------
    # Hypothesis tools (orchestrator only, return Command)
    # ------------------------------------------------------------------

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
        tool_call_id: Annotated[str, InjectedToolCallId] = "",
    ) -> Command:
        """Create or update a hypothesis in the DiagnosticNotebook."""
        HypothesisStatus(status)  # validate the value matches enum
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
                hypothesis_id=id,
            )

        return Command(
            update={
                "messages": [ToolMessage(content=content, tool_call_id=tool_call_id)]
            }
        )

    async def remove_hypothesis(
        id: str,
        tool_call_id: Annotated[str, InjectedToolCallId] = "",
    ) -> Command:
        """Remove a hypothesis from the DiagnosticNotebook."""
        if trajectory is not None:
            await trajectory.record(
                event_type="hypothesis_update",
                agent_path=["orchestrator"],
                data={
                    "hypothesis_id": id,
                    "status": "removed",
                    "description": "",
                },
                hypothesis_id=id,
            )

        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Hypothesis {id} removed",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )

    result: dict[str, Any] = {
        "update_hypothesis": update_hypothesis,
        "remove_hypothesis": remove_hypothesis,
    }

    # ------------------------------------------------------------------
    # Service profile tools (orchestrator + worker versions)
    # ------------------------------------------------------------------

    if profile_store is not None:
        # ---- Orchestrator versions (return Command) ----

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
            tool_call_id: Annotated[str, InjectedToolCallId] = "",
        ) -> Command:
            """Update the shared Service Profile for a service.

            Keep all inputs SHORT — a profile is a quick-reference card, not a report.
            - anomaly_summary: one terse line, e.g., "p99 60s vs 4s, 45% errors"
            - key_observation: one sentence max, factual, no reasoning
            """
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
            return Command(
                update={
                    "messages": [ToolMessage(content=content, tool_call_id=tool_call_id)]
                }
            )

        async def query_service_profile(
            service_name: str,
            anomalous_only: bool = False,
            list_all: bool = False,
            tool_call_id: Annotated[str, InjectedToolCallId] = "",
        ) -> Command:
            """Query the shared Service Profile store. Pass service_name="" to query all."""
            content = _do_query(profile_store, service_name, anomalous_only, list_all)
            return Command(
                update={
                    "messages": [ToolMessage(content=content, tool_call_id=tool_call_id)]
                }
            )

        result["update_service_profile"] = update_service_profile
        result["query_service_profile"] = query_service_profile

        # ---- Worker versions (return str, no Command/InjectedToolCallId) ----

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

            Keep all inputs SHORT — a profile is a quick-reference card, not a report.
            - anomaly_summary: one terse line, e.g., "p99 60s vs 4s, 45% errors"
            - key_observation: one sentence max, factual, no reasoning
            """
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
            return profile_store.format_profile(profile.service_name)

        def query_service_profile_worker(
            service_name: str,
            anomalous_only: bool = False,
            list_all: bool = False,
        ) -> str:
            """Query the shared Service Profile store. Pass service_name="" to query all."""
            return _do_query(profile_store, service_name, anomalous_only, list_all)

        # Build StructuredTool instances with LLM-facing names (no _worker suffix)
        worker_update = StructuredTool.from_function(
            func=update_service_profile_worker,
            name="update_service_profile",
            description=update_service_profile_worker.__doc__ or "",
        )
        worker_query = StructuredTool.from_function(
            func=query_service_profile_worker,
            name="query_service_profile",
            description=query_service_profile_worker.__doc__ or "",
        )
        result["_worker_profile_tools"] = [worker_update, worker_query]

    return result


def _do_query(
    store: ServiceProfileStore,
    service_name: str,
    anomalous_only: bool,
    list_all: bool,
) -> str:
    """Shared query logic for both orchestrator and worker versions."""
    if service_name:
        return store.format_profile(service_name)
    if list_all:
        text = store.format_for_llm()
        return text if text else "No service profiles recorded yet."
    if anomalous_only:
        profiles = store.query(anomalous_only=True)
        if not profiles:
            return "No anomalous services recorded yet."
        lines = [store.format_profile(p.service_name) for p in profiles]
        return "\n".join(lines)
    return store.format_for_llm() or "No service profiles recorded yet."
