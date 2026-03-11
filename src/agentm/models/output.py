"""Structured output schemas for Orchestrator final responses.

These Pydantic models are used with create_react_agent's response_format
parameter. When set, the framework appends a generate_structured_response
node that invokes the LLM with structured output after the ReAct loop.

Registry lookup: OUTPUT_SCHEMAS maps schema_name -> model class.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# CausalGraph — RCA investigation output
# ---------------------------------------------------------------------------


class CausalGraphNode(BaseModel):
    """A service node on the fault propagation path."""

    component: str = Field(description="Canonical service name from trace/metric data.")
    state: list[str] = Field(
        description="Observed states using Available States values."
    )
    timestamp: str = Field(
        description="Unix nanosecond timestamp of first anomaly, or empty string if unknown.",
    )


class CausalGraphEdge(BaseModel):
    """Directed fault propagation link between two services."""

    source: str = Field(
        description="Source service where the fault originated or passed through."
    )
    target: str = Field(
        description="Target service that received the propagated fault."
    )


class ComponentMapping(BaseModel):
    """Mapping from a component/pod name to its canonical service name."""

    component_name: str = Field(
        description="Span or pod component name as seen in trace data."
    )
    service_name: str = Field(
        description="Canonical service name this component belongs to."
    )


class CausalGraph(BaseModel):
    """Fault propagation graph — the final output of an RCA investigation.

    Traces HOW a failure propagated from the root cause to the alarm/SLO
    endpoint. Only includes services on the propagation path, not all
    anomalous services.
    """

    nodes: list[CausalGraphNode] = Field(
        description="Services on the fault propagation path ONLY.",
    )
    edges: list[CausalGraphEdge] = Field(
        description="Directed fault propagation links backed by trace or co-location evidence.",
    )
    root_causes: list[CausalGraphNode] = Field(
        description="Where the failure ORIGINATED. Typically 1-2 services.",
    )
    component_to_service: list[ComponentMapping] = Field(
        description="Mappings from span/pod component names to canonical service names. Empty list if not applicable.",
    )


# ---------------------------------------------------------------------------
# Schema registry
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# KnowledgeSummary — Memory Extraction investigation output
# ---------------------------------------------------------------------------


class KnowledgeSummary(BaseModel):
    """Final output of a memory_extraction run."""

    entries_created: int = Field(description="Number of new knowledge entries written.")
    entries_updated: int = Field(description="Number of existing entries updated.")
    entries_skipped: int = Field(
        description="Number of entries skipped (duplicate or low confidence)."
    )
    categories: list[str] = Field(
        description="Distinct knowledge categories populated during this run."
    )
    summary: str = Field(
        description="One-paragraph prose summary of what was learned and stored."
    )


OUTPUT_SCHEMAS: dict[str, type[BaseModel]] = {
    "CausalGraph": CausalGraph,
    "KnowledgeSummary": KnowledgeSummary,
}


def get_output_schema(schema_name: str) -> type[BaseModel]:
    """Look up an output schema by name.

    Raises ValueError if the schema is not registered.
    """
    if schema_name not in OUTPUT_SCHEMAS:
        available = list(OUTPUT_SCHEMAS.keys())
        raise ValueError(
            f"Unknown output schema: {schema_name!r}. Available: {available}"
        )
    return OUTPUT_SCHEMAS[schema_name]
