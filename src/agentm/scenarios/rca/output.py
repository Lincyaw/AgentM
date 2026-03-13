"""RCA-specific structured output schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


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
