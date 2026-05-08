"""Structured output schemas for RCA conclusions.

Currently consumed only as documentation; the orchestrator emits the final
report as free-text per ``prompts/orchestrator.md``. A future extension may
wire ``CausalGraph`` as an enforced output schema.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


PodState = Literal[
    "HEALTHY", "KILLED", "PROCESS_PAUSED", "HIGH_CPU", "HIGH_MEMORY",
    "HIGH_DISK_USAGE", "HIGH_NETWORK_ERRORS", "HIGH_HTTP_LATENCY",
    "HIGH_GC_PRESSURE", "DISK_SLOW", "DISK_FAULT", "DISK_CORRUPTION",
    "DISK_PERMISSION", "NETWORK_DELAY", "NETWORK_LOSS", "NETWORK_DUPLICATION",
    "NETWORK_CORRUPTION", "NETWORK_REORDERING", "NETWORK_BANDWIDTH_LIMIT",
    "NETWORK_PARTITION", "DNS_ERROR", "CLOCK_SKEW", "NO_CPU_AVAILABLE",
    "NO_MEMORY_AVAILABLE",
]
ServiceState = Literal["HEALTHY", "HIGH_ERROR_RATE", "HIGH_LATENCY", "UNAVAILABLE"]
SpanState = Literal[
    "HEALTHY", "HIGH_P99_LATENCY", "HIGH_AVG_LATENCY", "HIGH_ERROR_RATE",
    "TIMEOUT", "HIGH_LOG_ERROR", "CONNECTION_RESET", "MALFORMED_RESPONSE",
]
ComponentKind = Literal[
    "service", "span", "pod", "container", "deployment", "stateful_set", "replica_set",
]


class FaultNode(BaseModel):
    component: str = Field(description="Canonical service / span / pod name")
    kind: ComponentKind
    state: str = Field(description="One of the allowed states for this kind")
    evidence: str = Field(description="Evidence tag + measurement (TRACE / METRIC / LOG)")


class FaultEdge(BaseModel):
    source: str
    target: str
    evidence: str = Field(description="parent_span_id link or co-location proof")


class CausalGraph(BaseModel):
    incident: str = Field(description="One-line incident description from the input")
    root_causes: list[FaultNode] = Field(
        description="Earliest services where the failure originated (typically 1-2)",
    )
    chain: list[FaultNode] = Field(
        description="Services on the propagation path between root cause and alarm endpoint",
    )
    edges: list[FaultEdge] = Field(
        description="Evidence-backed propagation links",
    )
