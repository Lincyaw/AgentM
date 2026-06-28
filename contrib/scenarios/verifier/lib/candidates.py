"""Candidate-edge enumeration for verifier propagation."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .schema import CandidateEdge


def graph_rel_type(
    graph: Mapping[str, Sequence[Sequence[str]]],
    from_service: str,
    to_service: str,
) -> str | None:
    for info in graph.get(from_service, []):
        if len(info) >= 2 and info[0] == to_service:
            return str(info[1])
    return None


def service_from_subject(subject: object) -> str | None:
    text = str(subject or "")
    if text.startswith("svc:"):
        return text.removeprefix("svc:")
    return None


def structural_candidates(
    graph: Mapping[str, Sequence[Sequence[str]]],
    *,
    source_seed: str,
    from_service: str,
) -> list[CandidateEdge]:
    out: list[CandidateEdge] = []
    for info in graph.get(from_service, []):
        if len(info) < 2:
            continue
        out.append(
            {
                "source_seed": source_seed,
                "from_service": from_service,
                "to_service": str(info[0]),
                "rel_type": str(info[1]),
                "source": "structural_frontier",
                "reason": "neighbor relationship discovered from traces/metrics",
            }
        )
    return out


def anomaly_candidates(
    graph: Mapping[str, Sequence[Sequence[str]]],
    anomaly_inventory: Sequence[Mapping[str, Any]],
    *,
    source_seed: str,
    from_service: str,
    existing_targets: set[str],
    max_candidates: int = 3,
) -> list[CandidateEdge]:
    """Prioritize anomalous direct neighbors without opening arbitrary hops."""
    out: list[CandidateEdge] = []
    seen: set[str] = set(existing_targets)
    for record in anomaly_inventory:
        if record.get("status") != "changed":
            continue
        target = service_from_subject(record.get("subject"))
        if not target or target == from_service or target in seen:
            continue
        rel_type = graph_rel_type(graph, from_service, target)
        if not rel_type:
            continue
        seen.add(target)
        out.append(
            {
                "source_seed": source_seed,
                "from_service": from_service,
                "to_service": target,
                "rel_type": rel_type,
                "source": "anomaly_inventory",
                "reason": str(record.get("summary") or "changed telemetry signal"),
                "anomaly_ids": [str(record.get("id", ""))],
            }
        )
        if len(out) >= max_candidates:
            break
    return out


__all__ = [
    "anomaly_candidates",
    "graph_rel_type",
    "service_from_subject",
    "structural_candidates",
]
