"""Priority helpers for seed-rooted propagation edge scheduling."""
from __future__ import annotations

from collections import Counter, deque
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .candidates import service_from_subject
from .final_checks import frontend_anomalies, frontend_services, reaches
from .schema import CandidateEdge

_INF = 1_000_000


@dataclass(frozen=True)
class EdgePriorityContext:
    """Static + current graph facts used to order expansion candidates."""

    targets: frozenset[str]
    distance_to_target: Mapping[str, int]
    anomaly_services: frozenset[str]
    frontend_anomaly_terms: frozenset[str]
    accepted_adj: Mapping[str, Sequence[str]]
    source_adj_by_seed: Mapping[str, Mapping[str, Sequence[str]]]
    rejected_gate_counts: Mapping[tuple[str, str, str], int]


@dataclass(frozen=True)
class PropagationNeeds:
    """Unresolved final obligations that propagation can still close."""

    passed: bool
    unresolved_seed_sources: frozenset[str]
    needed_frontend_targets: frozenset[str]
    distance_to_needed_frontend: Mapping[str, int]


def build_priority_context(
    *,
    graph: Mapping[str, Sequence[Sequence[str]]],
    data_profile: Mapping[str, Any],
    data_dir: str | None = None,
    entry_services: set[str],
    anomaly_inventory: Sequence[Mapping[str, Any]],
    accepted_adj: Mapping[str, Sequence[str]],
    gate_log: Sequence[Mapping[str, Any]],
    source_adj_by_seed: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
) -> EdgePriorityContext:
    """Build per-round scheduling context without changing graph semantics."""
    targets = frozenset(
        frontend_services(
            graph=graph,
            data_profile=data_profile,
            entry_services=entry_services,
        )
    )
    anomaly_services = frozenset(
        service
        for record in anomaly_inventory
        if record.get("status") == "changed"
        for service in [service_from_subject(record.get("subject"))]
        if service
    )
    frontend_records = (
        frontend_anomalies(
            data_dir=data_dir or "",
            anomaly_inventory=anomaly_inventory,
            frontend_set=set(targets),
        )
        if targets
        else []
    )
    frontend_anomaly_terms = frozenset(
        _compact_text(record.get(field))
        for record in frontend_records
        for field in ("id", "component", "summary")
        if _compact_text(record.get(field))
    )
    rejected_gate_counts: Counter[tuple[str, str, str]] = Counter()
    for entry in gate_log:
        if entry.get("task") != "hop":
            continue
        gate = entry.get("gate")
        if not isinstance(gate, Mapping) or gate.get("accepted"):
            continue
        from_service = str(entry.get("from") or "")
        to_service = str(entry.get("to") or "")
        source_seed = str(entry.get("source_seed") or "")
        if from_service and to_service:
            rejected_gate_counts[(source_seed, from_service, to_service)] += 1
    return EdgePriorityContext(
        targets=targets,
        distance_to_target=_reverse_distances(graph, targets),
        anomaly_services=anomaly_services,
        frontend_anomaly_terms=frontend_anomaly_terms,
        accepted_adj=accepted_adj,
        source_adj_by_seed=source_adj_by_seed or {},
        rejected_gate_counts=rejected_gate_counts,
    )


def build_propagation_needs(
    *,
    report: Mapping[str, Any],
    graph: Mapping[str, Sequence[Sequence[str]]],
) -> PropagationNeeds:
    """Extract the propagation-relevant part of the final-check report."""
    unresolved_seeds: set[str] = set()
    needed_frontends: set[str] = set()
    for issue in report.get("issues", []):
        if not isinstance(issue, Mapping):
            continue
        check = issue.get("check")
        if check == "seed_reaches_entry":
            unresolved_seeds.add(str(issue.get("item") or ""))
            continue
        if check != "frontend_anomaly_explained":
            continue
        details = issue.get("details", {})
        if isinstance(details, Mapping):
            service = details.get("service")
            if service:
                needed_frontends.add(str(service))

    unresolved_seeds.discard("")
    needed_frontends.discard("")
    frontend_targets = frozenset(needed_frontends)
    return PropagationNeeds(
        passed=bool(report.get("passed")),
        unresolved_seed_sources=frozenset(unresolved_seeds),
        needed_frontend_targets=frontend_targets,
        distance_to_needed_frontend=_reverse_distances(graph, frontend_targets)
        if frontend_targets
        else {},
    )


def candidate_addresses_needs(
    candidate: CandidateEdge,
    needs: PropagationNeeds,
    *,
    active_seed_sources: Collection[str] | None = None,
) -> bool:
    """Return whether an edge can still help close an unresolved obligation."""
    if needs.passed:
        return False
    seed_sources = (
        needs.unresolved_seed_sources
        if active_seed_sources is None
        else active_seed_sources
    )
    if candidate["source_seed"] in seed_sources:
        return True
    if active_seed_sources:
        return False
    if not needs.needed_frontend_targets:
        return False
    return candidate["to_service"] in needs.distance_to_needed_frontend


def prioritize_candidates(
    candidates: Sequence[CandidateEdge],
    context: EdgePriorityContext,
) -> list[CandidateEdge]:
    """Return candidates ordered for seed-rooted, target-directed expansion."""
    return sorted(candidates, key=lambda candidate: edge_priority(candidate, context))


def edge_priority(
    candidate: CandidateEdge,
    context: EdgePriorityContext,
) -> tuple[int, int, int, int, int, int, int, int, int, str, str, str]:
    """Rank a candidate; lower tuple values run earlier.

    The ordering keeps expansion rooted at confirmed seeds but spends work first
    on edges that can close final obligations. It never drops an edge.
    """
    source_seed = candidate["source_seed"]
    from_service = candidate["from_service"]
    to_service = candidate["to_service"]
    targets = context.targets
    to_distance = context.distance_to_target.get(to_service, _INF)
    from_distance = context.distance_to_target.get(from_service, _INF)
    source_adj = context.source_adj_by_seed.get(source_seed, context.accepted_adj)

    seed_resolved = int(
        bool(targets)
        and any(
            source_seed == target or reaches(source_adj, source_seed, target)
            for target in targets
        )
    )
    # Prefer edges that move closer to an SLO target. If the current node is not
    # known to reach a target but the neighbor is, this edge enters a viable path.
    if to_distance >= _INF:
        reach_rank = 2
    elif from_distance >= _INF or to_distance < from_distance:
        reach_rank = 0
    else:
        reach_rank = 1

    terminal_rank = 0 if to_service in targets else 1
    frontend_affinity = -_frontend_anomaly_affinity(
        to_service,
        context.frontend_anomaly_terms,
    )
    anomaly_rank = 0 if to_service in context.anomaly_services else 1
    rel_rank = _relationship_rank(str(candidate.get("rel_type", "")))
    depth = _accepted_distance(source_adj, source_seed, from_service)
    retry_count = context.rejected_gate_counts.get(
        (source_seed, from_service, to_service),
        0,
    ) + context.rejected_gate_counts.get(("", from_service, to_service), 0)

    return (
        seed_resolved,
        reach_rank,
        to_distance,
        terminal_rank,
        frontend_affinity,
        rel_rank,
        anomaly_rank,
        depth,
        retry_count,
        source_seed,
        from_service,
        to_service,
    )


def _reverse_distances(
    graph: Mapping[str, Sequence[Sequence[str]]],
    targets: frozenset[str],
) -> dict[str, int]:
    reverse: dict[str, list[str]] = {}
    for src, neighbors in graph.items():
        for info in neighbors:
            if not info:
                continue
            reverse.setdefault(str(info[0]), []).append(src)

    distances: dict[str, int] = {target: 0 for target in targets}
    queue: deque[str] = deque(targets)
    while queue:
        cur = queue.popleft()
        next_distance = distances[cur] + 1
        for prev in reverse.get(cur, []):
            if prev in distances:
                continue
            distances[prev] = next_distance
            queue.append(prev)
    return distances


def _relationship_rank(rel_type: str) -> int:
    if rel_type == "callee_to_caller":
        return 0
    if rel_type == "other":
        return 1
    if rel_type == "caller_to_callee":
        return 2
    return 3


def _compact_text(value: object) -> str:
    return "".join(ch for ch in str(value or "").lower() if ch.isalnum())


def _frontend_anomaly_affinity(service: str, frontend_terms: frozenset[str]) -> int:
    """Prefer intermediates whose name appears in concrete frontend alarms."""
    if not frontend_terms:
        return 0
    compact = _compact_text(service)
    forms = {compact}
    if compact.startswith("ts"):
        forms.add(compact[2:])
    forms |= {
        form[:-7]
        for form in list(forms)
        if form.endswith("service") and len(form) > len("service")
    }
    return max(
        (
            len(form)
            for form in forms
            if len(form) >= 4
            for term in frontend_terms
            if form in term
        ),
        default=0,
    )


def _accepted_distance(
    adj: Mapping[str, Sequence[str]],
    src: str,
    dst: str,
) -> int:
    if src == dst:
        return 0
    queue: deque[tuple[str, int]] = deque([(src, 0)])
    seen = {src}
    while queue:
        cur, distance = queue.popleft()
        for nxt in adj.get(cur, []):
            if nxt == dst:
                return distance + 1
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, distance + 1))
    return _INF


__all__ = [
    "EdgePriorityContext",
    "PropagationNeeds",
    "build_priority_context",
    "build_propagation_needs",
    "candidate_addresses_needs",
    "edge_priority",
    "prioritize_candidates",
]
