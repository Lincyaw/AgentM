"""Deterministic work planner — gaps to verification tasks.

Generates candidate VerificationTasks from gaps, excluding exhausted/rejected
edges, and assigns priorities.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .schema import Gap, GapReport, VerificationTask
from .state import Case, GraphState


def plan_work(
    case: Case,
    state: GraphState,
    gap_report: GapReport,
) -> list[VerificationTask]:
    """Generate prioritized verification tasks from structural gaps."""
    tasks: list[VerificationTask] = []
    seen: set[str] = set()

    for gap in gap_report.gaps:
        new_tasks = _tasks_for_gap(case, state, gap)
        for task in new_tasks:
            if task.edge_key in seen:
                continue
            if task.edge_key in state.exhausted_edges:
                continue
            if task.edge_key in state.rejected_edges:
                continue
            seen.add(task.edge_key)
            tasks.append(task)

    tasks.sort(key=lambda t: t.priority)
    return tasks


def expand_frontier(
    case: Case,
    state: GraphState,
    node: str,
    source_seed: str,
) -> list[VerificationTask]:
    """Generate hop tasks from a newly confirmed node's neighbors.

    For link seeds (link:A->B), expand from both A and B since the link
    is a reified edge between two services, not a service node itself.
    """
    # Resolve effective service nodes to expand from
    expand_from = _resolve_expansion_nodes(node, case)

    tasks: list[VerificationTask] = []
    for from_node in expand_from:
        if from_node in case.infra_set:
            continue

        for neighbor, rel_type in _neighbors(case.graph, from_node):
            task = VerificationTask(
                kind="hop",
                source_seed=source_seed,
                from_entity=from_node,
                to_entity=neighbor,
                rel_type=rel_type,
                gap_ids=[],
                priority=_hop_priority(case, state, from_node, neighbor, source_seed),
                context="",
            )
            if task.edge_key in state.exhausted_edges:
                continue
            if task.edge_key in state.rejected_edges:
                continue
            if neighbor in state.nodes and state.reaches(neighbor, from_node, seed=source_seed):
                continue
            tasks.append(task)

    return tasks


def _resolve_expansion_nodes(node: str, case: Case) -> list[str]:
    """Resolve a node id to the service(s) that should be expanded.

    For link seeds (link:A->B), returns both A and B.
    For regular service nodes, returns [node].
    """
    if node.startswith("link:") and "->" in node:
        link_body = node.removeprefix("link:")
        parts = link_body.split("->", 1)
        return [p.strip() for p in parts if p.strip()]
    return [node]


def _tasks_for_gap(case: Case, state: GraphState, gap: Gap) -> list[VerificationTask]:
    """Generate specific tasks that could close a gap."""
    if gap.kind == "unconfirmed_seed":
        return [_seed_task(case, gap)]

    if gap.kind == "unreachable_seed":
        return _frontier_hop_tasks(case, state, gap)

    if gap.kind == "unexplained_anomaly":
        return _anomaly_hop_tasks(case, state, gap)

    return []


def _seed_task(case: Case, gap: Gap) -> VerificationTask:
    seed = gap.target
    inj = next(
        (i for i in case.injections if (i.get("node_id") or i["target"]) == seed),
        None,
    )
    fault_kind = inj["chaos_type"] if inj else "unknown"
    return VerificationTask(
        kind="seed",
        source_seed=seed,
        from_entity=seed,
        to_entity=seed,
        rel_type="injection",
        gap_ids=[gap.id],
        priority=(0, seed),  # seeds are highest priority
        context=(
            f"Verify injection: {fault_kind} on {seed}. "
            f"{gap.context}"
        ),
        max_retries=case.max_retries,
    )


def _frontier_hop_tasks(
    case: Case,
    state: GraphState,
    gap: Gap,
) -> list[VerificationTask]:
    """From the frontier of a confirmed seed, generate hops toward entry."""
    seed = gap.source_seed
    if not seed:
        return []

    tasks: list[VerificationTask] = []
    frontier = state.frontier_nodes(seed)

    for node in frontier:
        for neighbor, rel_type in _neighbors(case.graph, node):
            if neighbor in state.nodes and state.reaches(neighbor, node, seed=seed):
                continue
            tasks.append(VerificationTask(
                kind="hop",
                source_seed=seed,
                from_entity=node,
                to_entity=neighbor,
                rel_type=rel_type,
                gap_ids=[gap.id],
                priority=_hop_priority(case, state, node, neighbor, seed),
                context=gap.context,
                max_retries=case.max_retries,
            ))

    return tasks


def _anomaly_hop_tasks(
    case: Case,
    state: GraphState,
    gap: Gap,
) -> list[VerificationTask]:
    """Generate hops from confirmed nodes toward the unexplained anomaly's service."""
    target_service = gap.target
    tasks: list[VerificationTask] = []

    for node in state.nodes:
        if node == target_service:
            continue
        if node in case.entry_services:
            continue

        rel_type = _graph_rel_type(case.graph, node, target_service)
        if not rel_type:
            continue

        for seed in sorted(state.node_sources.get(node, set())):
            if state.reaches(node, target_service, seed=seed):
                continue  # already connected
            tasks.append(VerificationTask(
                kind="hop",
                source_seed=seed,
                from_entity=node,
                to_entity=target_service,
                rel_type=rel_type,
                gap_ids=[gap.id],
                priority=_hop_priority(case, state, node, target_service, seed),
                context=gap.context,
                max_retries=case.max_retries,
            ))

    return tasks


# -- Priority ------------------------------------------------------------------

def _hop_priority(
    case: Case,
    state: GraphState,
    from_svc: str,
    to_svc: str,
    source_seed: str,
) -> tuple[Any, ...]:
    """Lower tuple = higher priority."""
    # Prefer hops toward entry services
    distance_to_entry = _min_distance_to_entries(case.graph, to_svc, case.entry_services)
    # Prefer targets with anomalies
    has_anomaly = int(not _service_has_anomaly(to_svc, case.anomaly_inventory))
    # Prefer shorter paths from seed
    depth = _distance_in_adj(state.adj, source_seed, from_svc)
    # Prefer direct relationships over "other"
    rel_rank = 0 if _graph_rel_type(case.graph, from_svc, to_svc) else 1

    return (1, distance_to_entry, has_anomaly, rel_rank, depth, from_svc, to_svc)


def _min_distance_to_entries(
    graph: Mapping[str, Sequence[Sequence[str]]],
    service: str,
    entries: set[str],
) -> int:
    if service in entries:
        return 0
    # BFS backward from entries
    reverse: dict[str, list[str]] = {}
    for src, neighbors in graph.items():
        for info in neighbors:
            if info:
                reverse.setdefault(str(info[0]), []).append(src)
    distances: dict[str, int] = {e: 0 for e in entries}
    queue = list(entries)
    while queue:
        cur = queue.pop(0)
        for prev in reverse.get(cur, []):
            if prev not in distances:
                distances[prev] = distances[cur] + 1
                queue.append(prev)
    return distances.get(service, 999)


def _service_has_anomaly(service: str, inventory: Sequence[Mapping[str, Any]]) -> bool:
    for record in inventory:
        if record.get("status") != "changed":
            continue
        subject = str(record.get("subject", ""))
        if subject == f"svc:{service}":
            return True
    return False


def _distance_in_adj(adj: Mapping[str, Sequence[str]], src: str, dst: str) -> int:
    if src == dst:
        return 0
    queue: list[tuple[str, int]] = [(src, 0)]
    seen = {src}
    while queue:
        cur, d = queue.pop(0)
        for nxt in adj.get(cur, []):
            if nxt == dst:
                return d + 1
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, d + 1))
    return 999


# -- Graph utilities -----------------------------------------------------------

def _neighbors(
    graph: Mapping[str, Sequence[Sequence[str]]],
    service: str,
) -> list[tuple[str, str]]:
    """Return (neighbor, rel_type) pairs from the topology graph."""
    out: list[tuple[str, str]] = []
    for info in graph.get(service, []):
        if len(info) >= 2:
            out.append((str(info[0]), str(info[1])))
    return out


def _graph_rel_type(
    graph: Mapping[str, Sequence[Sequence[str]]],
    from_svc: str,
    to_svc: str,
) -> str | None:
    for info in graph.get(from_svc, []):
        if len(info) >= 2 and info[0] == to_svc:
            return str(info[1])
    return None
