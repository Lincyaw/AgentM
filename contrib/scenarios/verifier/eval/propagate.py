"""BFS fault propagation orchestration."""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from graph import SYNTHETIC, _duckdb_conn, check_node_degraded
from hop import run_hop
from injection import _load_fault_doc, get_target_evidence


def _verify_edge_sql(data_dir: Path, from_svc: str, to_svc: str) -> bool:
    """Check that a direct relationship exists between the two services."""
    conn = _duckdb_conn(data_dir)
    row = conn.execute(
        "SELECT COUNT(*) FROM normal_traces child "
        "JOIN normal_traces parent "
        "  ON child.parent_span_id = parent.span_id "
        "WHERE (parent.service_name = ? "
        "   AND child.service_name = ?) "
        "   OR (parent.service_name = ? "
        "   AND child.service_name = ?)",
        [to_svc, from_svc, from_svc, to_svc],
    ).fetchone()
    conn.close()
    return bool(row and row[0] >= 5)


def propagate(
    data_dir: Path,
    injections: list[dict[str, str]],
    neighbor_graph: dict[str, list[tuple[str, str]]],
    out_dir: Path,
    *,
    budget: int,
    parallel: int,
    infra_nodes: set[str] | None = None,
    node_map: dict[str, str] | None = None,
) -> dict:
    """Edge-level fault propagation.

    Every confirmed node enqueues ALL its neighbours. Each directed edge
    (from, to) is independently evaluated — a service rejected via one
    relationship (e.g. co_deployed) can still be confirmed via another
    (e.g. caller_to_callee) from a different source in a later round.
    """
    infra_nodes = infra_nodes or set()
    confirmed: set[str] = set()
    checked_edges: set[tuple[str, str]] = set()
    queue: deque[str] = deque()
    edges: list[tuple[str, str]] = []
    node_evidence: dict[str, dict] = {}
    hop_log: list[dict] = []

    fault_kind = injections[0]["chaos_type"] if injections else "unknown"
    inj_target = injections[0]["target"] if injections else "unknown"

    # The full fault set, handed to every hop. A node downstream of two
    # coexisting faults must be judged against both — the per-branch
    # inherited fault below is only a "which seed reached me" hint.
    all_faults = [
        (inj["chaos_type"], inj["target"], inj.get("params", ""))
        for inj in injections
        if inj.get("target") and inj["target"] not in SYNTHETIC
    ]
    fault_docs = {fk: _load_fault_doc(fk) for fk, _, _ in all_faults}

    # Per-branch fault context: each node carries the (fault_kind, target)
    # of the seed it descends from, so a hop reached from the NetworkLoss
    # seed is judged as NetworkLoss — not as injections[0]'s fault. Without
    # this, every hop in a multi-injection case is told the wrong fault.
    node_fault: dict[str, tuple[str, str]] = {}

    for inj in injections:
        target = inj["target"]
        if target and target not in SYNTHETIC:
            ev = get_target_evidence(data_dir, target)
            confirmed.add(target)
            queue.append(target)
            node_evidence[target] = {"source": "injection_target", **ev}
            node_fault[target] = (inj["chaos_type"], target)

    round_n = 0
    while queue:
        round_n += 1
        level_size = len(queue)

        sql_edges: list[tuple[str, str]] = []
        # Each entry is one edge to evaluate: (from_service, to_service, rel_type)
        pending_hops: list[tuple[str, str, str]] = []

        for _ in range(level_size):
            current = queue.popleft()
            for neighbor, rel_type in neighbor_graph.get(current, []):
                if neighbor in SYNTHETIC:
                    continue
                edge = (current, neighbor)
                if edge in checked_edges:
                    continue
                checked_edges.add(edge)

                if neighbor in confirmed:
                    sql_edges.append(edge)
                else:
                    pending_hops.append((current, neighbor, rel_type))

        for from_svc, to_svc in sql_edges:
            if _verify_edge_sql(data_dir, from_svc, to_svc):
                edges.append((from_svc, to_svc))
                hop_log.append({
                    "round": round_n, "from": from_svc, "to": to_svc,
                    "verdict": "edge_sql",
                })
                print(f"  edge (SQL): {from_svc} -> {to_svc}")

        # co_deployed pre-check: skip edges to services on healthy nodes
        # when ALL edges to that target are co_deployed.
        if node_map:
            _checked_nodes: dict[str, bool] = {}
            # Group by target to check if ALL edges are co_deployed
            target_rels: dict[str, list[tuple[str, str, str]]] = {}
            for from_svc, to_svc, rel in pending_hops:
                target_rels.setdefault(to_svc, []).append(
                    (from_svc, to_svc, rel)
                )
            skip_edges: set[tuple[str, str]] = set()
            for to_svc, hop_list in target_rels.items():
                if all(rel == "co_deployed" for _, _, rel in hop_list):
                    node = node_map.get(to_svc)
                    if node:
                        if node not in _checked_nodes:
                            _checked_nodes[node] = check_node_degraded(
                                data_dir, node,
                            )
                        if not _checked_nodes[node]:
                            for from_svc, _, _ in hop_list:
                                skip_edges.add((from_svc, to_svc))
                            hop_log.append({
                                "round": round_n, "from": "node_health",
                                "to": to_svc,
                                "verdict": "co_deployed_node_healthy",
                            })
                            print(f"    skip co_deployed {to_svc}: "
                                  f"node healthy")
            pending_hops = [
                h for h in pending_hops
                if (h[0], h[1]) not in skip_edges
            ]

        # Filter out edges whose target was confirmed earlier in this round
        # (by a parallel agent or a previous edge in this batch).
        pending_hops = [
            h for h in pending_hops if h[1] not in confirmed
        ]

        if pending_hops:
            print(f"  round {round_n}: {len(pending_hops)} agents, "
                  f"{len(sql_edges)} SQL edges")
            with ThreadPoolExecutor(max_workers=parallel) as pool:
                futures = {}
                for from_svc, to_svc, rel_type in pending_hops:
                    hop_fk, hop_target = node_fault.get(
                        from_svc, (fault_kind, inj_target)
                    )
                    fut = pool.submit(
                        run_hop, data_dir,
                        from_svc, to_svc, rel_type,
                        hop_fk, hop_target,
                        all_faults, fault_docs,
                        out_dir, budget,
                        to_svc in infra_nodes,
                        node_evidence.get(from_svc),
                    )
                    futures[fut] = (from_svc, to_svc, rel_type)
                for future in as_completed(futures):
                    from_svc, to_svc, rel_type = futures[future]
                    result = future.result()
                    verdict = result.get("verdict") if result else None
                    hop_log.append({
                        "round": round_n, "from": from_svc,
                        "to": to_svc, "verdict": verdict or "no-result",
                    })
                    print(f"    {from_svc} -> {to_svc}: "
                          f"{verdict or 'no-result'}")
                    if verdict == "confirmed" and to_svc not in confirmed:
                        confirmed.add(to_svc)
                        node_evidence[to_svc] = {
                            "source": "hop_agent", **(result or {}),
                        }
                        if from_svc in node_fault:
                            node_fault[to_svc] = node_fault[from_svc]
                        if to_svc not in infra_nodes:
                            queue.append(to_svc)
                        edges.append((from_svc, to_svc))

    return {
        "confirmed_nodes": sorted(confirmed),
        "edges": edges,
        "node_evidence": node_evidence,
        "hop_log": hop_log,
        "rounds": round_n,
    }
