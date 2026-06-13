"""Fault propagation verification workflow (module mode), fpg-native.

BFS over the neighbor graph, parallel hop agents, optional judge phase.
The internal state IS the fpg graph: nodes are fpg EventNode dicts and
edges are fpg Edge dicts, built incrementally as hops confirm. The fpg
structural rules are enforced at construction time, not filtered
afterwards:

  - injection seeds never gain incoming edges (they stay roots);
  - an edge that would close a cycle is never evaluated;
  - EVERY edge goes through a hop agent — including edges between two
    already-confirmed services (topological adjacency alone is not
    causal evidence);
  - a judge promotion must name the upstream service it cascades
    through, so promoted nodes are connected, never spurious roots.

Input via ``ctx.args``:
    data_dir, graph, injections, infra_nodes, node_map, fault_docs,
    budget, out_dir, skip_propagate, skip_judge,
    window               -- fpg TimeInterval dict for this case
    dataset_profile      -- mechanical table profile for hop prompts
    seed_nodes           -- {svc: fpg EventNode dict} prebuilt seeds
    rel_mechanism        -- {rel_type: fpg edge mechanism}
    existing_state       -- {nodes, edges, verdicts} for skip_propagate

Output: dict with nodes (list of fpg EventNode dicts), edges (list of
fpg Edge dicts), verdicts, hop_log, rounds, judge (if run).
"""
from __future__ import annotations

from collections.abc import Awaitable
from typing import Any, Required, TypedDict, cast

from agentm.extensions.builtin.workflow import AgentResult, WorkflowContext


class Injection(TypedDict, total=False):
    target: Required[str]
    chaos_type: Required[str]
    params: str


HopLogEntry = TypedDict(
    "HopLogEntry",
    {"round": int, "from": str, "to": str, "verdict": str},
)


class HopResult(TypedDict, total=False):
    verdict: Required[str]
    predicate: str | None
    rationale: str
    evidence: list[dict[str, Any]]
    relationship: dict[str, Any] | None
    claim: str


class JudgePromotion(TypedDict, total=False):
    service: Required[str]
    via_service: Required[str]
    predicate: str
    rationale: str


class JudgeReviewResult(TypedDict, total=False):
    add: list[JudgePromotion]
    suggested_remove: list[str]
    rationale: str


class PropagationResult(TypedDict, total=False):
    nodes: Required[list[dict[str, Any]]]
    edges: Required[list[dict[str, Any]]]
    verdicts: Required[dict[str, HopResult]]
    hop_log: Required[list[HopLogEntry]]
    rounds: Required[int]
    judge: JudgeReviewResult


def _reaches(adj: dict[str, list[str]], src: str, dst: str) -> bool:
    stack, seen = [src], {src}
    while stack:
        cur = stack.pop()
        if cur == dst:
            return True
        for nxt in adj.get(cur, []):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return False


def _node_from_verdict(
    svc: str,
    verdict: HopResult,
    window: dict[str, str],
) -> dict[str, Any]:
    """Build an fpg EventNode dict from a confirmed hop verdict."""
    evidence = list(verdict.get("evidence", []))
    relationship = verdict.get("relationship")
    if relationship:
        evidence.append({
            "query": relationship["query"],
            "explanation": "call relationship with the confirmed upstream: "
            + relationship.get("explanation", "see query"),
        })
    predicate = verdict.get("predicate") or "service_degraded"
    node: dict[str, Any] = {
        "kind": "event",
        "id": svc,
        "subject": f"svc:{svc}",
        "predicate": predicate,
        "time": window,
        "grounding": "observed" if evidence else "latent",
        "evidence": evidence,
        "annotation": "auto",
    }
    if predicate == "other":
        node["description"] = verdict.get("claim") or verdict.get("rationale", "")
    return node


def _edge_dict(
    src: str,
    dst: str,
    rel_type: str,
    rel_mechanism: dict[str, str],
    claim: str,
) -> dict[str, Any]:
    mechanism = rel_mechanism.get(rel_type, "other")
    edge: dict[str, Any] = {
        "src": src,
        "dst": dst,
        "mechanism": mechanism,
        "verification": "consistency-checked",
    }
    if mechanism == "other":
        edge["description"] = (
            claim or f"relationship type {rel_type!r} outside the vocabulary"
        )
    elif claim:
        edge["description"] = claim
    return edge


async def run(ctx: WorkflowContext) -> PropagationResult:
    args = ctx.args
    skip_propagate: bool = args.get("skip_propagate", False)
    injections = cast(list[Injection], args["injections"])
    window = cast(dict[str, str], args["window"])
    rel_mechanism = cast(dict[str, str], args.get("rel_mechanism", {}))
    seeds = {inj["target"] for inj in injections if inj.get("target")}

    nodes: dict[str, dict[str, Any]]  # svc -> fpg EventNode dict
    edges: list[dict[str, Any]]
    adj: dict[str, list[str]]  # accepted-edge adjacency, for cycle guard
    in_deg: dict[str, int]
    verdicts: dict[str, HopResult]  # "from__to" -> hop verdict
    hop_log: list[HopLogEntry]
    round_n: int
    node_fault: dict[str, list[str]]

    graph: dict[str, list[list[str]]] = args["graph"]
    dataset_profile = cast(dict[str, Any], args.get("dataset_profile", {}))
    infra_set = set(args.get("infra_nodes", []))
    data_dir: str = args["data_dir"]
    skip_judge: bool = args.get("skip_judge", False)
    fault_docs: dict[str, str] = args.get("fault_docs", {})

    all_faults = [
        [inj["chaos_type"], inj["target"], inj.get("params", "")]
        for inj in injections
        if inj.get("target")
    ]

    def _rebuild_adjacency() -> tuple[dict[str, list[str]], dict[str, int]]:
        a: dict[str, list[str]] = {}
        d: dict[str, int] = {}
        for e in edges:
            a.setdefault(e["src"], []).append(e["dst"])
            d[e["dst"]] = d.get(e["dst"], 0) + 1
        return a, d

    if skip_propagate:
        existing = cast(dict[str, Any], args.get("existing_state", {}))
        nodes = {n["id"]: n for n in existing.get("nodes", [])}
        edges = list(existing.get("edges", []))
        verdicts = cast(dict[str, HopResult], existing.get("verdicts", {}))
        hop_log = cast(list[HopLogEntry], existing.get("hop_log", []))
        round_n = int(existing.get("rounds", 0))
        adj, in_deg = _rebuild_adjacency()
        node_fault = {
            inj["target"]: [inj["chaos_type"], inj["target"]]
            for inj in injections if inj.get("target")
        }
    else:
        ctx.phase("propagate")
        seed_nodes = cast(dict[str, dict[str, Any]], args.get("seed_nodes", {}))
        nodes = dict(seed_nodes)
        edges = []
        adj = {}
        in_deg = {}
        verdicts = {}
        hop_log = []
        round_n = 0
        node_fault = {
            inj["target"]: [inj["chaos_type"], inj["target"]]
            for inj in injections if inj.get("target")
        }

        queue = list(nodes)
        checked_edges: set[str] = set()

        while queue:
            round_n += 1
            batch = list(queue)
            queue = []

            pending_hops: list[list[str]] = []
            for current in batch:
                for neighbor_info in graph.get(current, []):
                    neighbor = neighbor_info[0]
                    rel_type = neighbor_info[1]
                    edge_key = current + "__" + neighbor
                    if edge_key in checked_edges:
                        continue
                    checked_edges.add(edge_key)

                    # fpg root rule: seeds never gain incoming edges.
                    if neighbor in seeds:
                        hop_log.append({
                            "round": round_n, "from": current,
                            "to": neighbor, "verdict": "skipped_seed_target",
                        })
                        continue
                    # fpg DAG rule: never evaluate an edge that would
                    # close a cycle through already-accepted edges.
                    if neighbor in nodes and _reaches(adj, neighbor, current):
                        hop_log.append({
                            "round": round_n, "from": current,
                            "to": neighbor, "verdict": "skipped_cycle",
                        })
                        continue
                    pending_hops.append([current, neighbor, rel_type])

            if not pending_hops:
                continue

            ctx.log("Round " + str(round_n) + ": " + str(len(pending_hops)) + " hops")

            async def _make_hop_coro(
                from_svc: str, to_svc: str, rel_type: str,
            ) -> HopResult | None:
                hop_fault = node_fault.get(
                    from_svc, [all_faults[0][0], all_faults[0][1]],
                )
                result: AgentResult = await ctx.agent(
                    "Verify this propagation edge.",
                    scenario="verifier/hop",
                    atom_config={
                        "hop_context": {
                            "from_service": from_svc,
                            "to_service": to_svc,
                            "rel_type": rel_type,
                            "fault_kind": hop_fault[0],
                            "injection_target": hop_fault[1],
                            "all_faults": all_faults,
                            "fault_docs": fault_docs,
                            "is_infra": to_svc in infra_set,
                            "upstream_evidence": nodes.get(from_svc),
                            "dataset_profile": dataset_profile,
                        },
                        "duckdb_sql": {"data_dir": data_dir},
                        "hop_finalize": {"data_dir": data_dir},
                    },
                )
                return cast(HopResult, result) if isinstance(result, dict) else None

            coros: list[Awaitable[HopResult | None]] = [
                _make_hop_coro(item[0], item[1], item[2])
                for item in pending_hops
            ]
            results = await ctx.parallel(coros)

            for idx in range(len(pending_hops)):
                from_svc, to_svc, rel_type = pending_hops[idx]
                result = results[idx]
                verdict = result.get("verdict") if isinstance(result, dict) else None
                hop_log.append({
                    "round": round_n, "from": from_svc, "to": to_svc,
                    "verdict": verdict if verdict else "no-result",
                })
                ctx.log(
                    "  " + from_svc + " -> " + to_svc + ": "
                    + (verdict if verdict else "no-result")
                )
                if isinstance(result, dict) and verdict:
                    verdicts[from_svc + "__" + to_svc] = result

                if verdict != "confirmed":
                    continue
                # Re-check the cycle guard: edges accepted earlier in
                # this same round may have changed reachability.
                if _reaches(adj, to_svc, from_svc):
                    hop_log.append({
                        "round": round_n, "from": from_svc,
                        "to": to_svc, "verdict": "dropped_cycle",
                    })
                    continue

                assert isinstance(result, dict)
                if to_svc not in nodes:
                    nodes[to_svc] = _node_from_verdict(to_svc, result, window)
                    node_fault[to_svc] = node_fault.get(
                        from_svc, [all_faults[0][0], all_faults[0][1]],
                    )
                    if to_svc not in infra_set:
                        queue.append(to_svc)
                edges.append(_edge_dict(
                    from_svc, to_svc, rel_type, rel_mechanism,
                    str(result.get("claim", "")),
                ))
                adj.setdefault(from_svc, []).append(to_svc)
                in_deg[to_svc] = in_deg.get(to_svc, 0) + 1

    # -- Judge phase (optional) --
    judge_result: JudgeReviewResult | None = None
    if not skip_judge and verdicts:
        ctx.phase("judge")

        verdict_by_target: dict[str, dict[str, Any]] = {}
        for key, v in verdicts.items():
            from_svc, to_svc = key.split("__", 1)
            verdict_by_target[to_svc] = {
                "from": from_svc,
                "to": to_svc,
                "verdict": v.get("verdict", ""),
                "rationale": v.get("rationale", ""),
                "evidence": v.get("evidence", []),
            }
        rejected_verdicts = [
            v for v in verdict_by_target.values()
            if v.get("verdict") == "rejected" and v["to"] not in nodes
        ]

        judge_text: AgentResult = await ctx.agent(
            "Review the fault-propagation graph.",
            scenario="verifier/judge",
            atom_config={
                "duckdb_sql": {"data_dir": data_dir},
                "judge_context": {
                    "injections": injections,
                    "dataset_profile": dataset_profile,
                    "vanished_endpoints": cast(
                        dict[str, Any], args.get("vanished", {})
                    ),
                    "entry_services": cast(
                        list[str], args.get("entry_services", [])
                    ),
                    "confirmed": sorted(nodes),
                    "rejected_verdicts": rejected_verdicts,
                    "throughput": {},
                    "seeds": sorted(seeds),
                    "verdict_by_target": verdict_by_target,
                },
            },
        )
        if isinstance(judge_text, dict):
            judge_result = cast(JudgeReviewResult, judge_text)

        # Apply promotions: each must attach through a confirmed
        # upstream so promoted nodes never become spurious roots.
        for promo in (judge_result or {}).get("add", []):
            svc = promo.get("service", "")
            via = promo.get("via_service", "")
            if not svc or svc in nodes or svc in seeds:
                ctx.log(f"  judge add {svc!r}: skipped (missing/already present)")
                continue
            if via not in nodes:
                ctx.log(f"  judge add {svc!r}: skipped (via_service {via!r} not confirmed)")
                continue
            if _reaches(adj, svc, via):
                ctx.log(f"  judge add {svc!r}: skipped (would close a cycle)")
                continue
            rejected = verdicts.get(via + "__" + svc)
            evidence = list(rejected.get("evidence", [])) if rejected else []
            predicate = promo.get("predicate") or "service_unavailable"
            node: dict[str, Any] = {
                "kind": "event",
                "id": svc,
                "subject": f"svc:{svc}",
                "predicate": predicate,
                "time": window,
                "grounding": "observed" if evidence else "latent",
                "evidence": evidence,
                "annotation": "auto",
            }
            rel_type = next(
                (info[1] for info in graph.get(via, []) if info[0] == svc),
                "",
            )
            edge = _edge_dict(
                via, svc, rel_type, rel_mechanism,
                "judge cascade promotion: " + promo.get("rationale", ""),
            )
            nodes[svc] = node
            edges.append(edge)
            adj.setdefault(via, []).append(svc)
            in_deg[svc] = in_deg.get(svc, 0) + 1
            ctx.log(f"  judge add: {via} -> {svc} ({predicate})")

    # fpg rule: in-degree >= 2 requires combine; each confirmed edge is
    # an independently sufficient path, so the combination is OR.
    for svc, node in nodes.items():
        if in_deg.get(svc, 0) >= 2:
            node["combine"] = "OR"
        else:
            node.pop("combine", None)

    result_out: PropagationResult = {
        "nodes": [nodes[k] for k in sorted(nodes)],
        "edges": edges,
        "verdicts": verdicts,
        "hop_log": hop_log,
        "rounds": round_n,
    }
    if judge_result:
        result_out["judge"] = judge_result
    return result_out
