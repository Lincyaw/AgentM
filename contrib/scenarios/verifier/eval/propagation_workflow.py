"""Fault propagation verification workflow (module mode).

BFS over the neighbor graph, parallel hop agents, optional judge phase.
All prompt/domain logic is in the agent units (verifier/hop and
verifier/judge scenarios with their context atoms). The workflow passes
structured data via atom_config.

Input via ``ctx.args``:
    data_dir, graph, injections, infra_nodes, node_map,
    target_evidence, fault_docs, budget, out_dir, skip_judge

Output: dict with confirmed_nodes, edges, node_evidence, hop_log,
rounds, judge (if run).
"""
from __future__ import annotations

from collections.abc import Awaitable
from typing import Required, TypedDict, cast

from agentm.extensions.builtin.workflow import AgentResult, WorkflowContext


class Injection(TypedDict, total=False):
    target: Required[str]
    chaos_type: Required[str]
    params: str


class TargetEvidence(TypedDict, total=False):
    normal_avg_ms: float
    abnormal_avg_ms: float
    ratio: float


EvidenceEntry = TypedDict(
    "EvidenceEntry",
    {
        "source": str,
        "from": str,
        "rationale": str,
        "relationship_sql": str,
        "claim": str,
        "symptom_evidence": list[dict[str, str]],
        "normal_avg_ms": float,
        "abnormal_avg_ms": float,
        "ratio": float,
    },
    total=False,
)


HopLogEntry = TypedDict(
    "HopLogEntry",
    {"round": int, "from": str, "to": str, "verdict": str},
)


class HopResult(TypedDict, total=False):
    verdict: Required[str]
    rationale: str
    relationship_sql: str
    claim: str
    symptom_evidence: list[dict[str, str]]


class JudgeReviewResult(TypedDict, total=False):
    remove: list[str]
    add: list[str]
    rationale: str


class ExistingPropagation(TypedDict, total=False):
    confirmed_nodes: list[str]
    node_evidence: dict[str, EvidenceEntry]
    edges: list[list[str]]
    hop_log: list[HopLogEntry]
    rounds: int


JudgeTargetVerdict = TypedDict(
    "JudgeTargetVerdict",
    {
        "from": str,
        "to": str,
        "verdict": str,
        "rationale": str,
        "symptom_evidence": list[dict[str, str]],
    },
    total=False,
)


class PropagationResult(TypedDict, total=False):
    confirmed_nodes: Required[list[str]]
    edges: Required[list[list[str]]]
    node_evidence: Required[dict[str, EvidenceEntry]]
    hop_log: Required[list[HopLogEntry]]
    rounds: Required[int]
    judge: JudgeReviewResult


async def run(ctx: WorkflowContext) -> PropagationResult:
    args = ctx.args
    skip_propagate: bool = args.get("skip_propagate", False)
    injections = cast(list[Injection], args["injections"])

    confirmed: set[str]
    node_evidence: dict[str, EvidenceEntry]
    edges: list[list[str]]
    hop_log: list[HopLogEntry]
    round_n: int
    node_fault: dict[str, list[str]]

    if skip_propagate:
        existing = cast(ExistingPropagation, args.get("existing_propagation", {}))
        confirmed = set(existing.get("confirmed_nodes", []))
        node_evidence = existing.get("node_evidence", {})
        edges = existing.get("edges", [])
        hop_log = existing.get("hop_log", [])
        round_n = existing.get("rounds", 0)
        node_fault = {}
        for inj in injections:
            target: str = inj["target"]
            if target:
                node_fault[target] = [inj["chaos_type"], target]
    else:
        ctx.phase("propagate")
        confirmed = set()
        node_evidence = {}
        node_fault = {}
        edges = []
        hop_log = []

    graph: dict[str, list[list[str]]] = args["graph"]
    infra_nodes_list: list[str] = args.get("infra_nodes", [])
    infra_set = set(infra_nodes_list)
    node_map = cast(dict[str, str], args.get("node_map", {}))
    data_dir: str = args["data_dir"]
    skip_judge: bool = args.get("skip_judge", False)
    fault_docs: dict[str, str] = args.get("fault_docs", {})

    all_faults = [
        [inj["chaos_type"], inj["target"], inj.get("params", "")]
        for inj in injections
        if inj.get("target")
    ]

    if not skip_propagate:
        for inj in injections:
            target = inj["target"]
            if target:
                confirmed.add(target)
                ev = cast(
                    TargetEvidence,
                    args.get("target_evidence", {}).get(target, {}),
                )
                node_evidence[target] = cast(
                    EvidenceEntry,
                    {"source": "injection_target", **ev},
                )
                node_fault[target] = [inj["chaos_type"], target]

        queue = list(confirmed)
        checked_edges: set[str] = set()
        round_n = 0

        while queue:
            round_n += 1
            batch = list(queue)
            queue = []

            pending_hops: list[list[str]] = []
            sql_edges: list[list[str]] = []

            for current in batch:
                for neighbor_info in graph.get(current, []):
                    neighbor = neighbor_info[0]
                    rel_type = neighbor_info[1]
                    edge_key = current + "__" + neighbor
                    if edge_key in checked_edges:
                        continue
                    checked_edges.add(edge_key)

                    if neighbor in confirmed:
                        sql_edges.append([current, neighbor])
                    else:
                        pending_hops.append([current, neighbor, rel_type])

            for pair in sql_edges:
                edges.append(pair)
                hop_log.append({
                    "round": round_n,
                    "from": pair[0],
                    "to": pair[1],
                    "verdict": "edge_sql",
                })
                ctx.log("  edge (SQL): " + pair[0] + " -> " + pair[1])

            if node_map and pending_hops:
                pending_hops = [
                    h for h in pending_hops if h[1] not in confirmed
                ]

            if not pending_hops:
                continue

            ctx.log("Round " + str(round_n) + ": " + str(len(pending_hops)) + " hops")

            async def _make_hop_coro(
                from_svc: str, to_svc: str, rel_type: str,
            ) -> HopResult | None:
                hop_fault = node_fault.get(
                    from_svc, [all_faults[0][0], all_faults[0][1]],
                )
                is_infra = to_svc in infra_set
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
                            "is_infra": is_infra,
                            "upstream_evidence": node_evidence.get(from_svc),
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
                from_svc = pending_hops[idx][0]
                to_svc = pending_hops[idx][1]
                result = results[idx]
                verdict = result.get("verdict") if isinstance(result, dict) else None
                hop_log.append({
                    "round": round_n,
                    "from": from_svc,
                    "to": to_svc,
                    "verdict": verdict if verdict else "no-result",
                })
                ctx.log(
                    "  " + from_svc + " -> " + to_svc + ": "
                    + (verdict if verdict else "no-result")
                )

                if isinstance(result, dict) and verdict:
                    node_evidence[to_svc] = cast(
                        EvidenceEntry,
                        {"source": "hop_agent", "from": from_svc, **result},
                    )

                if verdict == "confirmed" and to_svc not in confirmed:
                    confirmed.add(to_svc)
                    if from_svc in node_fault:
                        node_fault[to_svc] = node_fault[from_svc]
                    if to_svc not in infra_set:
                        queue.append(to_svc)
                    edges.append([from_svc, to_svc])

    propagation_result: PropagationResult = {
        "confirmed_nodes": sorted(confirmed),
        "edges": edges,
        "node_evidence": node_evidence,
        "hop_log": hop_log,
        "rounds": round_n,
    }

    # -- Judge phase (optional) --
    if not skip_judge and len(confirmed) > len(injections):
        ctx.phase("judge")

        seeds = {inj["target"] for inj in injections}

        verdict_by_target: dict[str, JudgeTargetVerdict] = {}
        for entry in hop_log:
            to_svc = entry.get("to", "")
            if not to_svc or entry.get("verdict") == "edge_sql":
                continue
            target_evidence = node_evidence.get(to_svc)
            rationale = (
                str(target_evidence.get("rationale", ""))
                if target_evidence is not None
                else ""
            )
            symptom_evidence = (
                target_evidence.get("symptom_evidence", [])
                if target_evidence is not None
                else []
            )
            verdict_by_target[to_svc] = {
                "from": entry.get("from", ""),
                "to": to_svc,
                "verdict": entry.get("verdict", ""),
                "rationale": rationale,
                "symptom_evidence": cast(list[dict[str, str]], symptom_evidence),
            }

        rejected_verdicts = [
            v for v in verdict_by_target.values()
            if v.get("verdict") == "rejected" and v["to"] not in confirmed
        ]

        judge_text: AgentResult = await ctx.agent(
            "Review the fault-propagation graph.",
            scenario="verifier/judge",
            atom_config={
                "duckdb_sql": {"data_dir": data_dir},
                "judge_context": {
                    "injections": injections,
                    "confirmed": sorted(confirmed),
                    "rejected_verdicts": rejected_verdicts,
                    "throughput": {},
                    "seeds": sorted(seeds),
                    "verdict_by_target": verdict_by_target,
                },
            },
        )
        judge_result = cast(JudgeReviewResult, judge_text) if isinstance(judge_text, dict) else None

        if judge_result:
            propagation_result["judge"] = judge_result

    return propagation_result
