"""Fault propagation verification workflow (module mode), fpg-native.

BFS over the neighbor graph, parallel hop agents, optional judge phase.
The internal state IS the fpg graph: nodes are fpg EventNode dicts and
edges are fpg Edge dicts, built incrementally as hops confirm. The fpg
structural rules are enforced at construction time, not filtered
afterwards:

  - injection seeds are verified independently, but a seed-to-seed edge may
    still be accepted when a hop agent confirms that one injected fault
    propagated into another injected target;
  - an edge that would close a cycle is never evaluated;
  - EVERY edge goes through a hop agent — including edges between two
    already-confirmed services (topological adjacency alone is not
    causal evidence);
  - a judge promotion must name the upstream service it cascades
    through, so promoted nodes are connected, never spurious roots.

Input via ``ctx.args`` (built by ``prepare.CaseContext.to_workflow_args``):
    data_dir, graph, injections, infra_nodes, fault_docs,
    budget, out_dir, skip_propagate, skip_judge,
    window               -- fpg TimeInterval dict for this case
    rel_mechanism        -- {rel_type: fpg edge mechanism}
    existing_state       -- {nodes, edges, verdicts} for skip_propagate

Output: dict with nodes (list of fpg EventNode dicts), edges (list of
fpg Edge dicts), verdicts, hop_log, rounds, judge (if run).
"""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Any, Required, TypedDict, cast

from agentm.extensions.builtin.workflow import AgentResult, WorkflowContext

from .hop.hop_context import PriorVerdict, build_hop_prompt
from .judge.judge_context import build_judge_prompt
from .seed.seed_context import build_seed_prompt


class Injection(TypedDict, total=False):
    target: Required[str]
    chaos_type: Required[str]
    params: str
    node_id: str
    subject: str
    target_entity: str
    effect_target: str
    edge_source: str
    edge_target: str


HopLogEntry = TypedDict(
    "HopLogEntry",
    {"round": int, "from": str, "to": str, "verdict": str},
)


class SeedResult(TypedDict, total=False):
    verdict: Required[str]
    predicate: str | None
    rationale: str
    evidence: list[dict[str, Any]]
    claim: str


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


class JudgeReEval(TypedDict, total=False):
    service: Required[str]
    via_service: Required[str]
    context: str


class JudgeSeedReEval(TypedDict, total=False):
    seed: Required[str]
    context: str


class JudgeReviewResult(TypedDict, total=False):
    entry_explanation: str
    unexplained_entry_observations: list[str]
    add: list[JudgePromotion]
    re_evaluate: list[JudgeReEval]
    re_evaluate_seeds: list[JudgeSeedReEval]
    suggested_remove: list[str]
    rationale: str


class PropagationResult(TypedDict, total=False):
    nodes: Required[list[dict[str, Any]]]
    edges: Required[list[dict[str, Any]]]
    verdicts: Required[dict[str, HopResult]]
    hop_log: Required[list[HopLogEntry]]
    rounds: Required[int]
    judge: JudgeReviewResult
    judge_rounds: list[dict[str, Any]]
    unreachable_seeds: list[str]
    seed_verdicts: dict[str, SeedResult]
    confirmed_seeds: list[str]


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


def _injection_node_id(inj: Injection) -> str:
    return inj.get("node_id") or inj["target"]


def _injection_subject(inj: Injection) -> str:
    return inj.get("subject") or inj.get("target_entity") or f"svc:{inj['target']}"


def _injection_effect_target(inj: Injection) -> str:
    return inj.get("effect_target") or inj["target"]


def _is_link_injection(inj: Injection) -> bool:
    return _injection_subject(inj).startswith("link:")


def _fault_record(inj: Injection) -> list[str]:
    return [inj["chaos_type"], _injection_node_id(inj), inj.get("params", "")]


def _link_root_predicate(inj: Injection) -> str:
    chaos_type = inj.get("chaos_type", "").lower()
    if "partition" in chaos_type:
        return "network_partitioned"
    return "network_degraded"


def _node_from_seed(
    inj: Injection,
    verdict: SeedResult,
    window: dict[str, str],
) -> dict[str, Any]:
    """Build an fpg EventNode dict from a confirmed seed verdict."""
    predicate = verdict.get("predicate") or (
        _link_root_predicate(inj) if _is_link_injection(inj) else "other"
    )
    node: dict[str, Any] = {
        "kind": "event",
        "id": _injection_node_id(inj),
        "subject": _injection_subject(inj),
        "predicate": predicate,
        "time": window,
        "grounding": "observed",
        "evidence": list(verdict.get("evidence", [])),
        "annotation": "auto",
    }
    if predicate == "other":
        node["description"] = verdict.get("rationale", verdict.get("claim", "inconclusive"))
    return node


def _node_from_link_effect(
    inj: Injection,
    verdict: SeedResult,
    window: dict[str, str],
) -> dict[str, Any]:
    """Build the service-side symptom node for a confirmed link injection."""
    svc = _injection_effect_target(inj)
    predicate = verdict.get("predicate") or "flow_interrupted"
    node: dict[str, Any] = {
        "kind": "event",
        "id": svc,
        "subject": f"svc:{svc}",
        "predicate": predicate,
        "time": window,
        "grounding": "observed",
        "evidence": list(verdict.get("evidence", [])),
        "annotation": "auto",
    }
    if predicate == "other":
        node["description"] = verdict.get("rationale", verdict.get("claim", "inconclusive"))
    return node


def _node_from_verdict(
    svc: str,
    verdict: HopResult,
    window: dict[str, str],
) -> dict[str, Any]:
    """Build an fpg EventNode dict from a confirmed hop verdict."""
    evidence = list(verdict.get("evidence", []))
    relationship = verdict.get("relationship")
    if relationship:
        evidence.append(
            {
                "query": relationship["query"],
                "explanation": "call relationship with the confirmed upstream: "
                + relationship.get("explanation", "see query"),
            }
        )
    predicate = verdict.get("predicate") or "other"
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
    seeds = {_injection_node_id(inj) for inj in injections if inj.get("target")}

    nodes: dict[str, dict[str, Any]]  # svc -> fpg EventNode dict
    edges: list[dict[str, Any]]
    adj: dict[str, list[str]]  # accepted-edge adjacency, for cycle guard
    in_deg: dict[str, int]
    verdicts: dict[str, HopResult]  # "from__to" -> hop verdict
    hop_log: list[HopLogEntry]
    round_n: int
    node_fault: dict[str, list[str]]
    seed_verdicts: dict[str, SeedResult]
    confirmed_seed_ids: set[str]

    graph: dict[str, list[list[str]]] = args["graph"]
    infra_set = set(args.get("infra_nodes", []))
    data_dir: str = args["data_dir"]
    skip_judge: bool = args.get("skip_judge", False)
    judge_model_raw = args.get("judge_model")
    judge_model = (
        judge_model_raw.strip()
        if isinstance(judge_model_raw, str) and judge_model_raw.strip()
        else None
    )
    fault_docs: dict[str, str] = args.get("fault_docs", {})
    judge_result: JudgeReviewResult | None = None
    judge_rounds_log: list[dict[str, Any]] = []
    seed_recheck_handler: Any = None

    def _entry_services_from_graph() -> set[str]:
        callers: set[str] = set()
        callees: set[str] = set()
        for svc, neighbors in graph.items():
            for info in neighbors:
                if info[1] == "caller_to_callee":
                    callers.add(svc)
                    callees.add(info[0])
        explicit_entries = {
            svc for svc in callers | callees if svc in {"frontend", "ts-ui-dashboard"}
        }
        return explicit_entries or (callers - callees)

    entry_services = _entry_services_from_graph()

    all_faults = [
        _fault_record(inj)
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

    def _unreachable_seed_nodes() -> list[str]:
        fpg_adj: dict[str, set[str]] = {}
        for e in edges:
            fpg_adj.setdefault(e["src"], set()).add(e["dst"])

        unreachable: list[str] = []
        for seed_svc in sorted(confirmed_seed_ids):
            if seed_svc not in nodes:
                unreachable.append(seed_svc)
                continue
            visited: set[str] = set()
            queue = [seed_svc]
            while queue:
                cur = queue.pop()
                if cur in visited:
                    continue
                visited.add(cur)
                for nxt in fpg_adj.get(cur, set()):
                    queue.append(nxt)
            if not (visited & entry_services):
                unreachable.append(seed_svc)
        return unreachable

    if skip_propagate:
        existing = cast(dict[str, Any], args.get("existing_state", {}))
        nodes = {n["id"]: n for n in existing.get("nodes", [])}
        edges = list(existing.get("edges", []))
        verdicts = cast(dict[str, HopResult], existing.get("verdicts", {}))
        hop_log = cast(list[HopLogEntry], existing.get("hop_log", []))
        round_n = int(existing.get("rounds", 0))
        seed_verdicts = cast(dict[str, SeedResult], existing.get("seed_verdicts", {}))
        confirmed_seed_ids = set(existing.get("confirmed_seeds", []))
        if not confirmed_seed_ids and not seed_verdicts:
            confirmed_seed_ids = {seed for seed in seeds if seed in nodes}
        adj, in_deg = _rebuild_adjacency()
        node_fault = {
            _injection_node_id(inj): _fault_record(inj)
            for inj in injections
            if inj.get("target")
        }
        for inj in injections:
            if inj.get("target") and _is_link_injection(inj):
                node_fault[_injection_effect_target(inj)] = _fault_record(inj)
    else:
        # -- Phase 0: verify seeds ----------------------------------------
        ctx.phase("seed")
        node_fault = {
            _injection_node_id(inj): _fault_record(inj)
            for inj in injections
            if inj.get("target")
        }
        nodes = {}
        edges = []
        adj = {}
        in_deg = {}
        verdicts = {}
        hop_log = []
        round_n = 0
        seed_verdicts = {}
        confirmed_seed_ids = set()

        async def _verify_seed(
            inj: Injection,
            judge_context: str = "",
        ) -> tuple[Injection, SeedResult | None]:
            target = _injection_node_id(inj) if _is_link_injection(inj) else inj["target"]
            fault_kind = inj.get("chaos_type", "unknown")
            prompt = build_seed_prompt(
                target=target,
                fault_kind=fault_kind,
                params=inj.get("params", ""),
                fault_doc=fault_docs.get(fault_kind, ""),
                judge_context=judge_context,
            )
            result: AgentResult = await ctx.agent(
                prompt,
                scenario="verifier/seed",
                atom_config={
                    "duckdb_sql": {"data_dir": data_dir},
                    "seed_finalize": {"data_dir": data_dir},
                },
            )
            return inj, cast(SeedResult, result) if isinstance(result, dict) else None

        seed_coros: list[Awaitable[tuple[Injection, SeedResult | None]]] = [
            _verify_seed(inj) for inj in injections if inj.get("target")
        ]
        seed_results = await ctx.parallel(seed_coros)
        propagation_roots: list[str] = []

        def _accept_seed_node(inj: Injection, seed_verdict: SeedResult) -> str:
            root_id = _injection_node_id(inj)
            nodes[root_id] = _node_from_seed(inj, seed_verdict, window)
            if not _is_link_injection(inj):
                propagation_roots.append(root_id)
                return root_id

            effect_target = _injection_effect_target(inj)
            if effect_target not in nodes:
                nodes[effect_target] = _node_from_link_effect(
                    inj,
                    seed_verdict,
                    window,
                )
            node_fault[effect_target] = _fault_record(inj)
            edges.append(
                _edge_dict(
                    root_id,
                    effect_target,
                    "link_to_service",
                    rel_mechanism,
                    "link fault manifests on the rule-bearing service side",
                )
            )
            adj.setdefault(root_id, []).append(effect_target)
            in_deg[effect_target] = in_deg.get(effect_target, 0) + 1
            propagation_roots.append(effect_target)
            return root_id

        async def _apply_seed_rechecks(
            seed_rechecks: list[dict[str, Any]],
        ) -> tuple[list[dict[str, Any]], bool]:
            inj_by_seed = {
                _injection_node_id(inj): inj
                for inj in injections
                if inj.get("target")
            }
            valid_rechecks = [
                r for r in seed_rechecks
                if isinstance(r.get("seed"), str)
                and r["seed"] in inj_by_seed
                and r["seed"] not in confirmed_seed_ids
            ]
            if not valid_rechecks:
                return [], False

            ctx.phase("seed-re-evaluate")
            ctx.log(f"Re-evaluating {len(valid_rechecks)} seeds")
            seed_coros: list[Awaitable[tuple[Injection, SeedResult | None]]] = [
                _verify_seed(inj_by_seed[r["seed"]], str(r.get("context", "")))
                for r in valid_rechecks
            ]
            seed_re_results = await ctx.parallel(seed_coros)
            recheck_log: list[dict[str, Any]] = []
            new_confirmed = False
            for inj, seed_verdict in seed_re_results:
                root_id = _injection_node_id(inj)
                if seed_verdict:
                    seed_verdicts[root_id] = seed_verdict
                verdict = seed_verdict.get("verdict") if seed_verdict else "no result"
                recheck_log.append(
                    {
                        "seed": root_id,
                        "verdict": verdict,
                        "rationale": seed_verdict.get("rationale", "")
                        if seed_verdict
                        else "",
                    }
                )
                if seed_verdict and verdict == "confirmed":
                    accepted = _accept_seed_node(inj, seed_verdict)
                    confirmed_seed_ids.add(accepted)
                    new_confirmed = True
                    ctx.log(f"  seed re-eval {accepted}: confirmed")
                else:
                    ctx.log(f"  seed re-eval {root_id}: {verdict}")
            return recheck_log, new_confirmed

        seed_recheck_handler = _apply_seed_rechecks

        for inj, seed_verdict in seed_results:
            root_id = _injection_node_id(inj)
            if seed_verdict and seed_verdict.get("verdict") == "confirmed":
                root_id = _accept_seed_node(inj, seed_verdict)
                confirmed_seed_ids.add(root_id)
                ctx.log(
                    f"seed {root_id}: confirmed ({seed_verdict.get('predicate')})"
                )
            elif seed_verdict and seed_verdict.get("verdict") == "inconclusive":
                ctx.log(f"seed {root_id}: inconclusive — keeping for judge review")
            else:
                v = seed_verdict.get("verdict", "no result") if seed_verdict else "no result"
                ctx.log(f"seed {root_id}: {v} — skipping")

        for inj, seed_verdict in seed_results:
            if seed_verdict:
                seed_verdicts[_injection_node_id(inj)] = seed_verdict

        if not nodes:
            ctx.log("no seeds confirmed — running judge audit before propagation")
            if not skip_judge:
                ctx.phase("judge")
                judge_prompt = build_judge_prompt(
                    injections=injections,
                    confirmed=[],
                    confirmed_edges=[],
                    entry_services=sorted(entry_services),
                    unreachable_seeds=sorted(seeds),
                    seeds=seeds,
                    seed_verdicts=seed_verdicts,
                    verdict_by_target={},
                    inconclusive_verdicts=[],
                    rejected_verdicts=[],
                )
                pre_judge_text: AgentResult = await ctx.agent(
                    judge_prompt,
                    scenario="verifier/judge",
                    model=judge_model,
                    atom_config={
                        "duckdb_sql": {"data_dir": data_dir},
                    },
                )
                if not isinstance(pre_judge_text, dict):
                    ctx.log("  judge returned no structured review; retrying once")
                    pre_judge_text = await ctx.agent(
                        judge_prompt
                        + "\n\nIMPORTANT: Your previous response was not a structured "
                        "submit_judge_review tool result. Call submit_judge_review now; "
                        "do not answer in prose.",
                        scenario="verifier/judge",
                        model=judge_model,
                        atom_config={
                            "duckdb_sql": {"data_dir": data_dir},
                        },
                    )
                if isinstance(pre_judge_text, dict):
                    judge_result = cast(JudgeReviewResult, pre_judge_text)
                    if judge_result.get("entry_explanation"):
                        ctx.log(
                            "  judge entry audit: "
                            + judge_result["entry_explanation"][:500]
                        )
                    if judge_result.get("unexplained_entry_observations"):
                        ctx.log(
                            "  judge unexplained entry observations: "
                            + "; ".join(judge_result["unexplained_entry_observations"])
                        )
                pre_seed_recheck_log, new_seed_confirmed = await _apply_seed_rechecks(
                    cast(
                        list[dict[str, Any]],
                        (judge_result or {}).get("re_evaluate_seeds", []),
                    )
                )
                judge_rounds_log.append(
                    {
                        "round": 1,
                        "judge_decision": dict(judge_result) if judge_result else {},
                        "re_eval_results": [],
                        "seed_re_eval_results": pre_seed_recheck_log,
                        "new_confirmed": new_seed_confirmed,
                    }
                )

            if not nodes:
                ctx.log("no seeds confirmed after judge audit — aborting propagation")
                early_result_out: PropagationResult = {
                    "nodes": [],
                    "edges": [],
                    "verdicts": {},
                    "hop_log": [],
                    "rounds": 0,
                    "seed_verdicts": seed_verdicts,
                    "confirmed_seeds": [],
                }
                if judge_result:
                    early_result_out["judge"] = judge_result
                if judge_rounds_log:
                    early_result_out["judge_rounds"] = judge_rounds_log
                return early_result_out

        # -- Phase 1: propagate -------------------------------------------
        ctx.phase("propagate")

        queue = list(dict.fromkeys(propagation_roots or list(nodes)))
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

                    # fpg DAG rule: never evaluate an edge that would
                    # close a cycle through already-accepted edges.
                    if neighbor in nodes and _reaches(adj, neighbor, current):
                        hop_log.append(
                            {
                                "round": round_n,
                                "from": current,
                                "to": neighbor,
                                "verdict": "skipped_cycle",
                            }
                        )
                        continue
                    pending_hops.append([current, neighbor, rel_type])

            if not pending_hops:
                continue

            ctx.log("Round " + str(round_n) + ": " + str(len(pending_hops)) + " hops")

            async def _make_hop_coro(
                from_svc: str,
                to_svc: str,
                rel_type: str,
            ) -> HopResult | None:
                hop_fault = node_fault.get(
                    from_svc,
                    [all_faults[0][0], all_faults[0][1]],
                )
                edge_fault_kind = hop_fault[0]
                edge_fault_docs: dict[str, str] = {}
                if edge_fault_kind in fault_docs:
                    edge_fault_docs[edge_fault_kind] = fault_docs[edge_fault_kind]
                prompt = build_hop_prompt(
                    from_service=from_svc,
                    to_service=to_svc,
                    rel_type=rel_type,
                    fault_kind=edge_fault_kind,
                    all_faults=[
                        (f[0], f[1], f[2] if len(f) > 2 else "")
                        for f in all_faults
                        if len(f) >= 2
                    ],
                    fault_docs=edge_fault_docs,
                    is_infra=to_svc in infra_set,
                    upstream_evidence=nodes.get(from_svc),
                )
                result: AgentResult = await ctx.agent(
                    prompt,
                    scenario="verifier/hop",
                    atom_config={
                        "duckdb_sql": {"data_dir": data_dir},
                        "hop_finalize": {"data_dir": data_dir},
                    },
                )
                return cast(HopResult, result) if isinstance(result, dict) else None

            coros: list[Awaitable[HopResult | None]] = [
                _make_hop_coro(item[0], item[1], item[2]) for item in pending_hops
            ]
            results = await ctx.parallel(coros)

            for idx in range(len(pending_hops)):
                from_svc, to_svc, rel_type = pending_hops[idx]
                result = results[idx]
                verdict = result.get("verdict") if isinstance(result, dict) else None
                hop_log.append(
                    {
                        "round": round_n,
                        "from": from_svc,
                        "to": to_svc,
                        "verdict": verdict if verdict else "no-result",
                    }
                )
                ctx.log(
                    "  "
                    + from_svc
                    + " -> "
                    + to_svc
                    + ": "
                    + (verdict if verdict else "no-result")
                )
                if isinstance(result, dict) and verdict:
                    verdicts[from_svc + "__" + to_svc] = result

                if verdict != "confirmed":
                    continue
                # Re-check the cycle guard: edges accepted earlier in
                # this same round may have changed reachability.
                if _reaches(adj, to_svc, from_svc):
                    hop_log.append(
                        {
                            "round": round_n,
                            "from": from_svc,
                            "to": to_svc,
                            "verdict": "dropped_cycle",
                        }
                    )
                    continue

                assert isinstance(result, dict)
                if to_svc not in nodes:
                    nodes[to_svc] = _node_from_verdict(to_svc, result, window)
                    node_fault[to_svc] = node_fault.get(
                        from_svc,
                        [all_faults[0][0], all_faults[0][1]],
                    )
                    if to_svc not in infra_set and to_svc not in entry_services:
                        queue.append(to_svc)
                edges.append(
                    _edge_dict(
                        from_svc,
                        to_svc,
                        rel_type,
                        rel_mechanism,
                        str(result.get("claim", "")),
                    )
                )
                adj.setdefault(from_svc, []).append(to_svc)
                in_deg[to_svc] = in_deg.get(to_svc, 0) + 1

            if not _unreachable_seed_nodes():
                ctx.log("all confirmed seeds reach entry services; stopping propagation")
                queue = []
                break

    # -- Judge + re-evaluation loop --
    judge_needed = bool(nodes)
    if not judge_needed:
        ctx.log("skipping judge: no confirmed or candidate nodes to audit")
    if not skip_judge and judge_needed:
        max_judge_rounds = 3
        for judge_round in range(max_judge_rounds):
            ctx.phase("judge" if judge_round == 0 else f"judge-r{judge_round + 1}")

            verdict_by_target: dict[str, dict[str, Any]] = {}
            for key, vd in verdicts.items():
                from_svc, to_svc = key.split("__", 1)
                vdict = cast(dict[str, Any], vd)
                verdict_by_target[to_svc] = {
                    "from": from_svc,
                    "to": to_svc,
                    "verdict": vdict.get("verdict", ""),
                    "rationale": vdict.get("rationale", ""),
                    "evidence": vdict.get("evidence", []),
                }
            inconclusive_verdicts = [
                v
                for v in verdict_by_target.values()
                if v.get("verdict") == "inconclusive" and v["to"] not in nodes
            ]
            rejected_verdicts = [
                v
                for v in verdict_by_target.values()
                if v.get("verdict") == "rejected" and v["to"] not in nodes
            ]

            judge_prompt = build_judge_prompt(
                injections=injections,
                confirmed=sorted(nodes),
                confirmed_edges=edges,
                entry_services=sorted(entry_services),
                unreachable_seeds=_unreachable_seed_nodes(),
                seeds=seeds,
                seed_verdicts=seed_verdicts,
                verdict_by_target=verdict_by_target,
                inconclusive_verdicts=inconclusive_verdicts,
                rejected_verdicts=rejected_verdicts,
            )
            round_judge_text: AgentResult = await ctx.agent(
                judge_prompt,
                scenario="verifier/judge",
                model=judge_model,
                atom_config={
                    "duckdb_sql": {"data_dir": data_dir},
                },
            )
            if not isinstance(round_judge_text, dict):
                ctx.log("  judge returned no structured review; retrying once")
                round_judge_text = await ctx.agent(
                    judge_prompt
                    + "\n\nIMPORTANT: Your previous response was not a structured "
                    "submit_judge_review tool result. Call submit_judge_review now; "
                    "do not answer in prose.",
                    scenario="verifier/judge",
                    model=judge_model,
                    atom_config={
                        "duckdb_sql": {"data_dir": data_dir},
                    },
                )
            if isinstance(round_judge_text, dict):
                judge_result = cast(JudgeReviewResult, round_judge_text)
                if judge_result.get("entry_explanation"):
                    ctx.log("  judge entry audit: " + judge_result["entry_explanation"][:500])
                if judge_result.get("unexplained_entry_observations"):
                    ctx.log(
                        "  judge unexplained entry observations: "
                        + "; ".join(judge_result["unexplained_entry_observations"])
                    )

            round_seed_recheck_log: list[dict[str, Any]] = []
            seed_recheck_added = False
            seed_rechecks = cast(
                list[dict[str, Any]],
                (judge_result or {}).get("re_evaluate_seeds", []),
            )
            if seed_recheck_handler and seed_rechecks:
                round_seed_recheck_log, seed_recheck_added = await seed_recheck_handler(
                    seed_rechecks
                )

            # Apply direct promotions (judge has enough evidence to decide)
            direct_promotion_added = seed_recheck_added
            for promo in (judge_result or {}).get("add", []):
                svc = promo.get("service", "")
                via = promo.get("via_service", "")
                if not svc:
                    ctx.log("  judge add: skipped (missing service)")
                    continue
                if via not in nodes:
                    ctx.log(
                        f"  judge add {svc!r}: skipped (via_service {via!r} not confirmed)"
                    )
                    continue
                if _reaches(adj, svc, via):
                    ctx.log(f"  judge add {svc!r}: skipped (would close a cycle)")
                    continue
                prior = verdicts.get(via + "__" + svc)
                rel_type = next(
                    (info[1] for info in graph.get(via, []) if info[0] == svc),
                    "",
                )
                edge = _edge_dict(
                    via,
                    svc,
                    rel_type,
                    rel_mechanism,
                    "judge cascade promotion: " + promo.get("rationale", ""),
                )
                if svc in nodes:
                    if svc in adj.get(via, []):
                        ctx.log(f"  judge add {svc!r}: skipped (edge already present)")
                        continue
                    edges.append(edge)
                    adj.setdefault(via, []).append(svc)
                    in_deg[svc] = in_deg.get(svc, 0) + 1
                    direct_promotion_added = True
                    ctx.log(f"  judge add edge: {via} -> {svc}")
                    continue
                evidence = list(prior.get("evidence", [])) if prior else []
                predicate = promo.get("predicate") or "process_killed"
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
                nodes[svc] = node
                edges.append(edge)
                adj.setdefault(via, []).append(svc)
                in_deg[svc] = in_deg.get(svc, 0) + 1
                direct_promotion_added = True
                ctx.log(f"  judge add: {via} -> {svc} ({predicate})")

            # Re-evaluate edges the judge flagged for re-investigation
            re_eval_raw: list[dict[str, Any]] = cast(
                list[dict[str, Any]],
                (judge_result or {}).get("re_evaluate", []),
            )
            re_eval = [
                r
                for r in re_eval_raw
                if r.get("service")
                and r.get("via_service")
                and r["via_service"] in nodes
            ]
            if not re_eval:
                judge_rounds_log.append(
                    {
                        "round": judge_round + 1,
                        "judge_decision": dict(judge_result) if judge_result else {},
                        "re_eval_results": [],
                        "seed_re_eval_results": round_seed_recheck_log,
                        "new_confirmed": direct_promotion_added,
                    }
                )
                if not direct_promotion_added or not _unreachable_seed_nodes():
                    break
                continue

            ctx.phase(f"re-evaluate-r{judge_round + 1}")
            ctx.log(f"Re-evaluating {len(re_eval)} edges")

            async def _make_reeval_coro(
                re: dict[str, Any],
            ) -> HopResult | None:
                svc = re["service"]
                via = re.get("via_service", "")
                edge_key = via + "__" + svc
                prior_v: dict[str, Any] = verdicts.get(edge_key, {})  # type: ignore[assignment]
                hop_fault = node_fault.get(
                    via,
                    [all_faults[0][0], all_faults[0][1]],
                )
                edge_fault_kind = hop_fault[0]
                edge_fault_doc: dict[str, str] = {}
                if edge_fault_kind in fault_docs:
                    edge_fault_doc[edge_fault_kind] = fault_docs[edge_fault_kind]

                rel_type = next(
                    (info[1] for info in graph.get(via, []) if info[0] == svc),
                    "callee_to_caller",
                )
                prompt = build_hop_prompt(
                    from_service=via,
                    to_service=svc,
                    rel_type=rel_type,
                    fault_kind=edge_fault_kind,
                    all_faults=[
                        (f[0], f[1], f[2] if len(f) > 2 else "")
                        for f in all_faults
                        if len(f) >= 2
                    ],
                    fault_docs=edge_fault_doc,
                    is_infra=svc in infra_set,
                    upstream_evidence=nodes.get(via),
                    judge_context=re.get("context", ""),
                    prior_verdict=PriorVerdict(
                        verdict=prior_v.get("verdict", ""),
                        rationale=prior_v.get("rationale", ""),
                    ),
                )
                result: AgentResult = await ctx.agent(
                    prompt,
                    scenario="verifier/hop",
                    atom_config={
                        "duckdb_sql": {"data_dir": data_dir},
                        "hop_finalize": {"data_dir": data_dir},
                    },
                )
                return cast(HopResult, result) if isinstance(result, dict) else None

            re_coros: list[Awaitable[HopResult | None]] = [
                _make_reeval_coro(r) for r in re_eval
            ]
            re_results = await ctx.parallel(re_coros)

            any_new_confirmed = direct_promotion_added
            for idx_r in range(len(re_eval)):
                re_item = re_eval[idx_r]
                svc = re_item["service"]
                via = re_item.get("via_service", "")
                result = re_results[idx_r]
                verdict = result.get("verdict") if isinstance(result, dict) else None
                edge_key = via + "__" + svc
                hop_log.append(
                    {
                        "round": round_n + judge_round + 1,
                        "from": via,
                        "to": svc,
                        "verdict": (verdict or "no-result") + "(re-eval)",
                    }
                )
                ctx.log(f"  re-eval {via} -> {svc}: {verdict or 'no-result'}")
                if isinstance(result, dict) and verdict:
                    verdicts[edge_key] = result

                # confirmed → accept; inconclusive on re-eval → judge's
                # cascade determination stands, promote it.
                should_add = verdict == "confirmed" or (
                    verdict == "inconclusive" and via in nodes
                )
                if not should_add:
                    continue
                if _reaches(adj, svc, via):
                    continue

                if svc not in nodes:
                    if verdict == "confirmed" and isinstance(result, dict):
                        nodes[svc] = _node_from_verdict(svc, result, window)
                    else:
                        # Judge-promoted: inconclusive re-eval, use judge's
                        # predicate or default to flow_interrupted
                        evidence = (
                            list(result.get("evidence", []))
                            if isinstance(result, dict)
                            else []
                        )
                        nodes[svc] = {
                            "kind": "event",
                            "id": svc,
                            "subject": f"svc:{svc}",
                            "predicate": "flow_interrupted",
                            "time": window,
                            "grounding": "observed" if evidence else "latent",
                            "evidence": evidence,
                            "annotation": "auto",
                        }
                    node_fault[svc] = node_fault.get(
                        via,
                        [all_faults[0][0], all_faults[0][1]],
                    )
                    any_new_confirmed = True
                rel_type = next(
                    (info[1] for info in graph.get(via, []) if info[0] == svc),
                    "",
                )
                if svc not in adj.get(via, []):
                    edges.append(
                        _edge_dict(
                            via,
                            svc,
                            rel_type,
                            rel_mechanism,
                            re_item.get("context", "")[:200]
                            if verdict == "inconclusive"
                            else str(
                                result.get("claim", "")
                                if isinstance(result, dict)
                                else ""
                            ),
                        )
                    )
                    adj.setdefault(via, []).append(svc)
                    in_deg[svc] = in_deg.get(svc, 0) + 1
                    any_new_confirmed = True

            # Log this judge round
            re_eval_log: list[dict[str, Any]] = []
            for i in range(len(re_eval)):
                r = re_results[i]
                re_eval_log.append(
                    {
                        "service": re_eval[i]["service"],
                        "via_service": re_eval[i].get("via_service", ""),
                        "verdict": r.get("verdict") if isinstance(r, dict) else None,
                        "rationale": r.get("rationale", "")
                        if isinstance(r, dict)
                        else "",
                    }
                )
            judge_rounds_log.append(
                {
                    "round": judge_round + 1,
                    "judge_decision": dict(judge_result) if judge_result else {},
                    "re_eval_results": re_eval_log,
                    "seed_re_eval_results": round_seed_recheck_log,
                    "new_confirmed": any_new_confirmed,
                }
            )

            if not any_new_confirmed:
                break

    # fpg rule: in-degree >= 2 requires combine; each confirmed edge is
    # an independently sufficient path, so the combination is OR.
    for svc, node in nodes.items():
        if in_deg.get(svc, 0) >= 2:
            node["combine"] = "OR"
        else:
            node.pop("combine", None)

    # -- Reachability check: every confirmed seed should reach an entry service --
    ctx.phase("validate")
    unreachable = _unreachable_seed_nodes()
    for seed_svc in sorted(seeds):
        if seed_svc not in confirmed_seed_ids:
            ctx.log(f"⚠ seed {seed_svc}: not confirmed")
            continue
        if seed_svc in unreachable:
            ctx.log(
                f"⚠ seed {seed_svc}: no path to entry services "
                f"{sorted(entry_services)} in fpg"
            )

    if not unreachable:
        ctx.log("✓ all confirmed seeds reach entry services")

    result_out: PropagationResult = {
        "nodes": [nodes[k] for k in sorted(nodes)],
        "edges": edges,
        "verdicts": verdicts,
        "hop_log": hop_log,
        "rounds": round_n,
        "seed_verdicts": seed_verdicts,
        "confirmed_seeds": sorted(confirmed_seed_ids),
    }
    if judge_result:
        result_out["judge"] = judge_result
    if judge_rounds_log:
        result_out["judge_rounds"] = judge_rounds_log
    if unreachable:
        result_out["unreachable_seeds"] = unreachable
    return result_out
