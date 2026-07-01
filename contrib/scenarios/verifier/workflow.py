"""Fault propagation verification workflow (module mode), fpg-native.

This module is the thin orchestration core. It wires the existing verifier flow over a
``GraphState``:

  - seed_phase   — verify each injection (``discovery.verify_seed``);
                   confirmed seeds become propagation roots.
  - propagate    — scan structural and anomaly-informed candidate edges
                   (``discovery.verify_hop``); cycles are never evaluated.
  - audit_loop   — run the audit map/reduce (``audit_map``), apply its
                   rework + edge drops, repeat until it accepts or the
                   case is exhausted.

All graph state and pure operations live in ``state.GraphState``; the
agent-calling lives in ``discovery`` and ``audit_map``. The reducer owns
the accept decision — there is no harness force-accept.

Input via ``ctx.args`` (built by ``prepare.CaseContext.to_workflow_args``):
    data_dir, graph, injections, infra_nodes, fault_docs,
    gate_retries, agent_retries, max_audit_rounds, judge_model,
    max_parallel_tasks, propagation_window, skip_judge, window, rel_mechanism.

Output: a ``PropagationResult`` dict (see ``state.GraphState.to_result``).
"""

from __future__ import annotations

import asyncio
import heapq
from collections.abc import Awaitable, Sequence
from typing import Any

from agentm.extensions.builtin.workflow import WorkflowContext

from . import audit_map, discovery
from .hop.hop_context import PriorVerdict
from .lib.candidates import (
    anomaly_candidates,
    graph_rel_type,
    structural_candidates,
)
from .lib.final_checks import frontend_like
from .lib.fpg import injection_node_id
from .lib.obligations import (
    obligation_payload,
    obligations_from_report,
    rework_requests_for_obligations,
)
from .lib.scheduler import (
    build_priority_context,
    build_propagation_needs,
    candidate_addresses_needs,
    edge_priority,
)
from .lib.schema import (
    AuditOutcome,
    HopRecheckRequest,
    HopResult,
    Injection,
    CandidateEdge,
    FinalCheckReport,
    PropagationResult,
    ReworkRequest,
    SeedCoverageStatus,
    SeedRecheckRequest,
    SeedResult,
)
from .state import Case, GraphState, _reaches


async def run(ctx: WorkflowContext) -> PropagationResult:
    case = Case.from_args(ctx.args)
    state = GraphState(case, ctx.log)

    await seed_phase(ctx, case, state)
    await seed_recheck_unconfirmed_phase(ctx, case, state)
    ctx.phase("propagate")
    await propagate(ctx, case, state, state.propagation_roots or list(state.nodes))

    audit_result: AuditOutcome | None = None
    if case.skip_judge:
        ctx.log("audit skipped by request")
    else:
        audit_result = await audit_loop(ctx, case, state)
    await final_check_loop(ctx, case, state)

    ctx.phase("validate")
    state.finalize(audit_result)
    return state.to_result(audit_result)


async def seed_phase(ctx: WorkflowContext, case: Case, state: GraphState) -> None:
    """Verify every injection seed in parallel."""
    ctx.phase("seed")
    state.init_fresh()

    seed_injections = [inj for inj in case.injections if inj.get("target")]
    seed_coros: list[Awaitable[tuple[Injection, SeedResult | None]]] = [
        discovery.verify_seed(ctx, case, state, inj) for inj in seed_injections
    ]
    seed_results = await parallel_limited(ctx, seed_coros, case.max_parallel_tasks)
    for inj, result in zip(seed_injections, seed_results):
        if result is None:
            state.record_error(
                "seed",
                injection_node_id(inj),
                "seed verifier task failed before returning a result",
            )
            continue
        seed_verdict = result[1]
        root_id = injection_node_id(inj)
        if seed_verdict and seed_verdict.get("verdict") == "confirmed":
            root_id = state.accept_seed_node(inj, seed_verdict)
            state.confirmed_seed_ids.add(root_id)
            state.clear_error("seed", root_id)
            ctx.log(f"seed {root_id}: confirmed ({seed_verdict.get('predicate')})")
        elif seed_verdict and seed_verdict.get("verdict") == "inconclusive":
            ctx.log(f"seed {root_id}: inconclusive — keeping for audit review")
        else:
            v = (
                seed_verdict.get("verdict", "no result")
                if seed_verdict
                else "no result"
            )
            ctx.log(f"seed {root_id}: {v} — skipping")
        if seed_verdict:
            state.seed_verdicts[root_id] = seed_verdict

    if not state.nodes:
        ctx.log("no seeds confirmed after seed map; audit may request rechecks")


async def seed_recheck_unconfirmed_phase(
    ctx: WorkflowContext,
    case: Case,
    state: GraphState,
) -> None:
    """Run final-contract seed rechecks before propagation starts."""
    inj_by_seed = {
        injection_node_id(inj): inj for inj in case.injections if inj.get("target")
    }
    missing_seeds = [
        seed_id
        for seed_id in sorted(case.seeds)
        if seed_id not in state.confirmed_seed_ids and seed_id in inj_by_seed
    ]
    if not missing_seeds:
        return
    ctx.phase("seed-recheck")
    ctx.log(f"Rechecking {len(missing_seeds)} unconfirmed seeds before propagation")
    context_by_seed = {
        obligation.source_seed: obligation.context
        for obligation in obligations_from_report(state.evaluate_final_checks())
        if obligation.kind == "seed_confirmed" and obligation.source_seed
    }
    seed_results = await parallel_limited(
        ctx,
        [
            discovery.verify_seed(
                ctx,
                case,
                state,
                inj_by_seed[seed_id],
                context_by_seed.get(seed_id, ""),
            )
            for seed_id in missing_seeds
        ],
        case.max_parallel_tasks,
    )
    for seed_id, seed_pair in zip(missing_seeds, seed_results):
        if seed_pair is None:
            state.record_error(
                "seed",
                seed_id,
                "seed recheck failed before returning a result",
            )
            continue
        inj, seed_verdict = seed_pair
        if seed_verdict:
            state.seed_verdicts[seed_id] = seed_verdict
        verdict = seed_verdict.get("verdict") if seed_verdict else "no-result"
        ctx.log(f"seed recheck {seed_id}: {verdict}")
        if seed_verdict and verdict == "confirmed":
            accepted_seed = state.accept_seed_node(inj, seed_verdict)
            state.confirmed_seed_ids.add(accepted_seed)
            state.clear_error("seed", accepted_seed)


async def propagate(
    ctx: WorkflowContext,
    case: Case,
    state: GraphState,
    roots: Sequence[str],
) -> bool:
    """Stream candidates through a worker pool — no round or wave barriers.

    Confirmed hops immediately enqueue the new node's candidates. Workers
    pick them up as capacity frees. The only synchronization point is
    ``FIRST_COMPLETED``: each result is processed the moment it arrives
    and idle workers are refilled from the priority queue.

    Needs filtering happens at dequeue time (lazy): candidates that no
    longer address an unresolved obligation are skipped without an LLM
    call. Final checks are cached and only recomputed when the graph
    actually changes.
    """
    state.round_n += 1

    # -- Cached final checks (recomputed only when graph changes) ----------
    _fc_report: FinalCheckReport | None = None

    def _checks() -> FinalCheckReport:
        nonlocal _fc_report
        if _fc_report is None:
            _fc_report = state.evaluate_final_checks()
        return _fc_report

    def _invalidate() -> None:
        nonlocal _fc_report
        _fc_report = None

    if _checks()["passed"]:
        ctx.log("final checks already satisfied")
        return False

    # -- Priority context (rebuilt cheaply on each confirmed hop) ----------
    def _build_pctx() -> Any:
        return build_priority_context(
            graph=case.graph,
            data_profile=case.data_profile,
            data_dir=case.data_dir,
            entry_services=case.entry_services,
            anomaly_inventory=case.anomaly_inventory,
            accepted_adj=state.adj,
            gate_log=state.gate_log,
            source_adj_by_seed=state.source_adjacencies(),
        )

    pctx = _build_pctx()

    # -- Priority queue: (priority_tuple, seq, candidate) ------------------
    pq: list[tuple[Any, int, CandidateEdge]] = []
    _seq = 0
    _scheduled: set[str] = set()

    def _enqueue(candidate: CandidateEdge) -> None:
        nonlocal _seq
        key = state.edge_key(
            candidate["from_service"],
            candidate["to_service"],
            candidate["source_seed"],
        )
        if key in _scheduled or key in state.checked_edges:
            return
        _scheduled.add(key)
        src, dst = candidate["from_service"], candidate["to_service"]
        if dst in state.nodes and _reaches(state.adj, dst, src):
            state.checked_edges.add(key)
            state.hop_log.append({
                "round": state.round_n,
                "from": src,
                "to": dst,
                "verdict": "skipped_cycle",
                "source_seed": candidate["source_seed"],
            })
            return
        state.record_candidate(candidate)
        heapq.heappush(pq, (edge_priority(candidate, pctx), _seq, candidate))
        _seq += 1

    def _enqueue_frontier(node: str, source_seed: str) -> None:
        structural = structural_candidates(
            case.graph, source_seed=source_seed, from_service=node,
        )
        existing = {c["to_service"] for c in structural}
        for c in structural:
            _enqueue(c)
        for c in anomaly_candidates(
            case.graph, case.anomaly_inventory,
            source_seed=source_seed, from_service=node,
            existing_targets=existing,
        ):
            _enqueue(c)

    # -- Seed initial candidates from roots --------------------------------
    for root in dict.fromkeys(roots):
        if root not in state.nodes or root in case.infra_set or is_slo_endpoint(case, root):
            continue
        for source_seed in sorted(state.node_sources.get(root) or {root}):
            _enqueue_frontier(root, source_seed)

    if not pq:
        ctx.log("propagation frontier exhausted")
        return False

    ctx.log(
        f"Propagation: {len(pq)} candidates, "
        f"up to {case.max_parallel_tasks} concurrent workers"
    )

    # -- Worker pool -------------------------------------------------------
    in_flight: dict[
        asyncio.Task[tuple[CandidateEdge, HopResult | None]],
        CandidateEdge,
    ] = {}
    changed = False

    async def _verify(
        item: CandidateEdge,
    ) -> tuple[CandidateEdge, HopResult | None]:
        try:
            result = await discovery.verify_hop(
                ctx, case, state,
                item["from_service"],
                item["to_service"],
                item["rel_type"],
                source_seed=item["source_seed"],
                fault_record_override=state.fault_for_node(
                    item["from_service"], item["source_seed"],
                ),
            )
        except Exception as exc:  # noqa: BLE001
            ctx.log(
                f"  {item['source_seed']}: "
                f"{item['from_service']} -> {item['to_service']}: "
                f"task error {type(exc).__name__}: {exc}"
            )
            result = None
        return item, result

    def _fill() -> None:
        """Submit highest-priority candidates to idle workers."""
        needs = build_propagation_needs(
            report=_checks(), graph=case.graph,
        )
        while pq and len(in_flight) < case.max_parallel_tasks:
            if needs.passed:
                break
            _, _, candidate = heapq.heappop(pq)
            if not candidate_addresses_needs(candidate, needs):
                continue
            task = asyncio.create_task(_verify(candidate))
            in_flight[task] = candidate

    _fill()

    while in_flight:
        done, _ = await asyncio.wait(
            set(in_flight), return_when=asyncio.FIRST_COMPLETED,
        )
        for task in done:
            in_flight.pop(task)
            candidate, result = task.result()
            from_svc = candidate["from_service"]
            to_svc = candidate["to_service"]
            source_seed = candidate["source_seed"]
            rel_type = candidate["rel_type"]
            edge_key = state.edge_key(from_svc, to_svc, source_seed)
            state.checked_edges.add(edge_key)

            if result is None:
                state.record_error(
                    "hop", edge_key,
                    "hop verifier task failed before returning a result",
                )
            verdict = result.get("verdict") if result else None
            verdict_label = verdict or "no-result"
            state.hop_log.append({
                "round": state.round_n,
                "from": from_svc,
                "to": to_svc,
                "verdict": verdict_label,
                "source_seed": source_seed,
            })
            ctx.log(f"  {source_seed}: {from_svc} -> {to_svc}: {verdict_label}")
            if result and verdict:
                state.verdicts[edge_key] = result

            if verdict != "confirmed":
                continue
            assert result is not None
            accepted = state.accept_hop_result(
                from_svc, to_svc, rel_type, result,
                source_seed=source_seed,
                fault=state.fault_for_node(from_svc, source_seed),
            )
            if not accepted:
                continue
            changed = True
            _invalidate()
            pctx = _build_pctx()
            if to_svc not in case.infra_set and not is_slo_endpoint(case, to_svc):
                _enqueue_frontier(to_svc, source_seed)

        if _checks()["passed"]:
            if in_flight:
                ctx.log(
                    f"final checks satisfied; canceling "
                    f"{len(in_flight)} in-flight hops"
                )
                await cancel_tasks(set(in_flight))
                in_flight.clear()
            break

        _fill()

    return changed


async def audit_loop(
    ctx: WorkflowContext,
    case: Case,
    state: GraphState,
) -> AuditOutcome:
    """Run targeted audit passes plus a free exploration sweep to a fixpoint.

    The audit never edits the graph directly. Every gap becomes a
    re-dispatch request whose gated discovery verdict drives the actual
    node/edge changes. ``accepted`` is the fixpoint: a full round that
    surfaces no re-dispatch and no unexplained anomaly.
    """
    coverage: dict[str, SeedCoverageStatus] = {}
    unexplained: list[str] = []
    accepted = False
    rounds = 0
    for audit_round in range(case.max_audit_rounds):
        rounds = audit_round + 1
        ctx.phase("audit" if audit_round == 0 else f"audit-r{rounds}")

        coverage, reach_reqs, reach_reports = await audit_map.audit_pass_reachability(
            ctx, case, state, audit_round
        )
        unexplained, cov_reqs, cov_reports = await audit_map.audit_pass_coverage(
            ctx, case, state, audit_round
        )
        targeted_reqs: list[ReworkRequest] = [*reach_reqs, *cov_reqs]
        targeted_changed, targeted_log = await redispatch(
            ctx,
            case,
            state,
            targeted_reqs,
        )

        explore_reqs, explore_report = await audit_map.free_explore(
            ctx, case, state, audit_round
        )
        b_changed, b_log = await redispatch(ctx, case, state, explore_reqs)

        state.audit_rounds_log.append(
            {
                "round": rounds,
                "reachability_reports": reach_reports,
                "coverage_reports": cov_reports,
                "explore_report": explore_report,
                "seed_coverage": coverage,
                "unexplained_anomalies": unexplained,
                "redispatch_results": [*targeted_log, *b_log],
            }
        )

        # Fixpoint: a full round surfaced no gaps and left nothing unexplained.
        if not targeted_reqs and not explore_reqs and not unexplained:
            accepted = True
            break
        # No progress: the passes keep asking but no verdict changed the graph.
        if not targeted_changed and not b_changed:
            ctx.log("audit made no progress this round; stopping")
            break

    if not accepted and unexplained:
        state.record_error(
            "audit",
            "global",
            "audit closed with unexplained anomalies",
        )

    return AuditOutcome(
        accepted=accepted,
        rounds=rounds,
        seed_coverage=coverage,
        unexplained_anomalies=unexplained,
    )


async def final_check_loop(
    ctx: WorkflowContext,
    case: Case,
    state: GraphState,
) -> None:
    """Turn deterministic final-check gaps into gated re-verification work."""
    for round_idx in range(case.max_audit_rounds):
        report = state.evaluate_final_checks()
        if report["passed"]:
            return
        requests = final_check_rework_requests(case, state, report)
        state.audit_rounds_log.append(
            {
                "round": f"final-check-{round_idx + 1}",
                "final_check_report": report,
                "redispatch_requests": [
                    req.model_dump(mode="json") for req in requests
                ],
            }
        )
        if not requests:
            ctx.log("final checks found gaps but no direct recheck is available")
            return
        ctx.phase(
            "final-check-rework"
            if round_idx == 0
            else f"final-check-rework-{round_idx + 1}"
        )
        ctx.log(f"Final checks requested {len(requests)} rechecks")
        graph_before = graph_progress_signature(state)
        changed, rework_log = await redispatch(ctx, case, state, requests)
        state.audit_rounds_log[-1]["redispatch_results"] = rework_log
        resolution_changed = resolve_frontend_obligations_from_rework(
            state,
            requests,
            rework_log,
        )
        if resolution_changed:
            state.audit_rounds_log[-1]["resolved_frontend_anomalies"] = (
                state.final_anomaly_resolutions
            )
            changed = True
        graph_after = graph_progress_signature(state)
        if not changed or (graph_after == graph_before and not resolution_changed):
            ctx.log("final-check rework made no graph progress; stopping")
            return


def graph_progress_signature(state: GraphState) -> tuple[
    frozenset[str],
    tuple[tuple[object, object], ...],
    tuple[tuple[str, str, tuple[str, ...]], ...],
]:
    """Capture graph progress, including source-specific edge attribution."""
    return (
        frozenset(state.nodes),
        tuple((edge.get("src"), edge.get("dst")) for edge in state.edges),
        tuple(
            (src, dst, tuple(sorted(sources)))
            for (src, dst), sources in sorted(state.edge_sources.items())
        ),
    )


def final_check_rework_requests(
    case: Case,
    state: GraphState,
    report: FinalCheckReport,
) -> list[ReworkRequest]:
    obligations = obligations_from_report(report)
    return rework_requests_for_obligations(
        graph=case.graph,
        nodes=state.nodes,
        adj=state.adj,
        source_adj_by_seed=state.source_adjacencies(),
        node_sources=state.node_sources,
        obligations=obligations,
    )


def _hop_rework_verdict_key(
    state: GraphState,
    req: HopRecheckRequest,
    source_seed: str | None,
) -> str:
    edge_key = state.edge_key(req.from_service, req.to_service, source_seed)
    if req.obligation_kind == "frontend_anomaly" and req.obligation_id:
        return f"{edge_key}@@{req.obligation_id}"
    return edge_key


def resolve_frontend_obligations_from_rework(
    state: GraphState,
    requests: Sequence[ReworkRequest],
    rework_log: Sequence[dict[str, Any]],
) -> bool:
    """Resolve concrete frontend anomalies when all candidate paths reject."""
    request_counts: dict[str, int] = {}
    anomaly_by_obligation: dict[str, str] = {}
    for req in requests:
        if (
            isinstance(req, HopRecheckRequest)
            and req.obligation_kind == "frontend_anomaly"
            and req.obligation_id
            and req.anomaly_id
        ):
            request_counts[req.obligation_id] = request_counts.get(req.obligation_id, 0) + 1
            anomaly_by_obligation[req.obligation_id] = req.anomaly_id

    results_by_obligation: dict[str, list[dict[str, Any]]] = {}
    for row in rework_log:
        obligation = row.get("obligation", {})
        if not isinstance(obligation, dict):
            continue
        obligation_id = str(obligation.get("id") or "")
        if obligation_id in request_counts:
            results_by_obligation.setdefault(obligation_id, []).append(row)

    changed = False
    for obligation_id, expected_count in request_counts.items():
        rows = results_by_obligation.get(obligation_id, [])
        if len(rows) < expected_count:
            continue
        verdicts = [str(row.get("verdict") or "") for row in rows]
        if not verdicts or any(verdict != "rejected" for verdict in verdicts):
            continue
        anomaly_id = anomaly_by_obligation[obligation_id]
        rationale = (
            "All gated candidate paths for this frontend anomaly were rejected "
            "as not causally connected to the confirmed seed paths. Treating "
            "the anomaly as unrelated/background for final-check purposes."
        )
        evidence = [
            {
                "from": row.get("from"),
                "to": row.get("to"),
                "source_seed": row.get("source_seed"),
                "verdict": row.get("verdict"),
                "rationale": row.get("rationale", ""),
            }
            for row in rows
        ]
        changed = state.resolve_frontend_anomaly(
            anomaly_id,
            resolution="unrelated_or_background",
            rationale=rationale,
            evidence=evidence,
        ) or changed
    return changed


async def redispatch(
    ctx: WorkflowContext,
    case: Case,
    state: GraphState,
    requests: Sequence[ReworkRequest],
) -> tuple[bool, list[dict[str, Any]]]:
    """Execute the seed/hop rechecks an audit pass requested.

    Re-runs the gated discovery agents with the audit's context; the verdict
    drives the graph: a confirmed hop adds the edge, a rejected hop removes
    it. This is the only mechanism by which the audit changes the graph.
    """
    if not requests:
        return False, []
    rework_log: list[dict[str, Any]] = []
    changed = False
    inj_by_seed = {
        injection_node_id(inj): inj for inj in case.injections if inj.get("target")
    }

    seed_requests = [
        req
        for req in requests
        if isinstance(req, SeedRecheckRequest) and req.seed in inj_by_seed
    ]
    new_roots: list[str] = []
    if seed_requests:
        ctx.phase("audit-seed-rework")
        ctx.log(f"Audit requested {len(seed_requests)} seed rechecks")
        seed_results = await parallel_limited(
            ctx,
            [
                discovery.verify_seed(
                    ctx, case, state, inj_by_seed[req.seed], req.context
                )
                for req in seed_requests
            ],
            case.max_parallel_tasks,
        )
        for req, seed_pair in zip(seed_requests, seed_results):
            if seed_pair is None:
                state.record_error(
                    "seed",
                    req.seed,
                    "audit seed recheck failed before returning a result",
                )
                continue
            inj, seed_verdict = seed_pair
            seed_id = injection_node_id(inj)
            previous_seed_verdict = state.seed_verdicts.get(seed_id)
            if seed_verdict:
                state.seed_verdicts[seed_id] = seed_verdict
                if seed_verdict != previous_seed_verdict:
                    changed = True
            verdict = seed_verdict.get("verdict") if seed_verdict else "no-result"
            rework_log.append(
                {
                    "kind": "seed_recheck",
                    "seed": seed_id,
                    "verdict": verdict,
                    "rationale": seed_verdict.get("rationale", "")
                    if seed_verdict
                    else "",
                }
            )
            ctx.log(f"  seed recheck {seed_id}: {verdict}")
            if seed_verdict and verdict == "confirmed":
                before_roots = len(state.propagation_roots)
                accepted_seed = state.accept_seed_node(inj, seed_verdict)
                new_roots.extend(state.propagation_roots[before_roots:])
                state.confirmed_seed_ids.add(accepted_seed)
                state.clear_error("seed", accepted_seed)
                changed = True

    hop_requests = [
        req
        for req in requests
        if isinstance(req, HopRecheckRequest) and req.from_service in state.nodes
    ]
    if new_roots:
        ctx.phase("audit-propagate-rework")
        ctx.log(f"Propagating from {len(set(new_roots))} audit-confirmed roots")
        changed = await propagate(ctx, case, state, new_roots) or changed
        new_roots = []
        if state.evaluate_final_checks()["passed"]:
            return changed, rework_log
        active_obligations = {
            obligation.id
            for obligation in obligations_from_report(state.evaluate_final_checks())
        }
        before_filter = len(hop_requests)
        hop_requests = [
            req
            for req in hop_requests
            if not req.obligation_id or req.obligation_id in active_obligations
        ]
        if len(hop_requests) != before_filter:
            ctx.log(
                "Skipping "
                f"{before_filter - len(hop_requests)} obsolete hop rechecks "
                "after seed propagation"
            )
    if hop_requests:
        ctx.phase("audit-hop-rework")
        ctx.log(f"Audit requested {len(hop_requests)} hop rechecks")

        async def _one_hop_rework(
            req: HopRecheckRequest,
        ) -> tuple[HopRecheckRequest, HopResult | None]:
            rel_type = (
                req.rel_type
                or graph_rel_type(case.graph, req.from_service, req.to_service)
                or "other"
            )
            source_seed = req.source_seed or next(
                iter(sorted(state.node_sources.get(req.from_service, set()))),
                None,
            )
            prior = dict(
                state.verdicts.get(_hop_rework_verdict_key(state, req, source_seed))
                or state.verdicts.get(
                    state.edge_key(req.from_service, req.to_service, source_seed)
                )
                or {}
            )
            return req, await discovery.verify_hop(
                ctx,
                case,
                state,
                req.from_service,
                req.to_service,
                rel_type,
                judge_context=req.context,
                source_seed=source_seed,
                fault_record_override=state.fault_for_node(
                    req.from_service,
                    source_seed,
                ),
                prior_verdict=PriorVerdict(
                    verdict=str(prior.get("verdict", "")),
                    rationale=str(prior.get("rationale", "")),
                ),
                obligation_context=obligation_payload(req),
            )

        hop_results = await parallel_limited(
            ctx,
            [_one_hop_rework(req) for req in hop_requests],
            case.max_parallel_tasks,
        )
        for hop_req, pair in zip(hop_requests, hop_results):
            if pair is None:
                state.record_error(
                    "hop",
                    hop_req.from_service + "__" + hop_req.to_service,
                    "audit hop recheck failed before returning a result",
                )
                continue
            result = pair[1]
            from_svc = hop_req.from_service
            to_svc = hop_req.to_service
            source_seed = hop_req.source_seed or next(
                iter(sorted(state.node_sources.get(from_svc, set()))),
                None,
            )
            hop_verdict: str | None = result.get("verdict") if result else None
            edge_key = state.edge_key(from_svc, to_svc, source_seed)
            verdict_key = _hop_rework_verdict_key(state, hop_req, source_seed)
            obligation = obligation_payload(hop_req)
            state.hop_log.append(
                {
                    "round": state.round_n,
                    "from": from_svc,
                    "to": to_svc,
                    "verdict": (hop_verdict or "no-result") + "(audit-rework)",
                    "source_seed": source_seed or "",
                    "obligation_id": hop_req.obligation_id or "",
                }
            )
            if result and hop_verdict:
                previous_hop_verdict = state.verdicts.get(verdict_key)
                state.verdicts[verdict_key] = result
                if hop_req.obligation_kind == "frontend_anomaly" and (
                    hop_verdict == "confirmed"
                ):
                    state.verdicts[edge_key] = result
                if result != previous_hop_verdict:
                    changed = True
            rework_log.append(
                {
                    "kind": "hop_recheck",
                    "from": from_svc,
                    "to": to_svc,
                    "source_seed": source_seed,
                    "obligation": obligation,
                    "verdict": hop_verdict or "no-result",
                    "rationale": result.get("rationale", "") if result else "",
                }
            )
            ctx.log(
                f"  hop recheck {from_svc} -> {to_svc}: {hop_verdict or 'no-result'}"
            )
            # Removal is verdict-driven: a re-verification that rejects the
            # propagation hypothesis removes the edge (and prunes orphans).
            if result is not None and hop_verdict == "rejected":
                if (
                    hop_req.obligation_kind != "frontend_anomaly"
                    and state.remove_hop_edge(from_svc, to_svc, source_seed)
                ):
                    changed = True
                continue
            if hop_verdict != "confirmed" or result is None:
                continue
            rel_type = (
                hop_req.rel_type
                or graph_rel_type(case.graph, from_svc, to_svc)
                or "other"
            )
            accepted = state.accept_hop_result(
                from_svc,
                to_svc,
                rel_type,
                result,
                claim_override=hop_req.context[:200],
                source_seed=source_seed,
                fault=state.fault_for_node(from_svc, source_seed),
            )
            changed = accepted or changed
            if (
                accepted
                and
                to_svc in state.nodes
                and to_svc not in case.infra_set
                and not is_slo_endpoint(case, to_svc)
                and source_seed is not None
            ):
                new_roots.append(to_svc)
    if new_roots:
        ctx.phase("audit-propagate-rework")
        ctx.log(f"Propagating from {len(set(new_roots))} audit-confirmed roots")
        changed = await propagate(ctx, case, state, new_roots) or changed
    return changed, rework_log


def is_slo_endpoint(case: Case, service: str) -> bool:
    return frontend_like(service, case.entry_services)


async def parallel_limited(
    ctx: WorkflowContext,
    coros: Sequence[Awaitable[Any]],
    limit: int,
) -> list[Any]:
    """Run all tasks, but avoid spawning an unbounded number of child agents."""
    if not coros:
        return []
    if limit <= 0 or len(coros) <= limit:
        return list(await ctx.parallel(list(coros)))
    out: list[Any] = []
    for start in range(0, len(coros), limit):
        out.extend(await ctx.parallel(list(coros[start:start + limit])))
    return out


async def cancel_tasks(tasks: set[asyncio.Task[Any]]) -> None:
    """Cancel in-flight speculative checks once the current obligations are met."""
    if not tasks:
        return
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    tasks.clear()


