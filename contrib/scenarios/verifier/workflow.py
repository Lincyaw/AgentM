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
    skip_judge, window, rel_mechanism.

Output: a ``PropagationResult`` dict (see ``state.GraphState.to_result``).
"""

from __future__ import annotations

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
    for inj, result in zip(seed_injections, seed_results, strict=True):
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


async def propagate(
    ctx: WorkflowContext,
    case: Case,
    state: GraphState,
    roots: Sequence[str],
) -> bool:
    """Scan each confirmed fault chain over structural/anomaly candidates."""
    queue: list[tuple[str, str]] = []
    for root in dict.fromkeys(roots):
        if root not in state.nodes or root in case.infra_set or is_slo_endpoint(case, root):
            continue
        sources = state.node_sources.get(root) or {root}
        queue.extend((root, source_seed) for source_seed in sorted(sources))
    changed_any = False

    while queue:
        state.round_n += 1
        batch = list(dict.fromkeys(queue))
        queue = []

        pending_hops: list[CandidateEdge] = []
        for current, source_seed in batch:
            structural = structural_candidates(
                case.graph,
                source_seed=source_seed,
                from_service=current,
            )
            existing_targets = {candidate["to_service"] for candidate in structural}
            candidates = [
                *structural,
                *anomaly_candidates(
                    case.graph,
                    case.anomaly_inventory,
                    source_seed=source_seed,
                    from_service=current,
                    existing_targets=existing_targets,
                ),
            ]
            for candidate in candidates:
                neighbor = candidate["to_service"]
                edge_key = state.edge_key(current, neighbor, source_seed)
                if edge_key in state.checked_edges:
                    continue
                state.checked_edges.add(edge_key)
                state.record_candidate(candidate)

                # fpg DAG rule: never evaluate an edge that would close a cycle
                # through already-accepted edges. The source seed is logged so a
                # different co-injected fault can still scan the same edge.
                if neighbor in state.nodes and _reaches(state.adj, neighbor, current):
                    state.hop_log.append(
                        {
                            "round": state.round_n,
                            "from": current,
                            "to": neighbor,
                            "verdict": "skipped_cycle",
                            "source_seed": source_seed,
                        }
                    )
                    continue
                pending_hops.append(candidate)

        if not pending_hops:
            continue

        ctx.log(f"Round {state.round_n}: {len(pending_hops)} candidate hops")

        coros: list[Awaitable[HopResult | None]] = [
            discovery.verify_hop(
                ctx,
                case,
                state,
                item["from_service"],
                item["to_service"],
                item["rel_type"],
                source_seed=item["source_seed"],
                fault_record_override=state.fault_for_node(
                    item["from_service"],
                    item["source_seed"],
                ),
            )
            for item in pending_hops
        ]
        results = await parallel_limited(ctx, coros, case.max_parallel_tasks)

        for candidate, result in zip(pending_hops, results, strict=True):
            from_svc = candidate["from_service"]
            to_svc = candidate["to_service"]
            rel_type = candidate["rel_type"]
            source_seed = candidate["source_seed"]
            edge_key = state.edge_key(from_svc, to_svc, source_seed)
            if result is None:
                state.record_error(
                    "hop",
                    edge_key,
                    "hop verifier task failed before returning a result",
                )
            verdict = result.get("verdict") if result else None
            verdict_label = verdict or "no-result"
            state.hop_log.append(
                {
                    "round": state.round_n,
                    "from": from_svc,
                    "to": to_svc,
                    "verdict": verdict_label,
                    "source_seed": source_seed,
                }
            )
            ctx.log(f"  {source_seed}: {from_svc} -> {to_svc}: {verdict_label}")
            if result and verdict:
                state.verdicts[edge_key] = result

            if verdict != "confirmed":
                continue
            assert result is not None
            accepted = state.accept_hop_result(
                from_svc,
                to_svc,
                rel_type,
                result,
                source_seed=source_seed,
                fault=state.fault_for_node(from_svc, source_seed),
            )
            changed_any = accepted or changed_any
            if accepted and to_svc not in case.infra_set and not is_slo_endpoint(case, to_svc):
                queue.append((to_svc, source_seed))

        if not queue:
            ctx.log("propagation frontier exhausted for this round")

    return changed_any


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
        graph_before = (
            frozenset(state.nodes),
            tuple((edge.get("src"), edge.get("dst")) for edge in state.edges),
        )
        changed, rework_log = await redispatch(ctx, case, state, requests)
        state.audit_rounds_log[-1]["redispatch_results"] = rework_log
        graph_after = (
            frozenset(state.nodes),
            tuple((edge.get("src"), edge.get("dst")) for edge in state.edges),
        )
        if not changed or graph_after == graph_before:
            ctx.log("final-check rework made no graph progress; stopping")
            return


def final_check_rework_requests(
    case: Case,
    state: GraphState,
    report: FinalCheckReport,
) -> list[ReworkRequest]:
    requests: list[ReworkRequest] = []
    seen: set[tuple[str, str, str | None]] = set()
    for issue in report["issues"]:
        if issue["check"] != "frontend_anomaly_explained":
            continue
        service = str(issue.get("details", {}).get("service") or "")
        details = issue.get("details", {})
        requests.extend(
            _requests_to_target(
                case,
                state,
                service,
                context=(
                    "Final invariant gap: explain SLO/frontend anomaly "
                    f"{issue['item']} on {service}. "
                    f"Subject={details.get('subject')}; "
                    f"component={details.get('component')}. "
                    "This is an endpoint/anomaly coverage check, not a "
                    "generic service-health check. Add this hop only if "
                    "same-trace or endpoint-specific evidence connects the "
                    "confirmed upstream fault path to this requested SLO "
                    "symptom. Aggregate frontend/proxy error rate, latency, "
                    "or span count alone is insufficient. If only a subset "
                    "of frontend/proxy failures is path-aligned with the "
                    "fault, say exactly which subset is explained and "
                    "separate unrelated/background frontend failures."
                ),
                seen=seen,
                allow_other=True,
                frontier_only=True,
            )
        )
    targeted_seeds = {
        req.source_seed for req in requests
        if isinstance(req, HopRecheckRequest) and req.source_seed
    }
    for issue in report["issues"]:
        if issue["check"] == "seed_reaches_entry":
            seed = issue["item"]
            if seed in targeted_seeds:
                continue
            targets = issue.get("details", {}).get("frontend_services", [])
            if not isinstance(targets, list) or not targets:
                targets = sorted(case.entry_services)
            for target in sorted(str(target) for target in targets):
                requests.extend(
                    _requests_to_target(
                        case,
                        state,
                        target,
                        source_seed=seed,
                        context=(
                            "Final invariant gap: confirmed seed must reach an "
                            f"SLO/entry endpoint. Verify whether the path from "
                            f"{seed} reaches {target} through this hop."
                        ),
                        seen=seen,
                    )
                )
    return requests


def _requests_to_target(
    case: Case,
    state: GraphState,
    target: str,
    *,
    context: str,
    seen: set[tuple[str, str, str | None]],
    source_seed: str | None = None,
    allow_other: bool = False,
    frontier_only: bool = False,
) -> list[ReworkRequest]:
    if not target:
        return []
    out: list[ReworkRequest] = []
    for from_svc in sorted(state.nodes):
        if from_svc == target:
            continue
        if frontier_only and state.adj.get(from_svc):
            continue
        rel_type = graph_rel_type(case.graph, from_svc, target)
        if not rel_type and not allow_other:
            continue
        rel_type = rel_type or "other"
        source_seeds = (
            [source_seed]
            if source_seed
            else sorted(state.node_sources.get(from_svc, set()))
        )
        for seed in source_seeds:
            if not seed or not _reaches(state.adj, seed, from_svc):
                continue
            key = (from_svc, target, seed)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                HopRecheckRequest(
                    kind="hop_recheck",
                    from_service=from_svc,
                    to_service=target,
                    rel_type=rel_type,
                    source_seed=seed,
                    context=context,
                )
            )
    return out


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
        for req, seed_pair in zip(seed_requests, seed_results, strict=True):
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
                state.verdicts.get(
                    state.edge_key(req.from_service, req.to_service, source_seed),
                    {},
                )
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
            )

        hop_results = await parallel_limited(
            ctx,
            [_one_hop_rework(req) for req in hop_requests],
            case.max_parallel_tasks,
        )
        for hop_req, pair in zip(hop_requests, hop_results, strict=True):
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
            state.hop_log.append(
                {
                    "round": state.round_n,
                    "from": from_svc,
                    "to": to_svc,
                    "verdict": (hop_verdict or "no-result") + "(audit-rework)",
                    "source_seed": source_seed or "",
                }
            )
            if result and hop_verdict:
                previous_hop_verdict = state.verdicts.get(edge_key)
                state.verdicts[edge_key] = result
                if result != previous_hop_verdict:
                    changed = True
            rework_log.append(
                {
                    "kind": "hop_recheck",
                    "from": from_svc,
                    "to": to_svc,
                    "source_seed": source_seed,
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
                if state.remove_hop_edge(from_svc, to_svc, source_seed):
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
