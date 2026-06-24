"""Fault propagation verification workflow (module mode), fpg-native.

This module is the thin orchestration core. It wires three phases over a
``GraphState``:

  - seed_phase   — verify each injection (``discovery.verify_seed``);
                   confirmed seeds become propagation roots.
  - propagate    — BFS the neighbor graph, verifying each candidate edge
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
from .lib.fpg import injection_node_id
from .lib.schema import (
    AuditOutcome,
    HopRecheckRequest,
    HopResult,
    Injection,
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

    ctx.phase("validate")
    state.finalize(audit_result)
    return state.to_result(audit_result)


async def seed_phase(ctx: WorkflowContext, case: Case, state: GraphState) -> None:
    """Phase 0: verify every injection seed in parallel."""
    ctx.phase("seed")
    state.init_fresh()

    seed_injections = [inj for inj in case.injections if inj.get("target")]
    seed_coros: list[Awaitable[tuple[Injection, SeedResult | None]]] = [
        discovery.verify_seed(ctx, case, state, inj) for inj in seed_injections
    ]
    seed_results = await ctx.parallel(seed_coros)
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


async def propagate(
    ctx: WorkflowContext,
    case: Case,
    state: GraphState,
    roots: Sequence[str],
) -> bool:
    """Phase 1: BFS the neighbor graph, verifying each candidate edge."""
    queue = [
        root
        for root in dict.fromkeys(roots)
        if root in state.nodes
        and root not in case.infra_set
        and root not in case.entry_services
    ]
    changed_any = False

    while queue:
        state.round_n += 1
        batch = list(queue)
        queue = []

        pending_hops: list[list[str]] = []
        for current in batch:
            for neighbor_info in case.graph.get(current, []):
                neighbor = neighbor_info[0]
                rel_type = neighbor_info[1]
                edge_key = current + "__" + neighbor
                if edge_key in state.checked_edges:
                    continue
                state.checked_edges.add(edge_key)

                # fpg DAG rule: never evaluate an edge that would
                # close a cycle through already-accepted edges.
                if neighbor in state.nodes and _reaches(state.adj, neighbor, current):
                    state.hop_log.append(
                        {
                            "round": state.round_n,
                            "from": current,
                            "to": neighbor,
                            "verdict": "skipped_cycle",
                        }
                    )
                    continue
                pending_hops.append([current, neighbor, rel_type])

        if not pending_hops:
            continue

        ctx.log(f"Round {state.round_n}: {len(pending_hops)} hops")

        coros: list[Awaitable[HopResult | None]] = [
            discovery.verify_hop(ctx, case, state, item[0], item[1], item[2])
            for item in pending_hops
        ]
        results = await ctx.parallel(coros)

        for (from_svc, to_svc, rel_type), result in zip(pending_hops, results):
            edge_key = from_svc + "__" + to_svc
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
                }
            )
            ctx.log(f"  {from_svc} -> {to_svc}: {verdict_label}")
            if result and verdict:
                state.verdicts[edge_key] = result

            if verdict != "confirmed":
                continue
            assert result is not None
            was_new_node = to_svc not in state.nodes
            accepted = state.accept_hop_result(
                from_svc,
                to_svc,
                rel_type,
                result,
            )
            changed_any = accepted or changed_any
            if accepted and was_new_node:
                if to_svc not in case.infra_set and to_svc not in case.entry_services:
                    queue.append(to_svc)

        if not queue:
            ctx.log("propagation frontier exhausted for this round")

    return changed_any


async def audit_loop(
    ctx: WorkflowContext,
    case: Case,
    state: GraphState,
) -> AuditOutcome:
    """Phase 2: two targeted passes + a free exploration sweep, looped to a
    fixpoint.

    The audit never edits the graph. Each round runs Stage A (Pass 1
    reachability + Pass 2 coverage) and Stage B (free exploration); every
    gap becomes a re-dispatch request whose gated discovery verdict drives
    the actual node/edge changes. The round's verified changes feed back
    into the next round's passes. ``accepted`` is the fixpoint: a full round
    that surfaces no re-dispatch and no unexplained anomaly.
    """
    coverage: dict[str, SeedCoverageStatus] = {}
    unexplained: list[str] = []
    accepted = False
    rounds = 0
    for audit_round in range(case.max_audit_rounds):
        rounds = audit_round + 1
        ctx.phase("audit" if audit_round == 0 else f"audit-r{rounds}")

        # Stage A — two targeted passes → re-dispatch
        coverage, reach_reqs, reach_reports = await audit_map.audit_pass_reachability(
            ctx, case, state, audit_round
        )
        unexplained, cov_reqs, cov_reports = await audit_map.audit_pass_coverage(
            ctx, case, state, audit_round
        )
        stage_a_reqs: list[ReworkRequest] = [*reach_reqs, *cov_reqs]
        a_changed, a_log = await redispatch(ctx, case, state, stage_a_reqs)

        # Stage B — free exploration (last step of the round) → re-dispatch
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
                "redispatch_results": [*a_log, *b_log],
            }
        )

        # Fixpoint: a full round surfaced no gaps and left nothing unexplained.
        if not stage_a_reqs and not explore_reqs and not unexplained:
            accepted = True
            break
        # No progress: the passes keep asking but no verdict changed the graph.
        if not a_changed and not b_changed:
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
        seed_results = await ctx.parallel(
            [
                discovery.verify_seed(
                    ctx, case, state, inj_by_seed[req.seed], req.context
                )
                for req in seed_requests
            ]
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
    if hop_requests:
        ctx.phase("audit-hop-rework")
        ctx.log(f"Audit requested {len(hop_requests)} hop rechecks")

        async def _one_hop_rework(
            req: HopRecheckRequest,
        ) -> tuple[HopRecheckRequest, HopResult | None]:
            rel_type = next(
                (
                    info[1]
                    for info in case.graph.get(req.from_service, [])
                    if info[0] == req.to_service
                ),
                "callee_to_caller",
            )
            prior = dict(
                state.verdicts.get(
                    req.from_service + "__" + req.to_service,
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
                prior_verdict=PriorVerdict(
                    verdict=str(prior.get("verdict", "")),
                    rationale=str(prior.get("rationale", "")),
                ),
            )

        hop_results = await ctx.parallel([_one_hop_rework(req) for req in hop_requests])
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
            hop_verdict: str | None = result.get("verdict") if result else None
            edge_key = from_svc + "__" + to_svc
            state.hop_log.append(
                {
                    "round": state.round_n,
                    "from": from_svc,
                    "to": to_svc,
                    "verdict": (hop_verdict or "no-result") + "(audit-rework)",
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
                if state.remove_hop_edge(from_svc, to_svc):
                    changed = True
                continue
            if hop_verdict != "confirmed" or result is None:
                continue
            was_new_node = to_svc not in state.nodes
            rel_type = next(
                (info[1] for info in case.graph.get(from_svc, []) if info[0] == to_svc),
                "callee_to_caller",
            )
            changed = (
                state.accept_hop_result(
                    from_svc,
                    to_svc,
                    rel_type,
                    result,
                    claim_override=hop_req.context[:200],
                )
                or changed
            )
            if (
                was_new_node
                and to_svc not in case.infra_set
                and to_svc not in case.entry_services
            ):
                new_roots.append(to_svc)
    if new_roots:
        ctx.phase("audit-propagate-rework")
        ctx.log(f"Propagating from {len(set(new_roots))} audit-confirmed roots")
        changed = await propagate(ctx, case, state, new_roots) or changed
    return changed, rework_log
