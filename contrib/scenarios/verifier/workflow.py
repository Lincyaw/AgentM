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
    skip_propagate, skip_judge, window, rel_mechanism, existing_state.

Output: a ``PropagationResult`` dict (see ``state.GraphState.to_result``).
"""
from __future__ import annotations

from collections.abc import Awaitable, Sequence
from typing import Any, cast

from agentm.extensions.builtin.workflow import WorkflowContext

from . import audit_map, discovery
from .hop.hop_context import PriorVerdict
from .lib.fpg import injection_node_id
from .lib.parallel import normalize_parallel
from .lib.schema import (
    GlobalAudit,
    HopRecheckRequest,
    HopResult,
    Injection,
    PropagationResult,
    ReworkRequest,
    SeedRecheckRequest,
    SeedResult,
    is_hop_rework_parallel_item,
    is_seed_parallel_item,
)
from .state import Case, GraphState, _reaches


async def run(ctx: WorkflowContext) -> PropagationResult:
    case = Case.from_args(ctx.args)
    state = GraphState(case, ctx.log)

    if case.skip_propagate:
        state.load_existing()
    else:
        await seed_phase(ctx, case, state)
        ctx.phase("propagate")
        await propagate(ctx, case, state, state.propagation_roots or list(state.nodes))

    audit_result: GlobalAudit | None = None
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
    seed_result_items = normalize_parallel(await ctx.parallel(seed_coros))
    for idx, item in enumerate(seed_result_items):
        if is_seed_parallel_item(item):
            continue
        if idx < len(seed_injections):
            state.record_error(
                "seed",
                injection_node_id(seed_injections[idx]),
                "seed verifier task failed before returning a result",
            )
    for inj in seed_injections[len(seed_result_items):]:
        state.record_error(
            "seed",
            injection_node_id(inj),
            "seed verifier task was missing from parallel results",
        )
    seed_results = [
        item for item in seed_result_items
        if is_seed_parallel_item(item)
    ]
    missing_seed_results = len(seed_coros) - len(seed_results)
    if missing_seed_results > 0:
        ctx.log(
            f"{missing_seed_results} seed verifier task(s) returned no usable result"
        )
    for inj, seed_verdict in seed_results:
        root_id = injection_node_id(inj)
        if seed_verdict and seed_verdict.get("verdict") == "confirmed":
            root_id = state.accept_seed_node(inj, seed_verdict)
            state.confirmed_seed_ids.add(root_id)
            state.clear_error("seed", root_id)
            ctx.log(
                f"seed {root_id}: confirmed ({seed_verdict.get('predicate')})"
            )
        elif seed_verdict and seed_verdict.get("verdict") == "inconclusive":
            ctx.log(f"seed {root_id}: inconclusive — keeping for audit review")
        else:
            v = seed_verdict.get("verdict", "no result") if seed_verdict else "no result"
            ctx.log(f"seed {root_id}: {v} — skipping")

    for inj, seed_verdict in seed_results:
        if seed_verdict:
            state.seed_verdicts[injection_node_id(inj)] = seed_verdict

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
        root for root in dict.fromkeys(roots)
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

        ctx.log("Round " + str(state.round_n) + ": " + str(len(pending_hops)) + " hops")

        coros: list[Awaitable[HopResult | None]] = [
            discovery.verify_hop(ctx, case, state, item[0], item[1], item[2])
            for item in pending_hops
        ]
        results = normalize_parallel(await ctx.parallel(coros))

        for idx in range(len(pending_hops)):
            from_svc, to_svc, rel_type = pending_hops[idx]
            result = results[idx] if idx < len(results) else None
            edge_key = from_svc + "__" + to_svc
            if result is None:
                state.record_error(
                    "hop",
                    edge_key,
                    "hop verifier task failed before returning a result",
                )
            verdict = result.get("verdict") if isinstance(result, dict) else None
            state.hop_log.append(
                {
                    "round": state.round_n,
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
            hop_result = cast(HopResult, result) if isinstance(result, dict) else None
            if hop_result and verdict:
                state.verdicts[edge_key] = hop_result

            if verdict != "confirmed":
                continue
            assert hop_result is not None
            was_new_node = to_svc not in state.nodes
            accepted = state.accept_hop_result(
                from_svc,
                to_svc,
                rel_type,
                hop_result,
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
) -> GlobalAudit | None:
    """Phase 2: audit map/reduce + rework until accept or exhaustion."""
    audit_result: GlobalAudit | None = None
    for audit_round in range(case.max_audit_rounds):
        ctx.phase("audit" if audit_round == 0 else f"audit-r{audit_round + 1}")
        audit_result, reports = await audit_map.run_audit_round(
            ctx, case, state, audit_round
        )
        causal_reports = reports["causal_reports"]
        audit_payload = (
            audit_result.model_dump(mode="json") if audit_result else {}
        )

        applied_drops = state.drop_audit_edges(
            audit_result.drop_edges if audit_result else [],
            causal_reports,
        )
        is_last_audit_round = audit_round >= case.max_audit_rounds - 1
        if (
            is_last_audit_round
            and audit_result is not None
            and not audit_result.accepted
        ):
            ctx.log("audit reached max rounds; skipping further rework")
            rework_changed = False
            rework_log: list[dict[str, Any]] = []
        else:
            rework_changed, rework_log = await apply_rework(
                ctx, case, state,
                audit_result.rework_requests if audit_result else [],
            )
        state.audit_rounds_log.append({
            "round": audit_round + 1,
            "anomaly_reports": reports["anomaly_reports"],
            "causal_reports": reports["causal_reports"],
            "seed_coverage_reports": reports["seed_coverage_reports"],
            "audit": audit_payload,
            "dropped_edges": [
                item.model_dump(mode="json")
                for item in (audit_result.drop_edges if audit_result else [])
            ],
            "rework_results": rework_log,
        })

        if audit_result and audit_result.accepted:
            break
        if not applied_drops and not rework_changed:
            ctx.log("audit has no effective rework left; stopping audit loop")
            break

    # The audit reducer owns the accept decision. Edge drops and the
    # invalid_causal_paths it lists are resolved-by-action (drops are
    # applied here, paths are recorded in audit_rounds), so only
    # genuinely open findings — unexplained anomalies and outstanding
    # rework requests — count as an unresolved accepted audit.
    if audit_result and audit_result.accepted:
        unresolved_audit = (
            bool(audit_result.unexplained_anomalies)
            or bool(audit_result.rework_requests)
        )
        if unresolved_audit:
            state.record_error(
                "audit",
                "global",
                "accepted audit contains unresolved findings",
            )
    return audit_result


async def apply_rework(
    ctx: WorkflowContext,
    case: Case,
    state: GraphState,
    requests: Sequence[ReworkRequest],
) -> tuple[bool, list[dict[str, Any]]]:
    """Re-run the seed/hop rechecks an audit round requested."""
    if not requests:
        return False, []
    rework_log: list[dict[str, Any]] = []
    changed = False
    inj_by_seed = {
        injection_node_id(inj): inj
        for inj in case.injections
        if inj.get("target")
    }

    seed_requests = [
        req for req in requests
        if isinstance(req, SeedRecheckRequest) and req.seed in inj_by_seed
    ]
    new_roots: list[str] = []
    if seed_requests:
        ctx.phase("audit-seed-rework")
        ctx.log(f"Audit requested {len(seed_requests)} seed rechecks")
        seed_result_items = normalize_parallel(await ctx.parallel([
            discovery.verify_seed(ctx, case, state, inj_by_seed[req.seed], req.context)
            for req in seed_requests
        ]))
        for idx, item in enumerate(seed_result_items):
            if is_seed_parallel_item(item):
                continue
            if idx < len(seed_requests):
                state.record_error(
                    "seed",
                    seed_requests[idx].seed,
                    "audit seed recheck failed before returning a result",
                )
        for req in seed_requests[len(seed_result_items):]:
            state.record_error(
                "seed",
                req.seed,
                "audit seed recheck was missing from parallel results",
            )
        seed_results = [
            item for item in seed_result_items
            if is_seed_parallel_item(item)
        ]
        missing_seed_results = len(seed_requests) - len(seed_results)
        if missing_seed_results > 0:
            ctx.log(
                f"{missing_seed_results} audit seed recheck task(s) "
                "returned no usable result"
            )
        for inj, seed_verdict in seed_results:
            seed_id = injection_node_id(inj)
            previous_seed_verdict = state.seed_verdicts.get(seed_id)
            if seed_verdict:
                state.seed_verdicts[seed_id] = seed_verdict
                if seed_verdict != previous_seed_verdict:
                    changed = True
            verdict = seed_verdict.get("verdict") if seed_verdict else "no-result"
            rework_log.append({
                "kind": "seed_recheck",
                "seed": seed_id,
                "verdict": verdict,
                "rationale": seed_verdict.get("rationale", "")
                if seed_verdict
                else "",
            })
            ctx.log(f"  seed recheck {seed_id}: {verdict}")
            if seed_verdict and verdict == "confirmed":
                before_roots = len(state.propagation_roots)
                accepted_seed = state.accept_seed_node(inj, seed_verdict)
                new_roots.extend(state.propagation_roots[before_roots:])
                state.confirmed_seed_ids.add(accepted_seed)
                state.clear_error("seed", accepted_seed)
                changed = True

    hop_requests = [
        req for req in requests
        if isinstance(req, HopRecheckRequest)
        and req.from_service in state.nodes
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
            prior = dict(state.verdicts.get(
                req.from_service + "__" + req.to_service,
                {},
            ))
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

        hop_result_items = normalize_parallel(await ctx.parallel([
            _one_hop_rework(req) for req in hop_requests
        ]))
        for idx, item in enumerate(hop_result_items):
            if is_hop_rework_parallel_item(item):
                continue
            if idx < len(hop_requests):
                hop_req = hop_requests[idx]
                state.record_error(
                    "hop",
                    hop_req.from_service + "__" + hop_req.to_service,
                    "audit hop recheck failed before returning a result",
                )
        for hop_req in hop_requests[len(hop_result_items):]:
            state.record_error(
                "hop",
                hop_req.from_service + "__" + hop_req.to_service,
                "audit hop recheck was missing from parallel results",
            )
        hop_results = [
            item for item in hop_result_items
            if is_hop_rework_parallel_item(item)
        ]
        missing_hop_results = len(hop_requests) - len(hop_results)
        if missing_hop_results > 0:
            ctx.log(
                f"{missing_hop_results} audit hop recheck task(s) "
                "returned no usable result"
            )
        for hop_req, result in hop_results:
            from_svc = hop_req.from_service
            to_svc = hop_req.to_service
            hop_verdict: str | None = (
                result.get("verdict") if isinstance(result, dict) else None
            )
            edge_key = from_svc + "__" + to_svc
            state.hop_log.append({
                "round": state.round_n,
                "from": from_svc,
                "to": to_svc,
                "verdict": (hop_verdict or "no-result") + "(audit-rework)",
            })
            if isinstance(result, dict) and hop_verdict:
                previous_hop_verdict = state.verdicts.get(edge_key)
                state.verdicts[edge_key] = result
                if result != previous_hop_verdict:
                    changed = True
            rework_log.append({
                "kind": "hop_recheck",
                "from": from_svc,
                "to": to_svc,
                "verdict": hop_verdict or "no-result",
                "rationale": result.get("rationale", "")
                if isinstance(result, dict)
                else "",
            })
            ctx.log(
                f"  hop recheck {from_svc} -> {to_svc}: "
                f"{hop_verdict or 'no-result'}"
            )
            if hop_verdict != "confirmed" or not isinstance(result, dict):
                continue
            was_new_node = to_svc not in state.nodes
            rel_type = next(
                (
                    info[1]
                    for info in case.graph.get(from_svc, [])
                    if info[0] == to_svc
                ),
                "callee_to_caller",
            )
            changed = state.accept_hop_result(
                from_svc,
                to_svc,
                rel_type,
                result,
                claim_override=hop_req.context[:200],
            ) or changed
            if was_new_node and to_svc not in case.infra_set and to_svc not in case.entry_services:
                new_roots.append(to_svc)
    if new_roots:
        ctx.phase("audit-propagate-rework")
        ctx.log(f"Propagating from {len(set(new_roots))} audit-confirmed roots")
        changed = await propagate(ctx, case, state, new_roots) or changed
    return changed, rework_log
