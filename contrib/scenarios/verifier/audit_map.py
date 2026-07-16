"""Audit map — two passes + a free exploration sweep.

Each audit round runs three bounded agent fan-outs over the merged
evidence, and every one of them produces *re-dispatch requests* rather than
mutating the graph:

  - reachability — per confirmed seed: does it have a coherent
    causal path to an entry/SLO service? Invalid/unprovable paths emit a
    ``hop_recheck`` for the weakest edge; the gated re-verification decides
    whether the edge survives.
  - coverage — per entry/SLO scope: are its meaningful anomalies
    explained by the candidate graph? Gaps emit seed/hop rechecks.
  - Free exploration — one unconstrained agent over the whole dashboard:
    surface services/anomalies the targeted passes never investigated, and
    propose re-dispatch requests (incl. exploratory edges) to fill them.

The accept decision is NOT made here — the workflow's audit loop owns it as
a fixpoint (a round that produces no re-dispatch and no unexplained
anomalies). This module only produces requests and reports.
"""
from __future__ import annotations

from typing import Any

from agentm.extensions.builtin.workflow import AgentResult, WorkflowContext
from pydantic import BaseModel

from .audit.audit_context import build_audit_prompt
from .discovery import gate
from .lib.child import find_child_session
from .lib.retry import build_retry_context
from .lib.schema import (
    AnomalyCoverageReport,
    ExploreReport,
    ReworkRequest,
    SeedCoverageStatus,
    SeedReachabilityReport,
)
from .state import Case, GraphState


def _anomaly_service(record: dict[str, Any]) -> str | None:
    subject = str(record.get("subject", ""))
    if subject.startswith("svc:"):
        return subject.removeprefix("svc:")
    return None


def _coverage_scopes(case: Case) -> list[str]:
    scopes = set(case.entry_services)
    for record in case.anomaly_inventory:
        if record.get("status") != "changed":
            continue
        service = _anomaly_service(record)
        if service:
            scopes.add(service)
    return sorted(scopes) or ["<entry-services-not-discovered>"]


async def _agent(
    ctx: WorkflowContext,
    case: Case,
    *,
    label: str,
    role: str,
    instruction: str,
    payload: dict[str, Any],
    schema: type[BaseModel],
) -> BaseModel | None:
    prompt = build_audit_prompt(
        role=role,
        instruction=instruction,
        payload=payload,
    )
    result: AgentResult = await ctx.agent(
        prompt,
        scenario="verifier/audit",
        model=case.judge_model,
        schema=schema,
        atom_config={
            "duckdb_sql": {"data_dir": case.data_dir},
            "audit_context": {
                "role": role,
                "instruction": instruction,
                "payload": payload,
            },
        },
        retry=case.agent_retries,
        trace_label=label,
    )
    return result if isinstance(result, BaseModel) else None


async def _gated_agent(
    ctx: WorkflowContext,
    case: Case,
    state: GraphState,
    *,
    task_kind: str,
    label_base: str,
    role: str,
    instruction: str,
    payload: dict[str, Any],
    schema: type[BaseModel],
    gate_item: dict[str, Any],
) -> BaseModel | None:
    """Run one audit agent behind the same completeness gate that guards
    seed/hop discovery: submit → review its investigation → retry with
    focused feedback up to ``gate_retries`` times."""
    feedback = ""
    last: BaseModel | None = None
    for attempt in range(case.gate_retries + 1):
        label = f"{label_base}-a{attempt}"
        instr = instruction if not feedback else f"{instruction}\n\n{feedback}"
        result = await _agent(
            ctx,
            case,
            label=label,
            role=role,
            instruction=instr,
            payload=payload,
            schema=schema,
        )
        if result is None:
            return last
        last = result
        submitted = result.model_dump(mode="json")
        decision = await gate(
            ctx,
            case,
            task_kind=task_kind,
            label=label,
            task={
                "role": role,
                "instruction": instruction,
                "payload": payload,
                **gate_item,
            },
            submitted_result=submitted,
            child_session=find_child_session(ctx, label),
            attempt=attempt,
        )
        if decision is None:
            return last
        state.gate_log.append({
            "task": task_kind,
            "label": label,
            "gate": decision.model_dump(mode="json"),
        })
        if decision.accepted or attempt >= case.gate_retries:
            return last
        feedback = build_retry_context(
            base_context="",
            label=label,
            submitted_result=submitted,
            gate=decision,
        )
    return last


async def audit_pass_reachability(
    ctx: WorkflowContext,
    case: Case,
    state: GraphState,
    audit_round: int,
) -> tuple[dict[str, SeedCoverageStatus], list[ReworkRequest], list[dict[str, Any]]]:
    """Check whether every confirmed seed reaches an entry/SLO service."""
    graph_view = state.graph_snapshot()
    ledger_view = state.ledger_snapshot()
    case_view = case.case_summary()
    seeds = sorted(case.seeds)

    results = await ctx.parallel([
        _gated_agent(
            ctx,
            case,
            state,
            task_kind="reachability",
            label_base=f"audit-reach-{seed}-r{audit_round}",
            role="reachability",
            instruction=(
                "Audit whether this one seed has a coherent causal path to "
                "an entry/SLO service. Classify its coverage. When a "
                "candidate path is invalid or unprovable, do NOT assert a "
                "removal — emit a hop_recheck for the weakest edge so a "
                "gated re-verification can decide whether it survives."
            ),
            payload={
                "seed": seed,
                "case": case_view,
                "candidate_graph": graph_view,
                "evidence_ledger": ledger_view,
                "paths_from_seed": state.paths_from(seed),
                "anomaly_inventory": case.anomaly_inventory,
                "data_profile_context": case.profile_context_for_services(
                    set(case.entry_services) | {seed},
                ),
            },
            schema=SeedReachabilityReport,
            gate_item={"seed": seed},
        )
        for seed in seeds
    ])

    coverage: dict[str, SeedCoverageStatus] = {}
    requests: list[ReworkRequest] = []
    reports: list[dict[str, Any]] = []
    for seed, report in zip(seeds, results, strict=True):
        if not isinstance(report, SeedReachabilityReport):
            state.record_error(
                "audit",
                f"reachability:{seed}:r{audit_round}",
                "reachability audit task failed before returning a report",
            )
            continue
        coverage[report.seed] = report.coverage
        requests.extend(report.rework_requests)
        reports.append(report.model_dump(mode="json"))
    return coverage, requests, reports


async def audit_pass_coverage(
    ctx: WorkflowContext,
    case: Case,
    state: GraphState,
    audit_round: int,
) -> tuple[list[str], list[ReworkRequest], list[dict[str, Any]]]:
    """Check whether each meaningful anomaly is explained by a seed fault."""
    graph_view = state.graph_snapshot()
    ledger_view = state.ledger_snapshot()
    case_view = case.case_summary()
    scopes = _coverage_scopes(case)

    results = await ctx.parallel([
        _gated_agent(
            ctx,
            case,
            state,
            task_kind="coverage",
            label_base=f"audit-coverage-{scope}-r{audit_round}",
            role="coverage",
            instruction=(
                "Inspect this entry/SLO scope. Identify its meaningful "
                "abnormal trace/metric/log symptoms and decide whether the "
                "candidate graph explains each one. For every unexplained "
                "anomaly, emit a seed/hop recheck that would let an existing "
                "seed's fault reach this scope."
            ),
            payload={
                "scope": scope,
                "case": case_view,
                "candidate_graph": graph_view,
                "evidence_ledger": ledger_view,
                "anomaly_inventory": case.anomaly_inventory,
                "data_profile_context": case.profile_context_for_services({scope}),
            },
            schema=AnomalyCoverageReport,
            gate_item={"scope": scope},
        )
        for scope in scopes
    ])

    unexplained: list[str] = []
    requests: list[ReworkRequest] = []
    reports: list[dict[str, Any]] = []
    for scope, report in zip(scopes, results, strict=True):
        if not isinstance(report, AnomalyCoverageReport):
            state.record_error(
                "audit",
                f"coverage:{scope}:r{audit_round}",
                "coverage audit task failed before returning a report",
            )
            continue
        unexplained.extend(report.unexplained)
        requests.extend(report.rework_requests)
        reports.append(report.model_dump(mode="json"))
    return unexplained, requests, reports


async def free_explore(
    ctx: WorkflowContext,
    case: Case,
    state: GraphState,
    audit_round: int,
) -> tuple[list[ReworkRequest], dict[str, Any] | None]:
    """Run one free agent over the whole dashboard.

    Unlike the targeted seed/hop tasks, this agent is handed no single
    target — it surveys the full system dashboard for what the candidate
    graph fails to capture and proposes re-dispatch requests so each gap is
    verified by a gated discovery agent.
    """
    report = await _gated_agent(
        ctx,
        case,
        state,
        task_kind="explore",
        label_base=f"audit-explore-r{audit_round}",
        role="explore",
        instruction=(
            "You are NOT given a specific target. Survey the whole system "
            "dashboard for what the current graph fails to capture: services "
            "with clear degradation that are absent from the graph, entry "
            "anomalies no seed explains, propagation links resting on thin "
            "evidence. You may propose any seed/hop recheck (including edges "
            "not yet in the graph). You never edit the graph directly — every "
            "proposal is verified by a gated discovery agent. Be exhaustive "
            "about coverage; surface what has NOT been investigated."
        ),
        payload={
            "case": case.case_summary(),
            "candidate_graph": state.graph_snapshot(),
            "evidence_ledger": state.ledger_snapshot(),
            "entry_services": sorted(case.entry_services),
            "anomaly_inventory": case.anomaly_inventory,
            "data_profile_structure": case.data_profile.get("structure", {}),
        },
        schema=ExploreReport,
        gate_item={},
    )
    if not isinstance(report, ExploreReport):
        state.record_error(
            "audit",
            f"explore:r{audit_round}",
            "free-exploration audit task failed before returning a report",
        )
        return [], None
    return list(report.rework_requests), report.model_dump(mode="json")
