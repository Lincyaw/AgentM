"""Audit map/reduce — the precision-and-closure layer.

One round fans out three bounded audit questions (anomaly coverage,
causal-path consistency, per-seed coverage) over the merged evidence, then
an audit reducer merges the reports into a ``GlobalAudit`` decision. The
reducer owns the accept/closure call; the core applies its drops and
rework. This module only produces reports and the decision.
"""
from __future__ import annotations

from typing import Any

from agentm.extensions.builtin.workflow import AgentResult, WorkflowContext
from pydantic import BaseModel

from .audit.audit_context import build_audit_prompt
from .lib.schema import (
    AnomalyAuditReport,
    CausalAuditReport,
    GlobalAudit,
    SeedCoverageReport,
)
from .state import Case, GraphState


def _dump_model(value: Any) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return value
    return {}


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


async def run_audit_round(
    ctx: WorkflowContext,
    case: Case,
    state: GraphState,
    audit_round: int,
) -> tuple[GlobalAudit | None, dict[str, Any]]:
    """Run one audit map/reduce round; return (decision, report bundle)."""
    graph_view = state.graph_snapshot()
    ledger_view = state.ledger_snapshot()
    case_view = case.case_summary()

    anomaly_scopes = sorted(case.entry_services) or ["<entry-services-not-discovered>"]
    anomaly_results = await ctx.parallel([
        _agent(
            ctx,
            case,
            label=f"audit-anomaly-{scope}-r{audit_round}",
            role="anomaly_coverage",
            instruction=(
                "Inspect this entry/service scope. Identify meaningful "
                "abnormal trace, metric, or log symptoms and decide "
                "whether the current candidate graph explains each one."
            ),
            payload={
                "scope": scope,
                "case": case_view,
                "candidate_graph": graph_view,
                "evidence_ledger": ledger_view,
            },
            schema=AnomalyAuditReport,
        )
        for scope in anomaly_scopes
    ])
    anomaly_reports = []
    for scope, report in zip(anomaly_scopes, anomaly_results):
        if report is None:
            state.record_error(
                "audit",
                f"anomaly:{scope}:r{audit_round}",
                "anomaly audit task failed before returning a report",
            )
            continue
        anomaly_reports.append(_dump_model(report))

    path_items = state.candidate_paths()
    causal_results = await ctx.parallel([
        _agent(
            ctx,
            case,
            label=f"audit-causal-{path_id}-r{audit_round}",
            role="causal_path",
            instruction=(
                "Audit whether this single seed-to-entry path is a "
                "coherent causal explanation. Reject paths that only "
                "borrow another seed's symptoms or are merely "
                "topologically reachable."
            ),
            payload={
                "path_id": path_id,
                "seed": seed,
                "path": path,
                "case": case_view,
                "candidate_graph": graph_view,
                "evidence_ledger": ledger_view,
            },
            schema=CausalAuditReport,
        )
        for path_id, seed, path in path_items
    ])
    causal_reports = []
    for (path_id, _, _), report in zip(path_items, causal_results):
        if report is None:
            state.record_error(
                "audit",
                f"causal:{path_id}:r{audit_round}",
                "causal audit task failed before returning a report",
            )
            continue
        causal_reports.append(_dump_model(report))

    audit_seeds = sorted(case.seeds)
    seed_audit_results = await ctx.parallel([
        _agent(
            ctx,
            case,
            label=f"audit-seed-{seed}-r{audit_round}",
            role="seed_coverage",
            instruction=(
                "Classify this seed as explains_entry, local_only, "
                "benign_or_no_effect, needs_recheck, or invalid_path."
            ),
            payload={
                "seed": seed,
                "case": case_view,
                "candidate_graph": graph_view,
                "evidence_ledger": ledger_view,
                "paths_from_seed": state.paths_from(seed),
                "causal_reports": [
                    report for report in causal_reports
                    if str(report.get("path_id", "")).startswith(seed + ":")
                ],
                "anomaly_reports": anomaly_reports,
            },
            schema=SeedCoverageReport,
        )
        for seed in audit_seeds
    ])
    seed_coverage_reports = []
    for seed, report in zip(audit_seeds, seed_audit_results):
        if report is None:
            state.record_error(
                "audit",
                f"seed:{seed}:r{audit_round}",
                "seed coverage audit task failed before returning a report",
            )
            continue
        seed_coverage_reports.append(_dump_model(report))

    reducer = await _agent(
        ctx,
        case,
        label=f"audit-reducer-r{audit_round}",
        role="audit_reducer",
        instruction=(
            "Merge the audit reports and OWN the closure decision: the "
            "harness applies your drop_edges and records your "
            "invalid_causal_paths, but performs no post-hoc force-accept. "
            "Accept only when all meaningful anomalies are explained or "
            "resolved and every seed has a resolved coverage status "
            "(explains_entry / local_only / benign_or_no_effect). "
            "Otherwise emit concrete rework requests and/or "
            "path-specific edge drops. When this is the final round, do "
            "not leave a seed at needs_recheck: classify a seed whose "
            "candidate paths are invalid or unprovable, and whose "
            "fault-aligned path has no matching unexplained anomaly, as "
            "benign_or_no_effect so the audit can close."
        ),
        payload={
            "round": audit_round,
            "final_round": audit_round >= case.max_audit_rounds - 1,
            "case": case_view,
            "candidate_graph": graph_view,
            "evidence_ledger": ledger_view,
            "anomaly_reports": anomaly_reports,
            "causal_reports": causal_reports,
            "seed_coverage_reports": seed_coverage_reports,
        },
        schema=GlobalAudit,
    )
    audit_result = reducer if isinstance(reducer, GlobalAudit) else None
    if audit_result is None:
        state.record_error(
            "audit",
            f"reducer:r{audit_round}",
            "audit reducer returned no structured decision",
        )
    ctx.log(
        "  audit reducer: "
        + ("accepted" if audit_result and audit_result.accepted else "needs rework")
    )

    reports = {
        "anomaly_reports": anomaly_reports,
        "causal_reports": causal_reports,
        "seed_coverage_reports": seed_coverage_reports,
    }
    return audit_result, reports
