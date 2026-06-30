"""Discovery agent adapters — seed/hop verification with gate retries.

These are the effectful "map" functions: each runs a child agent session
(via ``ctx.agent``), gate-audits the result, and retries with focused
feedback. They return verdicts; they never mutate the graph (the core
applies accepted verdicts through ``GraphState``). Execution errors and
the gate log are recorded into the shared ``GraphState`` ledger.
"""
from __future__ import annotations

from typing import Any, cast

from agentm.extensions.builtin.workflow import AgentResult, WorkflowContext

from .gate.gate_context import build_gate_prompt
from .hop.hop_context import PriorVerdict, build_hop_prompt
from .lib.child import find_child_session
from .lib.fpg import (
    injection_node_id,
    injection_subject,
    is_link_injection,
)
from .lib.final_checks import frontend_like
from .lib.retry import build_retry_context
from .lib.schema import GateDecision, HopResult, Injection, SeedResult
from .seed.seed_context import build_seed_prompt
from .state import Case, GraphState


def _label_part(value: str | None) -> str:
    """Make a compact trace-label fragment for source-specific child sessions."""
    if not value:
        return ""
    return "".join(ch if ch.isalnum() else "-" for ch in value)[:80]


async def gate(
    ctx: WorkflowContext,
    case: Case,
    *,
    task_kind: str,
    label: str,
    task: dict[str, Any],
    submitted_result: dict[str, Any],
    child_session: dict[str, Any] | None,
    attempt: int,
) -> GateDecision | None:
    gate_label = f"gate-{label}"
    prompt = build_gate_prompt(
        task_kind=task_kind,
        task=task,
        submitted_result=submitted_result,
        child_session=child_session,
        attempt=attempt,
    )
    gate_result: AgentResult = await ctx.agent(
        prompt,
        scenario="verifier/gate",
        model=case.judge_model,
        schema=GateDecision,
        tool_allowlist=["submit_result"],
        atom_config={
            "duckdb_sql": {"data_dir": case.data_dir},
            "gate_context": {
                "task_kind": task_kind,
                "task": task,
                "submitted_result": submitted_result,
                "child_session": child_session,
                "attempt": attempt,
            },
        },
        retry=case.agent_retries,
        trace_label=gate_label,
    )
    return gate_result if isinstance(gate_result, GateDecision) else None


async def verify_seed(
    ctx: WorkflowContext,
    case: Case,
    state: GraphState,
    inj: Injection,
    judge_context: str = "",
) -> tuple[Injection, SeedResult | None]:
    target = injection_node_id(inj) if is_link_injection(inj) else inj["target"]
    fault_kind = inj.get("chaos_type", "unknown")
    observation_surface = case.seed_observation_surfaces.get(
        injection_node_id(inj),
        {},
    )
    task = {
        "seed": injection_node_id(inj),
        "target": target,
        "fault_kind": fault_kind,
        "fault_reference_document": {
            "kind": fault_kind,
            "content": case.fault_docs.get(fault_kind, ""),
        },
        "params": inj.get("params", ""),
        "subject": injection_subject(inj),
        "observation_surface": observation_surface,
    }
    feedback = judge_context
    last_result: SeedResult | None = None
    for attempt in range(case.gate_retries + 1):
        label = f"seed-{injection_node_id(inj)}-a{attempt}"
        prompt = build_seed_prompt(
            target=target,
            fault_kind=fault_kind,
            params=inj.get("params", ""),
            fault_doc=case.fault_docs.get(fault_kind, ""),
            observation_surface=observation_surface,
            judge_context=feedback,
        )
        result: AgentResult = await ctx.agent(
            prompt,
            scenario="verifier/seed",
            atom_config={
                "duckdb_sql": {"data_dir": case.data_dir},
                "seed_context": {
                    "target": target,
                    "fault_kind": fault_kind,
                    "params": inj.get("params", ""),
                    "fault_doc": case.fault_docs.get(fault_kind, ""),
                    "observation_surface": observation_surface,
                    "judge_context": feedback,
                },
                "seed_finalize": {"data_dir": case.data_dir},
            },
            retry=case.agent_retries,
            trace_label=label,
        )
        if not isinstance(result, dict):
            reason = "seed agent returned no structured result"
            state.record_error("seed", injection_node_id(inj), reason)
            return inj, {
                "verdict": "inconclusive",
                "_error": reason,
                "rationale": reason,
            }
        if result.get("verdict") not in {
            "confirmed",
            "rejected",
            "inconclusive",
        }:
            reason = f"seed agent returned invalid verdict: {result.get('verdict')!r}"
            state.record_error("seed", injection_node_id(inj), reason)
            return inj, {
                "verdict": "inconclusive",
                "_error": reason,
                "rationale": reason,
            }
        last_result = cast(SeedResult, result)
        child_session = find_child_session(ctx, label)
        gate_decision = await gate(
            ctx,
            case,
            task_kind="seed",
            label=label,
            task=task,
            submitted_result=dict(last_result),
            child_session=child_session,
            attempt=attempt,
        )
        if gate_decision is None:
            reason = "seed gate returned no structured decision"
            state.record_error("seed", injection_node_id(inj), reason)
            last_result["verdict"] = "inconclusive"
            last_result["_error"] = reason
            last_result["rationale"] = (
                reason + ": " + str(last_result.get("rationale", ""))
            )
            return inj, last_result
        gate_payload = gate_decision.model_dump(mode="json")
        last_result["gate"] = gate_payload
        state.gate_log.append({
            "task": "seed",
            "label": label,
            "seed": injection_node_id(inj),
            "gate": gate_payload,
        })
        if gate_decision.accepted:
            state.clear_error("seed", injection_node_id(inj))
            return inj, last_result
        if attempt >= case.gate_retries:
            reason = "seed gate rejected all retry attempts"
            state.record_error("seed", injection_node_id(inj), reason)
            last_result["verdict"] = "inconclusive"
            last_result["_error"] = reason
            last_result["rationale"] = (
                reason + ": " + str(last_result.get("rationale", ""))
            )
            return inj, last_result
        feedback = build_retry_context(
            base_context=judge_context,
            label=label,
            submitted_result=dict(last_result),
            gate=gate_decision,
        )
    return inj, last_result


async def verify_hop(
    ctx: WorkflowContext,
    case: Case,
    state: GraphState,
    from_svc: str,
    to_svc: str,
    rel_type: str,
    *,
    judge_context: str = "",
    prior_verdict: PriorVerdict | None = None,
    source_seed: str | None = None,
    fault_record_override: list[str] | None = None,
    obligation_context: dict[str, Any] | None = None,
) -> HopResult | None:
    hop_fault = fault_record_override or state.fault_for_node(from_svc, source_seed)
    edge_fault_kind = hop_fault[0]
    edge_fault_docs: dict[str, str] = {}
    if edge_fault_kind in case.fault_docs:
        edge_fault_docs[edge_fault_kind] = case.fault_docs[edge_fault_kind]
    observation_context = case.profile_context_for_services({from_svc, to_svc})
    is_entry_target = frontend_like(to_svc, case.entry_services)
    task = {
        "from_service": from_svc,
        "to_service": to_svc,
        "rel_type": rel_type,
        "fault_kind": edge_fault_kind,
        "fault_reference_document": {
            "kind": edge_fault_kind,
            "content": case.fault_docs.get(edge_fault_kind, ""),
        },
        "is_infra": to_svc in case.infra_set,
        "is_entry_target": is_entry_target,
        "source_seed": source_seed,
        "upstream_evidence": state.nodes.get(from_svc),
        "fault_record": hop_fault,
        "observation_context": observation_context,
        "obligation": obligation_context or {},
        "prior_verdict": prior_verdict.model_dump(mode="json")
        if prior_verdict
        else {},
    }
    if judge_context:
        task["audit_context"] = judge_context
    feedback = judge_context
    last_result: HopResult | None = None
    for attempt in range(case.gate_retries + 1):
        source_part = _label_part(source_seed)
        label = f"hop-{from_svc}-to-{to_svc}"
        if source_part:
            label += f"-src-{source_part}"
        label += f"-a{attempt}"
        prompt = build_hop_prompt(
            from_service=from_svc,
            to_service=to_svc,
            rel_type=rel_type,
            fault_kind=edge_fault_kind,
            all_faults=[
                (f[0], f[1], f[2] if len(f) > 2 else "")
                for f in case.all_faults
                if len(f) >= 2
            ],
            fault_docs=edge_fault_docs,
            is_infra=to_svc in case.infra_set,
            is_entry_target=is_entry_target,
            upstream_evidence=state.nodes.get(from_svc),
            observation_context=observation_context,
            obligation=obligation_context,
            judge_context=feedback,
            prior_verdict=prior_verdict,
        )
        result: AgentResult = await ctx.agent(
            prompt,
            scenario="verifier/hop",
            atom_config={
                "duckdb_sql": {"data_dir": case.data_dir},
                "hop_context": {
                    "from_service": from_svc,
                    "to_service": to_svc,
                    "rel_type": rel_type,
                    "fault_kind": edge_fault_kind,
                    "all_faults": [
                        (f[0], f[1], f[2] if len(f) > 2 else "")
                        for f in case.all_faults
                        if len(f) >= 2
                    ],
                    "fault_docs": edge_fault_docs,
                    "is_infra": to_svc in case.infra_set,
                    "is_entry_target": is_entry_target,
                    "upstream_evidence": state.nodes.get(from_svc),
                    "source_seed": source_seed,
                    "observation_context": observation_context,
                    "obligation": obligation_context or {},
                    "judge_context": feedback,
                    "prior_verdict": prior_verdict.model_dump(mode="json")
                    if prior_verdict
                    else None,
                },
                "hop_finalize": {"data_dir": case.data_dir},
            },
            retry=case.agent_retries,
            trace_label=label,
        )
        if not isinstance(result, dict):
            reason = "hop agent returned no structured result"
            state.record_error("hop", from_svc + "__" + to_svc, reason)
            return {
                "verdict": "inconclusive",
                "_error": reason,
                "rationale": reason,
            }
        if result.get("verdict") not in {
            "confirmed",
            "rejected",
            "inconclusive",
        }:
            reason = f"hop agent returned invalid verdict: {result.get('verdict')!r}"
            state.record_error("hop", from_svc + "__" + to_svc, reason)
            return {
                "verdict": "inconclusive",
                "_error": reason,
                "rationale": reason,
            }
        last_result = cast(HopResult, result)
        child_session = find_child_session(ctx, label)
        gate_decision = await gate(
            ctx,
            case,
            task_kind="hop",
            label=label,
            task=task,
            submitted_result=dict(last_result),
            child_session=child_session,
            attempt=attempt,
        )
        if gate_decision is None:
            reason = "hop gate returned no structured decision"
            state.record_error("hop", from_svc + "__" + to_svc, reason)
            last_result["_error"] = reason
            return last_result
        gate_payload = gate_decision.model_dump(mode="json")
        last_result["gate"] = gate_payload
        state.gate_log.append({
            "task": "hop",
            "label": label,
            "from": from_svc,
            "to": to_svc,
            "source_seed": source_seed,
            "obligation": obligation_context or {},
            "gate": gate_payload,
        })
        if gate_decision.accepted:
            state.clear_error("hop", from_svc + "__" + to_svc)
            return last_result
        if attempt >= case.gate_retries:
            reason = "hop gate rejected all retry attempts"
            ctx.log(f"⚠ hop {from_svc}__{to_svc}: {reason}")
            last_result["verdict"] = "inconclusive"
            last_result["rationale"] = (
                reason + ": " + str(last_result.get("rationale", ""))
            )
            return last_result
        feedback = build_retry_context(
            base_context=judge_context,
            label=label,
            submitted_result=dict(last_result),
            gate=gate_decision,
        )
    return last_result
