"""Discovery agent adapters — seed/hop verification with gate retries.

These are the effectful "map" functions: each runs a child agent session
(via ``ctx.agent``), gate-audits the result, and retries with focused
feedback. They return verdicts; they never mutate the graph (the core
applies accepted verdicts through ``GraphState``). Execution errors and
the gate log are recorded into the shared ``GraphState`` ledger.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
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

_VALID_VERDICTS = {"confirmed", "rejected", "inconclusive"}


def _label_part(value: str | None) -> str:
    if not value:
        return ""
    return "".join(ch if ch.isalnum() else "-" for ch in value)[:80]


# ---------------------------------------------------------------------------
# Shared gated-investigation primitive
# ---------------------------------------------------------------------------

@dataclass
class GatedVerdict:
    """The outcome of one gated investigation (agent + gate retries)."""

    result: dict[str, Any]
    gate: GateDecision | None
    accepted: bool

    @property
    def verdict(self) -> str:
        return str(self.result.get("verdict", ""))


async def _run_gated(
    ctx: WorkflowContext,
    case: Case,
    state: GraphState,
    *,
    task_kind: str,
    task: dict[str, Any],
    error_id: str,
    label_stem: str,
    scenario: str,
    build_prompt: Callable[[str], str],
    build_atom_config: Callable[[str], dict[str, Any]],
    build_gate_log_entry: Callable[[str, dict[str, Any]], dict[str, Any]],
    initial_context: str = "",
) -> GatedVerdict:
    """Run an agent, gate its result, retry with focused feedback on rejection.

    This is the shared loop for both seed and hop verification. Callers
    provide closures for the parts that differ (prompt, atom_config,
    gate_log entry shape).
    """
    feedback = initial_context
    last_result: dict[str, Any] | None = None

    for attempt in range(case.gate_retries + 1):
        label = f"{label_stem}-a{attempt}"
        prompt = build_prompt(feedback)
        result: AgentResult = await ctx.agent(
            prompt,
            scenario=scenario,
            atom_config=build_atom_config(feedback),
            retry=case.agent_retries,
            trace_label=label,
        )

        if not isinstance(result, dict):
            reason = f"{task_kind} agent returned no structured result"
            state.record_error(task_kind, error_id, reason)
            return GatedVerdict(
                {"verdict": "inconclusive", "_error": reason, "rationale": reason},
                None,
                False,
            )

        if result.get("verdict") not in _VALID_VERDICTS:
            reason = f"{task_kind} agent returned invalid verdict: {result.get('verdict')!r}"
            state.record_error(task_kind, error_id, reason)
            return GatedVerdict(
                {"verdict": "inconclusive", "_error": reason, "rationale": reason},
                None,
                False,
            )

        last_result = dict(result)
        child_session = find_child_session(ctx, label)

        gate_decision = await gate(
            ctx, case,
            task_kind=task_kind, label=label, task=task,
            submitted_result=last_result, child_session=child_session,
            attempt=attempt,
        )

        if gate_decision is None:
            reason = f"{task_kind} gate returned no structured decision"
            state.record_error(task_kind, error_id, reason)
            last_result["_error"] = reason
            if task_kind == "seed":
                last_result["verdict"] = "inconclusive"
                last_result["rationale"] = (
                    reason + ": " + str(last_result.get("rationale", ""))
                )
            return GatedVerdict(last_result, None, False)

        gate_payload = gate_decision.model_dump(mode="json")
        last_result["gate"] = gate_payload
        state.gate_log.append(build_gate_log_entry(label, gate_payload))

        if gate_decision.accepted:
            state.clear_error(task_kind, error_id)
            return GatedVerdict(last_result, gate_decision, True)

        if attempt >= case.gate_retries:
            reason = f"{task_kind} gate rejected all retry attempts"
            if task_kind == "hop":
                ctx.log(f"⚠ hop {error_id}: {reason}")
            else:
                state.record_error(task_kind, error_id, reason)
            last_result["verdict"] = "inconclusive"
            last_result["rationale"] = (
                reason + ": " + str(last_result.get("rationale", ""))
            )
            return GatedVerdict(last_result, gate_decision, False)

        feedback = build_retry_context(
            base_context=initial_context,
            label=label,
            submitted_result=last_result,
            gate=gate_decision,
        )

    assert last_result is not None
    return GatedVerdict(last_result, None, False)


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


# ---------------------------------------------------------------------------
# Seed verification
# ---------------------------------------------------------------------------

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
        injection_node_id(inj), {},
    )
    seed_id = injection_node_id(inj)

    task = {
        "seed": seed_id,
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

    def _prompt(feedback: str) -> str:
        return build_seed_prompt(
            target=target,
            fault_kind=fault_kind,
            params=inj.get("params", ""),
            fault_doc=case.fault_docs.get(fault_kind, ""),
            observation_surface=observation_surface,
            judge_context=feedback,
        )

    def _atom_config(feedback: str) -> dict[str, Any]:
        return {
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
        }

    def _gate_log(label: str, gate_payload: dict[str, Any]) -> dict[str, Any]:
        return {"task": "seed", "label": label, "seed": seed_id, "gate": gate_payload}

    gated = await _run_gated(
        ctx, case, state,
        task_kind="seed",
        task=task,
        error_id=seed_id,
        label_stem=f"seed-{seed_id}",
        scenario="verifier/seed",
        build_prompt=_prompt,
        build_atom_config=_atom_config,
        build_gate_log_entry=_gate_log,
        initial_context=judge_context,
    )
    return inj, cast(SeedResult, gated.result) if gated.result else None


# ---------------------------------------------------------------------------
# Hop verification
# ---------------------------------------------------------------------------

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

    all_faults_tuples = [
        (f[0], f[1], f[2] if len(f) > 2 else "")
        for f in case.all_faults
        if len(f) >= 2
    ]

    task: dict[str, Any] = {
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

    error_id = from_svc + "__" + to_svc
    source_part = _label_part(source_seed)
    label_stem = f"hop-{from_svc}-to-{to_svc}"
    if source_part:
        label_stem += f"-src-{source_part}"

    def _prompt(feedback: str) -> str:
        return build_hop_prompt(
            from_service=from_svc,
            to_service=to_svc,
            rel_type=rel_type,
            fault_kind=edge_fault_kind,
            all_faults=all_faults_tuples,
            fault_docs=edge_fault_docs,
            is_infra=to_svc in case.infra_set,
            is_entry_target=is_entry_target,
            upstream_evidence=state.nodes.get(from_svc),
            observation_context=observation_context,
            obligation=obligation_context,
            judge_context=feedback,
            prior_verdict=prior_verdict,
        )

    def _atom_config(feedback: str) -> dict[str, Any]:
        return {
            "duckdb_sql": {"data_dir": case.data_dir},
            "hop_context": {
                "from_service": from_svc,
                "to_service": to_svc,
                "rel_type": rel_type,
                "fault_kind": edge_fault_kind,
                "all_faults": all_faults_tuples,
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
        }

    def _gate_log(label: str, gate_payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "task": "hop",
            "label": label,
            "from": from_svc,
            "to": to_svc,
            "source_seed": source_seed,
            "obligation": obligation_context or {},
            "gate": gate_payload,
        }

    gated = await _run_gated(
        ctx, case, state,
        task_kind="hop",
        task=task,
        error_id=error_id,
        label_stem=label_stem,
        scenario="verifier/hop",
        build_prompt=_prompt,
        build_atom_config=_atom_config,
        build_gate_log_entry=_gate_log,
        initial_context=judge_context,
    )
    return cast(HopResult, gated.result) if gated.result else None
