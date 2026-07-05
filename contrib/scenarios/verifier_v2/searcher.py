"""Searcher agent — evidence collection without judgment.

The searcher has DuckDB SQL access. Its job is to find all relevant evidence
(for AND against propagation) and submit a structured EvidenceDossier.
It does NOT decide a verdict.
"""
from __future__ import annotations

import json
from typing import Any

from .schema import TaskAttempt, VerificationTask
from .state import Case, GraphState


def build_searcher_prompt(
    case: Case,
    task: VerificationTask,
    history: list[TaskAttempt],
    coverage_feedback: str = "",
    state: GraphState | None = None,
) -> str:
    """Build the searcher agent's prompt."""
    sections: list[str] = []

    # Fault reference
    fault_kind = _fault_kind_for_task(case, task)
    fault_doc = case.fault_docs.get(fault_kind, "")
    if fault_doc:
        sections.append(f"## Fault reference: {fault_kind}\n{fault_doc}")

    # Task description
    if task.kind == "seed":
        sections.append(_seed_task_section(case, task))
    else:
        sections.append(_hop_task_section(case, task))

    # Observation context (data profile slice)
    profile_context = _profile_context(case, task)
    if profile_context:
        sections.append(
            "## Available data\n```json\n"
            + json.dumps(profile_context, indent=2, ensure_ascii=False, default=str)[:2000]
            + "\n```"
        )

    # Multi-fault context: what the searcher needs to know about other faults
    multi_fault_ctx = _multi_fault_context(case, state, task)
    if multi_fault_ctx:
        sections.append(multi_fault_ctx)

    # Time windows
    sections.append(
        "## Time windows\n"
        f"- Normal: {case.window.get('start', '?')} (before injection)\n"
        f"- Abnormal: {case.window.get('end', '?')} (during injection)"
    )

    # Retry history
    if history:
        sections.append(_history_section(history))

    # Coverage feedback from compiler
    if coverage_feedback:
        sections.append(f"## Coverage feedback (from previous attempt)\n{coverage_feedback}")

    # Instructions
    sections.append(_instructions())

    return "\n\n".join(sections)


def _seed_task_section(case: Case, task: VerificationTask) -> str:
    fault_kind = _fault_kind_for_task(case, task)
    inj = _injection_for_seed(case, task.source_seed)
    params = inj.get("params", "") if inj else ""
    target = task.from_entity

    lines = [
        "## Task: Verify injection effect",
        f"- Fault: **{fault_kind}** on **{target}**",
    ]
    if params:
        lines.append(f"- Parameters: {params}")
    if target.startswith("link:") and "->" in target:
        link_body = target.removeprefix("link:")
        src_svc, dst_svc = link_body.split("->", 1)
        lines.extend([
            f"- This is a link/network fault between **{src_svc}** and **{dst_svc}**.",
            f"- You MUST verify the call relationship between {src_svc} and {dst_svc}:",
            "  - Use parent_span_id joins to find normal calls between them",
            f"  - Check both directions ({src_svc}→{dst_svc} AND {dst_svc}→{src_svc})",
            "  - In abnormal window: look for caller timeouts, missing child spans,",
            "    latency spikes, or connection errors on the link endpoints",
            "  - The target service's own spans may remain healthy while callers",
            "    show timeout/error symptoms across the degraded link",
        ])
    lines.append(
        f"\nDetermine whether this injection produced observable effects on {target}."
    )
    return "\n".join(lines)


def _hop_task_section(case: Case, task: VerificationTask) -> str:
    fault_kind = _fault_kind_for_task(case, task)
    lines = [
        "## Task: Investigate propagation edge",
        f"- Upstream (confirmed degraded): **{task.from_entity}**",
        f"- Target (to investigate): **{task.to_entity}**",
        f"- Fault on this chain: {fault_kind}",
        f"- Relationship type: {task.rel_type}",
        "",
        f"Collect evidence about whether the fault propagated from "
        f"{task.from_entity} to {task.to_entity}.",
    ]
    if task.context:
        lines.append(f"\nAdditional context: {task.context}")
    return "\n".join(lines)


def _instructions() -> str:
    return """## Instructions

Your role is **evidence searcher**, not judge. Collect all relevant evidence
and submit it. Do NOT decide a verdict — a separate judge will interpret
your findings.

1. **Discover schema**: List available tables, inspect columns and value
   distributions relevant to the target services.

2. **Relationship evidence**: Find SQL evidence showing how from_entity and
   to_entity are related (direct calls, shared resources, co-deployment,
   async messaging, or any other observable connection).

3. **Target observations**: Query traces/metrics/logs for the target entity.
   Compare normal vs abnormal windows. Report what changed (or didn't).

4. **Control observations**: Query the SAME metrics on comparison paths that
   are NOT on the fault propagation chain (sibling endpoints, unaffected
   callers, same service on different pods).

5. **Counter-evidence** (REQUIRED): Actively search for reasons the observed
   change might NOT be caused by the upstream fault:
   - Does the control path show a proportional change? (workload shift)
   - Is there another fault in the system affecting this target?
   - Is the timing misaligned?
   - Is the target service simply not exercised during this window?

6. **Endpoint granularity**: If the fault affects specific endpoints rather
   than the whole service, report which endpoints in affected_endpoints.

Submit your findings as a structured EvidenceDossier. Be thorough — check
all available modalities (traces, metrics, logs). Missing modalities should
be listed in modalities_unavailable with the reason."""


def _history_section(history: list[TaskAttempt]) -> str:
    lines = ["## Previous attempts"]
    for attempt in history:
        lines.append(f"\n### Attempt {attempt.attempt_n}")
        if attempt.coverage_gaps:
            lines.append("Coverage gaps: " + "; ".join(attempt.coverage_gaps))
        if attempt.verdict_kind:
            lines.append(f"Judge verdict: {attempt.verdict_kind}")
        if attempt.judge_rationale:
            lines.append(f"Judge rationale: {attempt.judge_rationale}")
        if attempt.sql_summary:
            lines.append("SQL already tried:\n- " + "\n- ".join(attempt.sql_summary[:10]))
    lines.append("\nDo NOT repeat the same SQL queries. Try different approaches.")
    return "\n".join(lines)


def _profile_context(case: Case, task: VerificationTask) -> dict[str, Any]:
    """Extract relevant data profile slice for the task's services."""
    services = {task.from_entity, task.to_entity} - {""}
    structure = case.data_profile.get("structure", {})
    stats = case.data_profile.get("statistics", {})

    context: dict[str, Any] = {}
    available_services = structure.get("services", [])
    if available_services:
        context["known_services"] = [
            s for s in available_services if s in services
        ]

    for modality in ("traces", "logs", "metrics"):
        mod_stats = stats.get(modality, {}).get("services", {})
        relevant = {
            svc: data for svc, data in mod_stats.items()
            if svc in services
        }
        if relevant:
            context[f"{modality}_stats"] = relevant

    return context


def _multi_fault_context(
    case: Case,
    state: GraphState | None,
    task: VerificationTask,
) -> str:
    """Build context about other confirmed faults in the system.

    This is critical for multi-fault cases: the searcher needs to know that
    another fault exists so it can distinguish this seed's effect from the
    other fault's propagation.
    """
    if state is None:
        return ""
    if not state.confirmed_seeds:
        return ""
    # Only relevant if there are OTHER confirmed seeds (not the one we're verifying)
    other_seeds = state.confirmed_seeds - {task.source_seed}
    if not other_seeds:
        return ""

    lines = ["## Multi-fault context"]
    lines.append(
        "This system has MULTIPLE injected faults. The following faults are "
        "already confirmed and their propagation paths are known. You must "
        "distinguish THIS seed's effect from the effects of these other faults."
    )

    lines.append("")
    lines.append("**Already confirmed faults:**")
    for seed in sorted(other_seeds):
        inj = _injection_for_seed(case, seed)
        fault_kind = inj["chaos_type"] if inj else "unknown"
        lines.append(f"- `{seed}` ({fault_kind})")

    # Show which services are affected by the other faults
    affected = set()
    for node_id, node_meta in state.nodes.items():
        sources = state.node_sources.get(node_id, set())
        if sources & other_seeds:
            affected.add(node_id)
    if affected:
        lines.append("")
        lines.append(
            "**Services already degraded by the other fault(s):** "
            + ", ".join(sorted(affected))
        )

    lines.append("")
    lines.append("**Investigation guidance for this seed:**")
    lines.append(
        "- Traffic drops on this seed's target might be CAUSED by the other "
        "fault (upstream killed traffic), not by this seed's injection."
    )
    lines.append(
        "- Look for effects SPECIFIC to this seed that the other fault "
        "cannot explain: different endpoints, different error patterns, "
        "different timing, or services NOT on the other fault's path."
    )
    lines.append(
        "- If this seed's target has zero traffic in abnormal AND the "
        "other fault's propagation path goes through its upstream callers, "
        "the zero traffic may be a consequence of the other fault — not "
        "confirmation of this seed."
    )
    lines.append(
        "- Conversely, if this seed's target shows degradation on endpoints "
        "that are NOT called by services on the other fault's path, that IS "
        "evidence of this seed's independent effect."
    )

    return "\n".join(lines)


def _fault_kind_for_task(case: Case, task: VerificationTask) -> str:
    inj = _injection_for_seed(case, task.source_seed)
    return inj["chaos_type"] if inj else "unknown"


def _injection_for_seed(case: Case, seed: str) -> dict[str, str] | None:
    for inj in case.injections:
        if (inj.get("node_id") or inj["target"]) == seed:
            return inj
    return None
