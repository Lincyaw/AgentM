"""Judge agent — independent verdict from verified evidence.

The judge has NO tools. It receives only the CompiledDossier (system-verified
SQL results) and the task description. It never sees the searcher's
interpretation or prior verdicts.
"""
from __future__ import annotations

import json

from .schema import CompiledDossier, JudgeVerdict, SQLResult, Verdict, VerificationTask
from .state import Case, GraphState


def build_judge_prompt(
    compiled: CompiledDossier,
    case: Case,
    state: GraphState,
    *,
    with_global_context: bool = True,
) -> str:
    """Build the judge agent's prompt from compiled evidence."""
    sections: list[str] = []
    task = compiled.task

    # Task description
    if task.kind == "seed":
        sections.append(
            "## Question\n"
            f"Did the injected fault take observable effect on **{task.from_entity}**?"
        )
    else:
        sections.append(
            "## Question\n"
            f"Did a fault propagate from **{task.from_entity}** to **{task.to_entity}**?"
        )

    # Global context (optional — omitted for voting diversity)
    if with_global_context and task.kind == "hop":
        sections.append(_global_context(case, state, task))

    # Verified evidence
    sections.append(_evidence_section("Relationship evidence", compiled.relationship_results))
    sections.append(_evidence_section("Target observations", compiled.target_results))
    sections.append(_evidence_section("Control observations", compiled.control_results))
    sections.append(_evidence_section("Counter-evidence", compiled.counter_results))

    # Searcher's relationship description (for context, not as trusted claim)
    if compiled.observed_relationship:
        sections.append(
            f"## Searcher's relationship description\n"
            f"(Not verified — use the SQL results above as ground truth)\n\n"
            f"{compiled.observed_relationship}"
        )

    # Coverage gaps
    if compiled.coverage_gaps:
        lines = ["## Coverage limitations"]
        for gap in compiled.coverage_gaps:
            lines.append(f"- [{gap.category}] {gap.description}")
        sections.append("\n".join(lines))

    # Instructions
    sections.append(_judge_instructions(task))

    return "\n\n".join(sections)


def compute_verdict(judge_verdict: JudgeVerdict, compiled: CompiledDossier) -> Verdict:
    """Compute the final verdict from the judge's answers."""
    return Verdict.from_judge(judge_verdict, compiled.affected_endpoints)


def majority_vote(verdicts: list[Verdict]) -> Verdict:
    """Take majority vote from multiple judge verdicts."""
    if not verdicts:
        return Verdict(kind="inconclusive", rationale="no verdicts")

    confirmed = [v for v in verdicts if v.kind == "confirmed"]
    rejected = [v for v in verdicts if v.kind == "rejected"]
    inconclusive = [v for v in verdicts if v.kind == "inconclusive"]

    if len(confirmed) > len(verdicts) / 2:
        return confirmed[0]
    if len(rejected) > len(verdicts) / 2:
        return rejected[0]
    # No majority → inconclusive (genuinely ambiguous)
    return Verdict(
        kind="inconclusive",
        rationale=f"No majority: {len(confirmed)} confirmed, "
        f"{len(rejected)} rejected, {len(inconclusive)} inconclusive",
    )


def _global_context(case: Case, state: GraphState, task: VerificationTask) -> str:
    """Provide graph context so the judge understands the causal chain."""
    lines = ["## System context"]
    lines.append(f"- Fault source seed: {task.source_seed}")

    # Show confirmed path from seed to from_entity
    path = _find_path(state, task.source_seed, task.from_entity)
    if path and len(path) > 1:
        lines.append(f"- Confirmed path: {' → '.join(path)}")

    # Show what fault kind
    for inj in case.injections:
        if (inj.get("node_id") or inj["target"]) == task.source_seed:
            lines.append(f"- Fault type: {inj['chaos_type']}")
            break

    # Entry services
    lines.append(f"- Entry/frontend services: {', '.join(sorted(case.entry_services))}")

    return "\n".join(lines)


def _evidence_section(title: str, results: list[SQLResult]) -> str:
    """Format verified SQL results for the judge."""
    if not results:
        return f"## {title}\n(none provided)"

    lines = [f"## {title}"]
    for r in results:
        status = f"✓ {r.row_count} rows" if r.success else f"✗ {r.error}"
        lines.append(f"\n### {r.location} [{status}]")
        lines.append(f"**Purpose**: {r.explanation}")
        lines.append(f"```sql\n{r.sql}\n```")
        if r.success and r.sample_values:
            lines.append("**Results** (first rows):")
            lines.append("```json")
            lines.append(json.dumps(r.sample_values[:3], indent=2, default=str))
            lines.append("```")

    return "\n".join(lines)


def _judge_instructions(task: VerificationTask) -> str:
    """Instructions tailored to the task type."""
    if task.kind == "seed":
        return """## Instructions

You are an independent judge. Answer based ONLY on the verified SQL results
above. The searcher's descriptions are context, not evidence.

Answer this single question:

**Did the injection take observable, selective effect?**
- "yes" = target shows significant change AND the change is selective
  (not proportional to control/system-wide patterns)
- "no" = target shows no meaningful change, OR change is proportional to
  control (workload shift, not fault effect)
- "insufficient_evidence" = can't determine from available data

For seeds, the causal_path question is automatically "yes" (injection is
the cause by definition). Focus on effect_aligned and selective.

If confirmed, classify the failure mode (predicate): latency_degraded,
error_rate_elevated, process_killed, network_partitioned, flow_interrupted,
etc."""

    return """## Instructions

You are an independent judge. Answer based ONLY on the verified SQL results
above. The searcher's descriptions are context, not evidence.

Answer three questions independently:

**1. Causal path**: Does a causal relationship exist between from and to?
- Can be: direct synchronous call, co-deployment resource contention,
  async messaging, shared infrastructure, temporal correlation
- "yes" = SQL results show an observable connection
- "no" = SQL results show NO connection (different pods, no calls, no
  shared resources)
- "insufficient_evidence" = can't determine

**2. Effect aligned**: Did the target change in the abnormal window?
- "yes" = clear metric/trace/log change aligned with fault timing
- "no" = no meaningful change observed
- "insufficient_evidence" = data not available or ambiguous

**3. Selective**: Is the target's change disproportionate to control?
- "yes" = target changed significantly MORE than control paths
- "no" = control paths show equal or greater change (workload shift)
- "insufficient_evidence" = no control comparison available

Consider counter-evidence seriously. If it shows the change is global or
timing doesn't align, that should influence your answers.

If all three are "yes", also provide a predicate (failure mode classification)."""


def _find_path(state: GraphState, src: str, dst: str) -> list[str]:
    """BFS to find the confirmed path from src to dst."""
    if src == dst:
        return [src]
    queue: list[tuple[str, list[str]]] = [(src, [src])]
    seen = {src}
    while queue:
        cur, path = queue.pop(0)
        for nxt in state.adj.get(cur, []):
            if nxt == dst:
                return path + [nxt]
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, path + [nxt]))
    return []
