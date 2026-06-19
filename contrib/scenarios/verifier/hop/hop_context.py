"""Build the per-edge user prompt for hop agents.

Pure function module — no atom, no event subscription. The workflow
imports :func:`build_hop_prompt` and passes the result as the user
message to ``ctx.agent(prompt=...)``.
"""

from __future__ import annotations

from typing import Final

from pydantic import BaseModel, ConfigDict

from fpg.scenario import EventNode


class PriorVerdict(BaseModel):
    model_config = ConfigDict(extra="ignore")
    verdict: str = ""
    rationale: str = ""


# ---------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------

_REL_DESCRIPTIONS: Final = {
    "callee_to_caller": "{to} calls {frm}, so {frm} is {to}'s downstream "
    "dependency. A degraded callee propagates UP to its "
    "caller {to}, which blocks on or fails with the bad "
    "response. This is the usual direction for latency "
    "and error faults.",
    "caller_to_callee": "{frm} calls {to}, so {to} is {frm}'s downstream "
    "dependency. A caller affects its callee ONLY for "
    "data-corruption / bad-request faults (it sends {to} "
    "a wrong or corrupted request). A merely slow or "
    "failing caller does NOT by itself degrade {to} — be "
    "skeptical of confirming on this edge. If {frm} simply "
    "sends fewer calls, use that as evidence about {frm}'s "
    "interrupted path; do not confirm {to} unless {to} has "
    "its own alarm/user-visible path interruption or stronger "
    "timeout/error/fail-fast evidence.",
    "co_deployed": "{frm} and {to} share a k8s node — ONLY a node-level "
    "resource fault (CPU/memory/disk exhaustion) on one can "
    "degrade the other. An app-logic, JVM, or network fault "
    "does not cross to a co-located pod.",
    "infra_dependency": "{frm} depends on the backing component {to} "
    "(database/cache/broker). {to} is uninstrumented: it "
    "has NO spans of its own — its calls live inside {frm}.",
}


def _fault_context(
    all_faults: list[tuple[str, str, str]],
    edge_fault_kind: str,
) -> list[str]:
    primary = next(
        ((fk, tgt, params) for fk, tgt, params in all_faults if fk == edge_fault_kind),
        all_faults[0],
    )
    fk, tgt, params = primary
    suffix = f" ({params})" if params else ""
    lines = [f"- Fault on this propagation chain: {fk} on {tgt}{suffix}"]

    others = [(fk2, tgt2) for fk2, tgt2, _ in all_faults if fk2 != edge_fault_kind]
    if others:
        lines.append(
            "- Other faults in the system (evaluated on separate edges): "
            + ", ".join(f"{fk2} on {tgt2}" for fk2, tgt2 in others)
        )
    return lines


def _format_upstream_node(node: EventNode | dict) -> str:  # type: ignore[type-arg]
    if isinstance(node, dict):
        node = EventNode.model_validate(node)
    lines = [f"Classified failure mode: {node.predicate}"]
    for ev in node.evidence:
        if ev.explanation:
            lines.append(f"- {ev.explanation}")
        if ev.query.statement:
            lines.append(f"  ```sql\n  {ev.query.statement}\n  ```")
    return "\n".join(lines)


def build_hop_prompt(
    *,
    from_service: str,
    to_service: str,
    rel_type: str,
    fault_kind: str,
    all_faults: list[tuple[str, str, str]],
    fault_docs: dict[str, str],
    is_infra: bool = False,
    upstream_evidence: EventNode | dict | None = None,  # type: ignore[type-arg]
    judge_context: str = "",
    prior_verdict: PriorVerdict | None = None,
) -> str:
    """Build the complete user prompt for a hop agent.

    Keep the reference document separate from case state:
    fault reference → current state → task → (re-evaluation context).
    """
    sections: list[str] = []

    # -- 1. Fault reference document ----------------------------------------
    if fault_kind in fault_docs:
        sections.append(f"## Fault reference document: {fault_kind}\n{fault_docs[fault_kind]}")
    else:
        sections.append(f"## Fault reference document: {fault_kind}\n(no reference document available)")

    # -- 2. Current state: fault instance + upstream evidence ----------------
    state_lines = _fault_context(all_faults, fault_kind)
    state_lines.append(f"- Confirmed degraded upstream: **{from_service}**")
    if upstream_evidence is not None:
        ev_text = _format_upstream_node(upstream_evidence)
        if ev_text:
            state_lines.append("\nUpstream evidence:\n" + ev_text)
    sections.append("## Current state\n" + "\n".join(state_lines))

    # -- 3. Task: relationship + target -------------------------------------
    rel_desc = _REL_DESCRIPTIONS.get(rel_type, "{frm} and {to} are related.")
    rel_text = rel_desc.format(frm=from_service, to=to_service)
    task_lines = [
        f"Target: **{to_service}**",
        f"Relationship: {rel_text}",
        f"Task: verify whether the fault propagated from **{from_service}** to **{to_service}** on this single edge.",
    ]
    if is_infra:
        task_lines.append(
            f"`{to_service}` is an uninstrumented backing component — it "
            f"has no spans of its own. Verify via client spans inside "
            f"`{from_service}` (attr.span_kind = 'Client') and/or "
            f"`{to_service}`'s own resource metrics."
        )
    sections.append("## Task\n" + "\n".join(task_lines))

    # -- 4. Re-evaluation context (optional) --------------------------------
    if prior_verdict and prior_verdict.verdict:
        sections.append(
            f"## Re-evaluation (prior verdict: {prior_verdict.verdict})\n"
            f'You previously concluded: "{prior_verdict.rationale}"\n'
            f"The judge asks you to re-evaluate with the context below."
        )
    if judge_context:
        sections.append(f"## Judge's global context\n{judge_context}")

    return "\n\n".join(sections)
