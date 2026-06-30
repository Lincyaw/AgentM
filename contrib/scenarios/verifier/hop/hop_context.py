"""Prompt builder/config carrier for hop agents.

The workflow passes the built prompt as the child session's user
message, while atom_config keeps the scenario manifest/config shape
stable for verifier/hop as a callable map function.
"""

from __future__ import annotations

import json
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field

from agentm.core.abi import ExtensionAPI
from agentm.extensions import ExtensionManifest
from fpg.scenario import EventNode


class PriorVerdict(BaseModel):
    model_config = ConfigDict(extra="ignore")
    verdict: str = ""
    rationale: str = ""


class HopContextConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    from_service: str
    to_service: str
    rel_type: str
    fault_kind: str
    all_faults: list[tuple[str, str, str]] = Field(default_factory=list)
    fault_docs: dict[str, str] = Field(default_factory=dict)
    is_infra: bool = False
    is_entry_target: bool = False
    upstream_evidence: dict[str, Any] | None = None
    source_seed: str | None = None
    observation_context: dict[str, Any] = Field(default_factory=dict)
    obligation: dict[str, Any] = Field(default_factory=dict)
    judge_context: str = ""
    prior_verdict: PriorVerdict | None = None


MANIFEST = ExtensionManifest(
    name="hop_context",
    description="Builds structured single-hop verification prompts.",
    registers=(),
    config_schema=HopContextConfig,
)


# ---------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------

_REL_DESCRIPTIONS: Final = {
    "callee_to_caller": "Observed traces contain calls where {to} depends on "
    "{frm}. Verify whether the already-confirmed symptom on {frm} is followed "
    "by a statistically meaningful trace/metric/log change on {to}; do not "
    "confirm from topology alone.",
    "caller_to_callee": "Observed traces contain calls where {frm} sends work "
    "to {to}. Verify whether the behavior sent by {frm} produced a meaningful "
    "target-side trace/metric/log change on {to}; separate target degradation "
    "from simple reduced demand.",
    "co_deployed": "{frm} and {to} share runtime placement. Verify whether "
    "resource, restart, node, or saturation evidence connects their windows; "
    "do not assume co-location is causal.",
    "infra_dependency": "{frm} depends on the backing component {to} "
    "(database/cache/broker). Verify through client spans, resource metrics, "
    "logs, or peer attributes that are actually observable in this case.",
    "other": "{frm} and {to} were proposed by telemetry or audit rather than "
    "by a known structural edge. First establish whether an observable "
    "relationship exists, then test propagation evidence.",
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
    is_entry_target: bool = False,
    upstream_evidence: EventNode | dict | None = None,  # type: ignore[type-arg]
    observation_context: dict[str, Any] | None = None,
    obligation: dict[str, Any] | None = None,
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
    if is_entry_target:
        task_lines.append(
            f"`{to_service}` is a user-facing SLO/entry target. For this target, "
            "path-specific endpoint symptoms count as propagation: vanished or "
            "collapsed request volume for the affected endpoint, frontend/client "
            "latency or error changes, timeout-like spans, or other endpoint-level "
            "SLO changes aligned with the upstream path. Do not require the whole "
            "frontend process to show CPU/memory/log degradation. In multi-fault "
            "cases, unrelated frontend endpoints may also change; that does not "
            "disprove this edge if a concrete endpoint/anomaly on this path has "
            "its own aligned change. Reject only when the requested or affected "
            "endpoint has no signal beyond global drift or reduced demand. If "
            "frontend-wide volume also collapsed, do not overclaim span-count "
            "selectivity; compare endpoint ratios against the service total and "
            "use same-trace, endpoint-latency, error/timeout, terminal-alarm, or "
            "vanished-child-path evidence for the path-specific part."
        )
    sections.append("## Task\n" + "\n".join(task_lines))

    if observation_context:
        sections.append(
            "## Observation context\n"
            "This deterministic profile slice lists available modalities, "
            "service-level statistics, nearby relationships, and precomputed "
            "normal/abnormal anomaly candidates. Use it to plan checks; it is "
            "not causal evidence by itself.\n"
            "```json\n"
            + json.dumps(
                observation_context,
                indent=2,
                ensure_ascii=False,
                default=str,
            )
            + "\n```"
        )

    if obligation:
        sections.append(
            "## Final proof obligation\n"
            "This hop is being rechecked for a concrete final invariant. "
            "Answer this obligation, not only the generic service-to-service "
            "edge. The same edge may be relevant to one frontend symptom and "
            "irrelevant to another.\n"
            "```json\n"
            + json.dumps(obligation, indent=2, ensure_ascii=False, default=str)
            + "\n```"
        )

    # -- 4. Re-evaluation context (optional) --------------------------------
    if prior_verdict and prior_verdict.verdict:
        sections.append(
            f"## Re-evaluation (prior verdict: {prior_verdict.verdict})\n"
            f'You previously concluded: "{prior_verdict.rationale}"\n'
            f"The audit layer asks you to re-evaluate with the context below."
        )
    if judge_context:
        sections.append(
            "## Retry / audit feedback context\n"
            + judge_context
            + "\n\nUse this context to continue from the previous failed attempt. "
            "It summarizes prior submitted evidence and the gate/audit objections, "
            "but it is not a substitute for final evidence. Re-run or repair SQL "
            "as needed and submit a verdict only when the new evidence answers "
            "the missing checks."
        )

    return "\n\n".join(sections)


def install(api: ExtensionAPI, config: HopContextConfig) -> None:
    del api, config


__all__: Final = ["MANIFEST", "install", "PriorVerdict", "build_hop_prompt"]
