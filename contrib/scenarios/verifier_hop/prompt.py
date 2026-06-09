"""Hop agent prompt construction for the verifier_hop scenario."""

from __future__ import annotations

_REL_DESCRIPTIONS = {
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
                        "skeptical of confirming on this edge.",
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
    to_service: str,
) -> str:
    """One line for a single fault; a list, with params, when several coexist."""
    if len(all_faults) <= 1:
        fk, tgt, params = all_faults[0]
        suffix = f" ({params})" if params else ""
        return f"Fault injected: {fk} on {tgt}{suffix}"
    lines = [f"Faults injected in this system ({len(all_faults)}):"]
    for fk, tgt, params in all_faults:
        suffix = f" ({params})" if params else ""
        lines.append(f"- {fk} on {tgt}{suffix}")
    lines.append(
        f"\n{to_service} may sit downstream of any of these. Each fault's "
        f"category and intensity predicts a specific fingerprint — read "
        f"each fault reference below and check for the signal it predicts. "
        f"Do not assume a single fault is responsible."
    )
    return "\n".join(lines)


def _format_upstream_evidence(evidence: dict) -> str:
    """Format upstream node evidence for the hop agent prompt."""
    lines: list[str] = []
    src = evidence.get("source", "")
    if src == "injection_target":
        n_ms = evidence.get("normal_avg_ms")
        a_ms = evidence.get("abnormal_avg_ms")
        ratio = evidence.get("ratio")
        if n_ms is not None and a_ms is not None:
            lines.append(
                f"Avg latency: normal {n_ms:.1f}ms → abnormal {a_ms:.1f}ms "
                f"({ratio}x)"
            )
    elif src == "hop_agent":
        rationale = evidence.get("rationale")
        if rationale:
            lines.append(f"Rationale: {rationale}")
        for ev in evidence.get("symptom_evidence", []):
            claim = ev.get("claim", "")
            sql = ev.get("sql", "")
            if claim:
                lines.append(f"- {claim}")
            if sql:
                lines.append(f"  ```sql\n  {sql}\n  ```")
    return "\n".join(lines)


def build_hop_prompt(
    from_service: str,
    to_service: str,
    rel_type: str,
    fault_kind: str,
    injection_target: str,
    all_faults: list[tuple[str, str, str]],
    fault_docs: dict[str, str],
    is_infra: bool,
    upstream_evidence: dict | None,
) -> str:
    """Build the full user prompt for a single hop verification agent."""
    rel_desc = _REL_DESCRIPTIONS.get(rel_type, "{frm} and {to} are related.")
    rel_text = rel_desc.format(frm=from_service, to=to_service)

    parts = [
        f"Confirmed degraded: **{from_service}**",
        f"Service to check: **{to_service}**",
        f"Relationship: {rel_text}",
        _fault_context(all_faults, to_service),
    ]
    if upstream_evidence:
        ev_text = _format_upstream_evidence(upstream_evidence)
        if ev_text:
            parts.append(
                f"\n## Observed symptoms on {from_service}\n{ev_text}\n\n"
                f"This is only a partial picture of the upstream's "
                f"degradation. Look for **different signals** on "
                f"{to_service} — do not just repeat the same queries. "
                f"The propagation may manifest differently on the "
                f"downstream (e.g. errors vs latency vs missing spans)."
            )
    # Show every injected fault's doc (primary first, deduped by kind).
    shown: set[str] = set()
    ordered = [fault_kind] + [fk for fk, _, _ in all_faults if fk != fault_kind]
    for fk in ordered:
        if fk in shown:
            continue
        shown.add(fk)
        doc = fault_docs.get(fk)
        if doc:
            parts.append(f"\n## Fault reference ({fk})\n{doc}")
    if is_infra:
        parts.append(
            f"\n## {to_service} is an uninstrumented backing component\n"
            f"`{to_service}` has NO spans of its own — `service_name = "
            f"'{to_service}'` returns nothing in *_traces. Verify it via:\n"
            f"- (A) the Client DB/cache spans **inside {from_service}**: "
            f"`WHERE service_name = '{from_service}' AND "
            f"\"attr.span_kind\" = 'Client'` with SQL/ORM span_name shapes "
            f"(SELECT/INSERT/UPDATE/DELETE/Transaction/Session/%Repository%). "
            f"Compare normal vs abnormal latency and error rate.\n"
            f"- (B) `{to_service}`'s own resource metrics: `*_metrics` tables "
            f"`WHERE service_name = '{to_service}'`.\n"
            f"The component is degraded ONLY if (B) its own metrics worsen, "
            f"or its DB/cache spans error/slow across MULTIPLE independent "
            f"callers. A single caller's slow or failing client spans is that "
            f"caller's egress problem — especially under a fault that lives on "
            f"`{from_service}` (a JVM/JDBC fault, or a `tc netem` delay/loss "
            f"that slows ALL of {from_service}'s packets). Do NOT count "
            f"`{to_service}` degraded from `{from_service}`'s client spans "
            f"alone — that double-counts {from_service}'s own degradation."
        )
    parts.append(
        f"\nDetermine whether {to_service} is genuinely degraded due to "
        f"this relationship with {from_service}. Query normal_* vs "
        f"abnormal_* tables, verify the relationship, then submit."
    )
    return "\n".join(parts)


__all__ = ["build_hop_prompt"]
