"""Inject per-edge fault-propagation context into the hop agent session.

Reads structured config describing one propagation edge (from_service,
to_service, faults, etc.), builds the full domain context, and appends
it to the system prompt so the agent starts with complete case-specific
knowledge.

The workflow orchestrator passes this data via ``atom_config`` — the
workflow script itself stays pure orchestration (BFS + parallel +
structured data), while all prompt/domain logic lives here in the
agent unit.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from agentm.core.abi.events import BeforeAgentStartEvent
from agentm.core.abi.extension import ExtensionAPI
from agentm.extensions import ExtensionManifest

MANIFEST = ExtensionManifest(
    name="hop_context",
    description="Inject per-edge fault-propagation context into the hop agent.",
    registers=("event:before_agent_start",),
    config_schema={
        "type": "object",
        "properties": {
            "from_service": {"type": "string"},
            "to_service": {"type": "string"},
            "rel_type": {"type": "string"},
            "fault_kind": {"type": "string"},
            "injection_target": {"type": "string"},
            "all_faults": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "fault_docs": {
                "type": "object",
                "additionalProperties": {"type": "string"},
            },
            "is_infra": {"type": "boolean"},
            "upstream_evidence": {
                "type": ["object", "null"],
                "additionalProperties": True,
            },
        },
        "required": [
            "from_service", "to_service", "rel_type",
            "fault_kind", "injection_target", "all_faults",
        ],
        "additionalProperties": False,
    },
)

# ---------------------------------------------------------------
# Fault doc loading
# ---------------------------------------------------------------

_FAULT_KINDS_DIR = (
    Path(__file__).resolve().parents[1] / "verifier" / "fault_kinds"
)

_FAULT_DOC_ALIAS = {
    "memorystress": "memstress",
    "jvmlatency": "jvmmethodlatency",
    "podkill": "podfailure",
    "containerkill": "podfailure",
}


def _norm_fault(name: str) -> str:
    key = re.sub(r"[^a-z0-9]", "", name.lower())
    return _FAULT_DOC_ALIAS.get(key, key)


def _load_fault_doc(fault_kind: str) -> str:
    if not fault_kind:
        return ""
    p = _FAULT_KINDS_DIR / f"{fault_kind}.md"
    if p.is_file():
        return p.read_text().strip()
    target = _norm_fault(fault_kind)
    for doc in _FAULT_KINDS_DIR.glob("*.md"):
        if _norm_fault(doc.stem) == target:
            return doc.read_text().strip()
    return ""


# ---------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------

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


def _build_hop_prompt(
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


# ---------------------------------------------------------------
# Atom install
# ---------------------------------------------------------------

def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    from_service = config.get("from_service", "")
    to_service = config.get("to_service", "")
    if not from_service or not to_service:
        return

    all_faults_raw = config.get("all_faults", [])
    all_faults = [
        (f[0], f[1], f[2] if len(f) > 2 else "")
        for f in all_faults_raw
        if len(f) >= 2
    ]

    fault_docs_cfg: dict[str, str] = config.get("fault_docs") or {}
    fault_docs: dict[str, str] = {}
    for fk, _, _ in all_faults:
        if fk in fault_docs_cfg:
            fault_docs[fk] = fault_docs_cfg[fk]
        elif fk not in fault_docs:
            fault_docs[fk] = _load_fault_doc(fk)

    context = _build_hop_prompt(
        from_service=from_service,
        to_service=to_service,
        rel_type=config.get("rel_type", ""),
        fault_kind=config.get("fault_kind", ""),
        injection_target=config.get("injection_target", ""),
        all_faults=all_faults,
        fault_docs=fault_docs,
        is_infra=bool(config.get("is_infra", False)),
        upstream_evidence=config.get("upstream_evidence"),
    )

    def before_agent_start(event: BeforeAgentStartEvent) -> None:
        current = str(event.system or "")
        event.system = f"{current}\n\n{context}" if current else context

    api.on(BeforeAgentStartEvent.CHANNEL, before_agent_start)


__all__ = ["MANIFEST", "install"]
