"""Inject per-edge fault-propagation context into the hop agent session.

Reads structured config describing one propagation edge (from_service,
to_service, faults, etc.), builds the full domain context, and appends
it to the system prompt so the agent starts with complete case-specific
knowledge.

The workflow orchestrator passes this data via ``atom_config`` — the
workflow script itself stays pure orchestration (BFS + parallel +
structured data), while all prompt/domain logic lives here in the
agent unit. The upstream's confirmed state arrives as an fpg EventNode
(strongly typed), so the evidence shown to the agent is exactly what
the scenario file will carry.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from agentm.core.abi import BeforeAgentStartEvent, ExtensionAPI
from agentm.extensions import ExtensionManifest
from fpg.scenario import EventNode


class TableProfile(BaseModel):
    """Mechanical profile of one normal/abnormal table pair."""

    model_config = ConfigDict(extra="ignore")
    columns: list[str] = Field(default_factory=list)
    # column -> window ("normal"/"abnormal") -> value -> count
    value_distributions: dict[str, dict[str, dict[str, int]]] = Field(
        default_factory=dict
    )


class HopContextConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    from_service: str
    to_service: str
    rel_type: str
    fault_kind: str
    injection_target: str
    # [fault_kind, target, params] triples for every injected fault
    all_faults: list[list[str]]
    fault_docs: dict[str, str] = Field(default_factory=dict)
    is_infra: bool = False
    # fpg EventNode of the confirmed upstream (structural layer; the
    # predicate is validated against the profile by the submit tool)
    upstream_evidence: EventNode | None = None
    # Mechanically profiled dataset shape (harness work, not LLM work):
    # table -> columns + low-cardinality value distributions per window
    dataset_profile: dict[str, TableProfile] = Field(default_factory=dict)


MANIFEST = ExtensionManifest(
    name="hop_context",
    description="Inject per-edge fault-propagation context into the hop agent.",
    registers=("event:before_agent_start",),
    config_schema=HopContextConfig,
)

# ---------------------------------------------------------------
# Fault doc loading
# ---------------------------------------------------------------

_FAULT_KINDS_DIR = Path(__file__).resolve().parent.parent / "fault_kinds"

_FAULT_DOC_ALIAS: Final = {
    "memorystress": "memstress",
    "jvmlatency": "jvmmethodlatency",
    "jvmexception": "jvmmethodexception",
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

def _format_dist(dist: dict[str, int], limit: int = 6) -> str:
    items = sorted(dist.items(), key=lambda kv: -kv[1])
    text = ", ".join(f"{v}={c}" for v, c in items[:limit])
    if len(items) > limit:
        text += f", … (+{len(items) - limit} more)"
    return text


def _format_dataset_profile(profile: dict[str, TableProfile]) -> str:
    """Render the mechanical dataset profile for the prompt."""
    lines: list[str] = []
    for base, tp in profile.items():
        lines.append(f"### {base} (tables normal_{base} / abnormal_{base})")
        lines.append("columns: " + ", ".join(tp.columns))
        for col, windows in tp.value_distributions.items():
            normal = _format_dist(windows.get("normal", {}))
            abnormal = _format_dist(windows.get("abnormal", {}))
            marker = (
                "  <-- CHANGED"
                if windows.get("normal", {}).keys()
                != windows.get("abnormal", {}).keys()
                else ""
            )
            lines.append(
                f"- {col}: normal [{normal}] | abnormal [{abnormal}]{marker}"
            )
        lines.append("")
    return "\n".join(lines).rstrip()


def _format_upstream_node(node: EventNode) -> str:
    lines = [f"Classified failure mode: {node.predicate}"]
    for ev in node.evidence:
        if ev.explanation:
            lines.append(f"- {ev.explanation}")
        if ev.query.statement:
            lines.append(f"  ```sql\n  {ev.query.statement}\n  ```")
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
    upstream_evidence: EventNode | None,
    dataset_profile: dict[str, TableProfile],
) -> str:
    rel_desc = _REL_DESCRIPTIONS.get(rel_type, "{frm} and {to} are related.")
    rel_text = rel_desc.format(frm=from_service, to=to_service)

    parts = [
        f"Confirmed degraded: **{from_service}**",
        f"Service to check: **{to_service}**",
        f"Relationship: {rel_text}",
        _fault_context(all_faults, to_service),
    ]
    if dataset_profile:
        parts.append(
            "\n## Dataset shape (mechanically profiled — authoritative for "
            "THIS dataset)\n"
            "Columns per table, and value distributions of every "
            "low-cardinality column in the normal vs abnormal window. Use "
            "this to decide WHERE signals live (which column carries error "
            "status, what level values logs use, which metrics exist) "
            "instead of assuming. A `<-- CHANGED` marker means new values "
            "appeared or vanished in the abnormal window — investigate "
            "whether they belong to the target.\n\n"
            + _format_dataset_profile(dataset_profile)
        )
    if upstream_evidence is not None:
        ev_text = _format_upstream_node(upstream_evidence)
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

def install(api: ExtensionAPI, config: HopContextConfig) -> None:
    if not config.from_service or not config.to_service:
        return

    all_faults = [
        (f[0], f[1], f[2] if len(f) > 2 else "")
        for f in config.all_faults
        if len(f) >= 2
    ]

    fault_docs: dict[str, str] = {}
    for fk, _, _ in all_faults:
        if fk in config.fault_docs:
            fault_docs[fk] = config.fault_docs[fk]
        elif fk not in fault_docs:
            fault_docs[fk] = _load_fault_doc(fk)

    context = _build_hop_prompt(
        from_service=config.from_service,
        to_service=config.to_service,
        rel_type=config.rel_type,
        fault_kind=config.fault_kind,
        injection_target=config.injection_target,
        all_faults=all_faults,
        fault_docs=fault_docs,
        is_infra=config.is_infra,
        upstream_evidence=config.upstream_evidence,
        dataset_profile=config.dataset_profile,
    )

    def before_agent_start(event: BeforeAgentStartEvent) -> None:
        current = str(event.system or "")
        event.system = f"{current}\n\n{context}" if current else context

    api.on(BeforeAgentStartEvent.CHANNEL, before_agent_start)

__all__: Final = ["MANIFEST", "install"]
