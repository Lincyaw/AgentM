"""Inject per-edge fault-propagation context into the hop agent session.

Reads structured config describing one propagation edge (from_service,
to_service, faults, etc.), builds the full domain context via
``prompt.build_hop_prompt``, and appends it to the system prompt so the
agent starts with complete case-specific knowledge.

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

from prompt import build_hop_prompt

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

    # Build fault docs: use provided docs, fill missing from disk
    fault_docs_cfg: dict[str, str] = config.get("fault_docs") or {}
    fault_docs: dict[str, str] = {}
    for fk, _, _ in all_faults:
        if fk in fault_docs_cfg:
            fault_docs[fk] = fault_docs_cfg[fk]
        elif fk not in fault_docs:
            fault_docs[fk] = _load_fault_doc(fk)

    context = build_hop_prompt(
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
