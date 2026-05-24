"""System prompt assembly for the Phase 2 auditor child session.

The framing text — trust-asymmetry, reminder bar, lenses, verdict
shape — lives in markdown files under :mod:`audit.auditor.prompts`.
Pick a variant by name (e.g. ``"minimal"``, ``"full"``) or by an
absolute path. The dynamic per-firing data (events / edges / phases /
findings / continuation notes) is appended on top of the chosen
framing inside :func:`build_auditor_system_prompt`.

Available named variants:

* ``minimal`` (default) — pairs with the ``minimal`` profile (only
  ``submit_verdict``). No drill-down references.
* ``full`` — pairs with the ``with_drill_down`` profile. Mentions
  ``get_event_detail`` / ``get_turn``.

Drop in a new variant by adding ``audit/auditor/prompts/auditor_<name>.md``
and pointing the adapter config at it; no code change needed.
"""

from __future__ import annotations

import json

from ...schema import Edge, Event, Finding, Phase
from .._prompt_loader import load_prompt

DEFAULT_PROMPT_NAME = "minimal"


def load_auditor_prompt(name_or_path: str = DEFAULT_PROMPT_NAME) -> str:
    """Load the auditor framing text for the given variant.

    Result is cached by :func:`_prompt_loader.load_prompt`, so repeated
    calls for the same name skip the disk read.
    """
    return load_prompt("auditor", name_or_path, filename_prefix="auditor")


def _degrade_event(ev_dict: dict[str, object]) -> dict[str, object]:
    return {
        "id": ev_dict.get("id"),
        "kind": ev_dict.get("kind"),
        "summary": ev_dict.get("summary"),
        "source_turns": ev_dict.get("source_turns", []),
    }


def _degrade_edge(ed_dict: dict[str, object]) -> dict[str, object]:
    return {
        "src": ed_dict.get("src"),
        "dst": ed_dict.get("dst"),
        "kind": ed_dict.get("kind"),
        "reason": ed_dict.get("reason"),
    }


def build_auditor_system_prompt(
    *,
    events: tuple[Event, ...],
    edges: tuple[Edge, ...],
    phases: tuple[Phase, ...] = (),
    findings: list[Finding],
    check_errors: dict[str, str],
    continuation_notes: list[str],
    summary_threshold: int = 30,
    base_prompt: str | None = None,
) -> str:
    """Assemble the auditor system prompt for one firing.

    ``base_prompt`` defaults to the ``minimal`` variant loaded via
    :func:`load_auditor_prompt`. The dynamic sections are appended
    after the framing in this order: PHASES (optional), GRAPH,
    FINDINGS, CONTINUATION_NOTES.

    Degrade behaviour is independent of the framing: when
    ``len(events) > summary_threshold``, witness fields are stripped
    from the embedded event / edge records and a ``degraded`` flag is
    surfaced in the GRAPH header. The framing file is expected to
    explain to the auditor what to do about it (e.g. the ``full``
    framing tells the auditor to use drill-down tools; the ``minimal``
    framing tells it to reason from what is embedded).
    """
    framing = (
        base_prompt
        if base_prompt is not None
        else load_auditor_prompt(DEFAULT_PROMPT_NAME)
    )
    degraded = len(events) > summary_threshold

    if degraded:
        events_payload = [_degrade_event(ev.to_dict()) for ev in events]
        edges_payload = [_degrade_edge(ed.to_dict()) for ed in edges]
    else:
        events_payload = [ev.to_dict() for ev in events]
        edges_payload = [ed.to_dict() for ed in edges]

    findings_payload = [f.to_dict() for f in findings]

    sections: list[str] = [framing.rstrip(), ""]

    if phases:
        sections.append("## PHASES (primary view — merged basic blocks)")
        sections.append(
            f"phases ({len(phases)} total). Each phase wraps one or more raw "
            "events; ``member_event_ids`` lists them in order. Consecutive "
            "``act`` events are coalesced into ``act_run`` blocks; "
            "``task`` / ``hyp`` / ``dec`` / ``concl`` always stay singleton. "
            "Reason at this level by default; consult the raw events block "
            "below when a specific witness needs verification."
        )
        sections.append(json.dumps([p.to_dict() for p in phases], ensure_ascii=False))
        sections.append("")

    sections.append("## GRAPH")
    sections.append(
        f"events ({len(events_payload)} total"
        + (
            f", degraded — threshold={summary_threshold}, witness fields stripped)"
            if degraded
            else ")"
        )
        + ":"
    )
    sections.append(json.dumps(events_payload, ensure_ascii=False))
    sections.append("")
    sections.append(f"edges ({len(edges_payload)} total):")
    sections.append(json.dumps(edges_payload, ensure_ascii=False))
    sections.append("")

    sections.append("## FINDINGS (advisory)")
    sections.append(json.dumps(findings_payload, ensure_ascii=False))
    if check_errors:
        sections.append(
            "checks_failed: "
            + json.dumps(check_errors, ensure_ascii=False)
            + " (non-blocking; other checks ran)"
        )
    sections.append("")

    sections.append("## CONTINUATION_NOTES (from your prior firing)")
    sections.append(json.dumps(list(continuation_notes), ensure_ascii=False))
    sections.append("")

    return "\n".join(sections)


__all__ = [
    "DEFAULT_PROMPT_NAME",
    "build_auditor_system_prompt",
    "load_auditor_prompt",
]
