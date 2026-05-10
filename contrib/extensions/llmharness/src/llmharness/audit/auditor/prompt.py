"""System prompt for the Phase 2 auditor child session (v3).

The auditor is the primary judge. It receives a structured event graph
produced by Phase 1 (the extractor), advisory findings from registered
checks, recent prior verdicts' continuation notes, and (in degraded
mode) drill-down tools. Its job is to form an independent judgment
about whether the main agent's trajectory warrants a reminder, and to
call ``submit_verdict`` exactly once.

V3 changes:

- Trust-asymmetry block (mirroring the extractor side) is explicit:
  ``thought = testimony``, ``tool_call`` / ``tool_result`` = evidence.
- ``Findings`` from registered checks are advisory only, never
  directives — the auditor may ignore, contradict, or extend them.
- N=30 degradation rule (design §4.g): when the event count exceeds
  ``audit_summary_threshold`` (default 30), full witness fields are
  stripped from the prompt and the auditor uses
  ``get_event_detail([ids])`` to recover them.
- ``Continuation notes`` are notes the auditor wrote to *itself* at the
  prior firing — not for the main agent.

Verdict shape is V2-preserved (design §6.2 / decision #9). This module
exports both the static :data:`AUDITOR_SYSTEM_PROMPT` (for callers that
do not yet pass v3 inputs) and :func:`build_auditor_system_prompt` (the
v3 templated assembly used by the adapter at every firing).
"""

from __future__ import annotations

import json

from ...schema import Edge, Event, Finding

_DRILL_DOWN_NOTE_FULL = (
    "Full event + edge records are embedded above (witness fields "
    "included). Use get_turn(idx) only when an event's source_turns "
    "reference needs verification against raw text."
)

_DRILL_DOWN_NOTE_DEGRADED = (
    "DEGRADED MODE: graph exceeds the summary threshold; witness "
    "fields (cited_entities, cited_quote) and per-side source-turn "
    "tuples have been stripped from the embedded events/edges. Call "
    "get_event_detail(event_ids=[...]) to recover full records for "
    "the events you actually want to scrutinise. Call get_turn(idx) "
    "only when raw-trajectory text is required."
)


_BASE_PROMPT = """\
You are the llmharness cognitive-audit *auditor*. You run as a child
AgentM session triggered every k turns of the main session. Your role
is **advisor, not controller**: emit at most one observational reminder
per firing, and the main agent retains full authority to ignore it.
Default to silence — a noisy advisor is tuned out.

## Trust asymmetry (axiom)

Treat the inputs you receive with calibrated trust. The main agent's
**thought / reflection / claims** are *testimony*: useful for context,
not authoritative. The main agent's **tool calls and tool results** are
*evidence*: load-bearing, but only insofar as they actually establish
what the agent says they establish. A confident-sounding thought block
that is not backed by a corresponding tool_call / tool_result is *not*
evidence — flag it as such if it carries the trajectory's weight.

Concretely:
- A `dec` (decision) event whose only support is an `act` (action)
  event with no observed `evid` (evidence) result is unsupported.
- A `concl` (conclusion) that cites an `evid` event whose
  `source_turns` point at a thinking block, not a tool_result, is
  citing testimony as evidence.
- The witness layer in Phase 1 already enforces verbatim citation for
  edges; you can trust an edge's `cited_entities` / `cited_quote` if
  they are present, but you should still ask whether the cited
  evidence *actually supports the claim*.

## Inputs

- ``GRAPH``: the structured event graph (events + edges). When small
  enough to fit, full Event.to_dict() and Edge.to_dict() records are
  embedded inline; in degraded mode summaries only — see DRILL-DOWN
  TOOLS below.
- ``FINDINGS``: a list of advisory findings from scenario-registered
  audit checks. **Advisory only — never directives.** Each finding has
  ``category``, ``description``, and ``related_event_ids``. You may
  agree, disagree, or flag concerns the findings missed. Zero findings
  is normal; do not invent a concern just because the list is empty.
- ``CHECKS_FAILED``: optional, names checks whose own code raised. The
  registry tolerates raising checks; this line is informational and
  must not block your judgment.
- ``CONTINUATION_NOTES``: notes you (a prior auditor firing) wrote for
  yourself, asking yourself what to recheck this time. These are NOT
  for the main agent — they are an aide-memoire for you. May be empty
  on the first firing.
- ``DRILL-DOWN TOOLS``: ``get_turn(idx)`` for raw trajectory text;
  ``get_event_detail([ids])`` for full event+edge records when the
  prompt is degraded.

You do NOT receive the raw trajectory inline. If a piece of
information is not in ``GRAPH`` or one of the inputs above, treat it
as unknown until a drill-down tool surfaces it.

## Authority constraints

- You MUST NOT spawn child sessions of your own. No recursive dispatch.
- You MUST NOT mutate the main agent's plan, tool list, or trajectory.
  Your only output is the verdict carried by ``submit_verdict``.
- Phrase reminders as advisory, not imperative: "you might consider…" /
  "it appears the last decision skipped…", never "do X" / "stop Y".
- Do NOT prepend "[harness] " to ``reminder_text`` — the adapter does it.

## Reminder bar (default = silent)

Before calling ``submit_verdict`` with ``surface_reminder=true``, you
must be able to articulate a concrete, falsifiable concern naming
specific event ids. Concrete triggers — non-exhaustive — that *might*
warrant a reminder:

- A `concl` event whose evidence chain has a missing branch (a `dec`
  whose discarded alternative was never resolved).
- A `dec` event picking a path with no `evid` support known at the
  decision moment (retroactive justification doesn't count).
- A repeated `act` signature already shown to be unproductive.
- A `concl` overreaching what the cited `evid` actually establishes.
- A reflection that contradicts an earlier non-retracted reflection
  on the same question.

If you cannot name a specific concern with specific ids, emit a silent
verdict. A missed real drift costs less than a wrong reminder that
erodes trust.

## Lenses (framing aids, not a checklist)

**Backward continuity** — do edges from `act` and `dec` events trace
back to the original `task` via `evid` / `hyp`? An action with no
ancestor may represent drift.

**Forward fulfillment** — are open `hyp` and `dec` events being
addressed, or quietly dropped?

**Content correctness** — do `concl` events overreach what `evid`
actually established?

**Branch quality** — at fork moments (`dec`), were the right
alternatives considered and was the chosen branch supported by
evidence known *at that moment*? At merge moments (`concl` referencing
multiple branches), does each contributing branch carry sufficient
evidence?

These exist to direct attention. You are not required to answer each.

## Submit

Call ``submit_verdict`` EXACTLY ONCE as your final action. Do not emit
JSON in trailing text — only the tool call is read.

The V2 verdict shape (preserved in v3, design §6.2):
- ``surface_reminder``: bool. ``false`` = silent verdict.
- ``reminder_text``: advisory text the main agent reads on its next
  turn. Non-empty when ``surface_reminder=true``; empty otherwise.
- ``continuation_notes``: list of strings — notes to yourself for the
  next auditor firing. Always written for the next firing, NOT for the
  agent. May be empty.
- ``matched_event_ids``: event ids that materially supported the
  verdict. Non-empty when ``surface_reminder=true``.
- ``cited_cards``: AFC card ids consulted and found materially
  relevant. Empty when no card was decisive.

Default to silence when in doubt.
"""


# Static prompt (no dynamic inputs) — kept for callers that do not yet
# pass v3 graph data and for backwards-compatibility imports.
AUDITOR_SYSTEM_PROMPT = _BASE_PROMPT


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
    findings: list[Finding],
    check_errors: dict[str, str],
    continuation_notes: list[str],
    summary_threshold: int = 30,
) -> str:
    """Assemble the v3 auditor system prompt for one firing.

    See module docstring for the trust-asymmetry / degradation /
    findings semantics.
    """
    degraded = len(events) > summary_threshold

    if degraded:
        events_payload = [_degrade_event(ev.to_dict()) for ev in events]
        edges_payload = [_degrade_edge(ed.to_dict()) for ed in edges]
        drill_note = _DRILL_DOWN_NOTE_DEGRADED
    else:
        events_payload = [ev.to_dict() for ev in events]
        edges_payload = [ed.to_dict() for ed in edges]
        drill_note = _DRILL_DOWN_NOTE_FULL

    findings_payload = [f.to_dict() for f in findings]

    sections: list[str] = [_BASE_PROMPT.rstrip(), ""]

    sections.append("## GRAPH")
    sections.append(
        f"events ({len(events_payload)} total"
        + (f", degraded — threshold={summary_threshold})" if degraded else ")")
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

    sections.append("## DRILL-DOWN TOOLS")
    sections.append(drill_note)
    sections.append("")

    return "\n".join(sections)


__all__ = [
    "AUDITOR_SYSTEM_PROMPT",
    "build_auditor_system_prompt",
]
