"""Build the user prompt for judge agents.

Pure function module — no atom, no event subscription. The workflow
imports :func:`build_judge_prompt` and passes the result as the user
message to ``ctx.agent(prompt=...)``.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def _ev_summary(verdict: Mapping[str, Any]) -> str:
    claims = [
        e.get("explanation", "")
        for e in verdict.get("evidence", [])
        if e.get("explanation")
    ]
    return "; ".join(claims[:4]) or "(none)"


def build_judge_prompt(
    *,
    injections: Sequence[Mapping[str, Any]],
    confirmed: Sequence[str],
    seeds: set[str],
    verdict_by_target: Mapping[str, Mapping[str, Any]],
    inconclusive_verdicts: Sequence[Mapping[str, Any]],
    rejected_verdicts: Sequence[Mapping[str, Any]],
) -> str:
    """Build the complete user prompt for a judge agent.

    Sections follow the same narrative pattern as hop_context:
    fault injection → confirmed graph → inconclusive → rejected.
    """
    sections: list[str] = []

    # -- Fault injection --
    inj_lines = [
        f"- {i['target']} ({i['chaos_type']})"
        for i in injections
        if i.get("target")
    ]
    sections.append("## Fault injection\n" + "\n".join(inj_lines))

    # -- Confirmed graph --
    confirmed_nonseed = [s for s in confirmed if s not in seeds]
    confirmed_lines: list[str] = [
        f"- **{s}** (injection seed)"
        for s in sorted(seeds)
        if s in confirmed
    ]
    for s in confirmed_nonseed:
        v = verdict_by_target.get(s)
        frm = v.get("from", "?") if v else "?"
        rationale = v.get("rationale", "(no rationale)") if v else "(no rationale)"
        confirmed_lines.append(
            f"- {frm} → **{s}**: {rationale}\n"
            f"    evidence: {_ev_summary(v) if v else '(none)'}"
        )
    sections.append(
        f"## Confirmed graph ({len(confirmed)} services)\n"
        + ("\n".join(confirmed_lines) or "(none)")
    )

    # -- Inconclusive edges --
    if inconclusive_verdicts:
        inc_lines = [
            f"- {v.get('from', '?')} → {v.get('to', '?')}: {v.get('rationale', '')}"
            for v in inconclusive_verdicts
        ]
        sections.append(
            f"## Inconclusive edges ({len(inconclusive_verdicts)})\n"
            + "\n".join(inc_lines)
        )

    # -- Rejected edges --
    if rejected_verdicts:
        rej_lines = [
            f"- {v.get('from', '?')} → {v.get('to', '?')}: {v.get('rationale', '')}"
            for v in rejected_verdicts
        ]
        sections.append(
            f"## Rejected edges ({len(rejected_verdicts)})\n"
            + "\n".join(rej_lines)
        )

    return "\n\n".join(sections)
