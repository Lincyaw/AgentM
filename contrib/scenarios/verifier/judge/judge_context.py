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
    confirmed_edges: Sequence[Mapping[str, Any]],
    entry_services: Sequence[str],
    unreachable_seeds: Sequence[str],
    seeds: set[str],
    seed_verdicts: Mapping[str, Mapping[str, Any]],
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
        f"- {i.get('subject') or i.get('node_id') or i['target']} ({i['chaos_type']})"
        for i in injections
        if i.get("target")
    ]
    sections.append("## Fault injection\n" + "\n".join(inj_lines))

    seed_lines: list[str] = []
    for seed in sorted(seeds):
        verdict = seed_verdicts.get(seed, {})
        status = verdict.get("verdict", "missing")
        predicate = verdict.get("predicate")
        rationale = verdict.get("rationale", "")
        suffix = f" ({predicate})" if predicate else ""
        seed_lines.append(f"- **{seed}**: {status}{suffix}. {rationale}")
    sections.append(
        f"## Seed verification results ({len(seed_lines)})\n"
        + ("\n".join(seed_lines) or "(none)")
        + "\n\nIf the confirmed graph is empty or an injection seed was rejected, "
        "still inspect entry/frontend symptoms directly. If those symptoms suggest "
        "a rejected or inconclusive seed may have been missed, request "
        "`re_evaluate_seeds` with concrete global context for the seed agent. "
        "If there is no meaningful entry anomaly or the seed's path was never "
        "exercised, leave `re_evaluate_seeds` empty and say so."
    )

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

    edge_lines = [
        f"- {e.get('src', '?')} → {e.get('dst', '?')}: {e.get('description', '')}"
        for e in confirmed_edges
    ]
    sections.append(
        f"## Confirmed causal edges ({len(confirmed_edges)})\n"
        + ("\n".join(edge_lines) or "(none)")
    )

    entry_lines = [f"- {svc}" for svc in sorted(entry_services)]
    unreachable_lines = [f"- {svc}" for svc in sorted(unreachable_seeds)]
    sections.append(
        "## Entry explanation audit\n"
        + "Entry services to inspect:\n"
        + ("\n".join(entry_lines) or "(none discovered)")
        + "\n\nConfirmed seeds without an entry path:\n"
        + ("\n".join(unreachable_lines) or "(none)")
        + "\n\nReverse-check whether the confirmed causal edges explain the entry services' abnormal-window symptoms. Query entry endpoint metrics directly; do not assume reachability alone is explanation."
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
