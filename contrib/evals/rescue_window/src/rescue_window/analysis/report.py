"""Report generator (doc §9.1, §4.3, §10).

Pure function over rows + aggregate + windows. Produces dataset-level metrics
(opportunity prevalence, rescue/harm, window summary), per-prefix taxonomy
labels (doc §10, derived not annotated), and a markdown summary.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from ..model import ContentLevel, EvalUnit
from .aggregate import AggregateResult, PrefixStat
from .ladder import PrefixLadder
from .window import WindowMetrics


def build_report(
    rows: list[EvalUnit],
    result: AggregateResult,
    windows: list[WindowMetrics],
    *,
    ladders: list[PrefixLadder] | None = None,
) -> dict[str, Any]:
    eps = result.epsilon
    prefixes = result.prefixes
    failing = [p for p in prefixes if p.continue_success is False]
    ladder_by_prefix = {pl.prefix_id: pl for pl in (ladders or [])}

    rescue_rate, harm_rate, n_rescue_denom, n_harm_denom = _binary_rescue_harm(rows, prefixes)

    summary = {
        "epsilon": eps,
        "n_prefixes": len(prefixes),
        "n_trajectories": len({p.trajectory_id for p in prefixes}),
        "n_cases": len({p.case_id for p in prefixes}),
        "opportunity_prevalence_all": _frac(prefixes, lambda p: p.oracle_actionable),
        "opportunity_prevalence_failing": _frac(failing, lambda p: p.oracle_actionable),
        "mean_g_star": _mean([p.g_star for p in prefixes if p.g_star is not None]),
        "mean_gap": _mean([p.gap for p in prefixes if p.gap is not None]),
        "harm_sensitive_rate": _frac(prefixes, lambda p: p.harm_sensitive),
        "rescue_rate": rescue_rate,
        "rescue_denominator": n_rescue_denom,
        "harm_rate": harm_rate,
        "harm_denominator": n_harm_denom,
        "window": _window_summary(windows),
    }

    prefix_cards = [
        {
            "prefix_id": p.prefix_id,
            "trajectory_id": p.trajectory_id,
            "case_id": p.case_id,
            "progress": p.progress,
            "stratum": p.stratum,
            "continue_score": p.continue_score,
            "continue_success": p.continue_success,
            "g_star": p.g_star,
            "g_star_treatment": p.g_star_treatment,
            "g_critic": p.g_critic,
            "gap": p.gap,
            "labels": _taxonomy(p, ladder_by_prefix.get(p.prefix_id), eps),
        }
        for p in prefixes
    ]

    return {"summary": summary, "prefixes": prefix_cards, "windows": [w.__dict__ for w in windows]}


def write_report(report: dict[str, Any], out_prefix: Path) -> dict[str, str]:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_prefix.with_suffix(".json")
    md_path = out_prefix.with_suffix(".md")
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(to_markdown(report), encoding="utf-8")
    return {"json": str(json_path), "md": str(md_path)}


def to_markdown(report: dict[str, Any]) -> str:
    s = report["summary"]
    lines = [
        "# Rescue Window report",
        "",
        f"- prefixes: {s['n_prefixes']} | trajectories: {s['n_trajectories']} | cases: {s['n_cases']}",
        f"- epsilon: {s['epsilon']}",
        f"- oracle opportunity (all): {_pct(s['opportunity_prevalence_all'])}",
        f"- oracle opportunity (failing baseline): {_pct(s['opportunity_prevalence_failing'])}",
        f"- mean G*: {_num(s['mean_g_star'])} | mean gap: {_num(s['mean_gap'])}",
        f"- rescue rate: {_pct(s['rescue_rate'])} (n={s['rescue_denominator']}) | "
        f"harm rate: {_pct(s['harm_rate'])} (n={s['harm_denominator']})",
        f"- harm-sensitive prefixes: {_pct(s['harm_sensitive_rate'])}",
    ]
    w = s["window"]
    lines.append(
        f"- window: exists {_pct(w['exists_rate'])} | mean width {_num(w['mean_width'])} "
        f"| mean area {_num(w['mean_area'])} | mean peak {_num(w['mean_peak'])}"
    )
    lines += ["", "## Per-prefix", "", "| prefix | progress | cont✓ | G* | best | gap | labels |", "|---|---|---|---|---|---|---|"]
    for card in report["prefixes"]:
        lines.append(
            f"| {card['prefix_id']} | {card['progress']} | {card['continue_success']} "
            f"| {_num(card['g_star'])} | {card['g_star_treatment'] or '-'} "
            f"| {_num(card['gap'])} | {', '.join(card['labels'])} |"
        )
    return "\n".join(lines) + "\n"


def _taxonomy(
    prefix: PrefixStat, ladder: PrefixLadder | None, eps: float
) -> list[str]:
    labels: list[str] = []
    if prefix.continue_success is False:
        labels.append("high_continuation_risk")
    if prefix.oracle_actionable:
        labels.append("oracle_actionable")
    else:
        labels.append("irrecoverable_under_scope")
    if prefix.harm_sensitive:
        labels.append("harm_sensitive")
    # channel-limited: ORACLE_DIAG (actor) helps but channel best ~0
    if ladder is not None and ladder.channel_to_actor_gap is not None:
        channel = ladder.rung_best.get("channel")
        if (channel is None or channel <= eps) and ladder.channel_to_actor_gap > eps:
            labels.append("channel_limited")
    return labels


def _binary_rescue_harm(
    rows: list[EvalUnit], prefixes: list[PrefixStat]
) -> tuple[float | None, float | None, int, int]:
    continue_success = {p.prefix_id: p.continue_success for p in prefixes}
    rescued = harmed = n_fail = n_ok = 0
    for row in rows:
        if row.content_level is ContentLevel.CONTINUE or row.status != "succeeded":
            continue
        if row.binary_success is None:
            continue
        cont = continue_success.get(row.prefix_id)
        if cont is False:
            n_fail += 1
            if row.binary_success:
                rescued += 1
        elif cont is True:
            n_ok += 1
            if not row.binary_success:
                harmed += 1
    return (
        rescued / n_fail if n_fail else None,
        harmed / n_ok if n_ok else None,
        n_fail,
        n_ok,
    )


def _window_summary(windows: list[WindowMetrics]) -> dict[str, Any]:
    if not windows:
        return {"exists_rate": None, "mean_width": None, "mean_area": None, "mean_peak": None}
    existing = [w for w in windows if w.exists]
    return {
        "exists_rate": len(existing) / len(windows),
        "mean_width": _mean([w.width for w in existing if w.width is not None]),
        "mean_area": _mean([w.area for w in windows]),
        "mean_peak": _mean([w.peak for w in windows if w.peak is not None]),
    }


def _frac(items: list[PrefixStat], predicate: Any) -> float | None:
    if not items:
        return None
    return sum(1 for item in items if predicate(item)) / len(items)


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _num(value: float | None) -> str:
    return f"{value:.3f}" if isinstance(value, int | float) else "-"


def _pct(value: float | None) -> str:
    return f"{value * 100:.1f}%" if isinstance(value, int | float) else "-"
