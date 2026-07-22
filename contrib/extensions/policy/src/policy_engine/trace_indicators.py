# code-health: ignore-file[AM025] -- trace indicators normalize persisted JSON rows
"""Policy metric rows for the shared AgentM trace viewer."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import cast

from sqlalchemy.engine import Connection, RowMapping

from agentm.presenter.trajectory import TraceRow

from .ifg_regions import IfgRegionState, RegionReadQuery
from .types import ToolArgs


# These mirror the shipped rules in base_policy.yaml so the TUI can show both
# the observed value and the boundary that the rule evaluates.
_CONTEXT_USAGE_WARNING_PCT = 80.0
_REGION_READ_WINDOW = 20
_REGION_READ_MIN_RATIO = 0.8
_REGION_READ_MIN_LINES = 10
_REGION_READ_MIN_REPEATS = 3
_REGION_READ_MIN_OVERALL_RATIO = 0.15


def load_policy_indicator_rows(
    conn: Connection,
    session_id: str,
    existing_tables: set[str],
    session_summary: object,
) -> list[TraceRow]:
    """Build current policy indicator rows from persisted session state."""

    rows: list[TraceRow] = []
    if "policy_context_state" in existing_tables:
        context_row = _context_budget_row(conn, session_id)
        if context_row is not None:
            rows.append(context_row)

    intervention = _mapping_value(session_summary, "intervention")
    if isinstance(intervention, Mapping):
        rows.append(_intervention_row(session_id, intervention))

    investigation = _mapping_value(session_summary, "investigation")
    if isinstance(investigation, Mapping):
        rows.append(_investigation_row(session_id, investigation))

    if "policy_tool_events" in existing_tables:
        region_row = _region_read_row(conn, session_id)
        if region_row is not None:
            rows.append(region_row)
    return rows


def _context_budget_row(conn: Connection, session_id: str) -> TraceRow | None:
    snapshots: list[tuple[int, Mapping[str, object]]] = []
    for row in _rows(
        conn,
        """
        SELECT turn_index, context_json
        FROM policy_context_state
        WHERE session_id = ?
        ORDER BY turn_index ASC
        """,
        (session_id,),
    ):
        context = _loads(row["context_json"])
        turn = _coerce_int(row["turn_index"])
        if turn is not None and isinstance(context, Mapping):
            snapshots.append((turn, context))
    if not snapshots:
        return None

    latest_turn, latest = snapshots[-1]
    peak_turn, peak = max(
        snapshots,
        key=lambda item: _coerce_float(item[1].get("context_usage_pct")) or 0.0,
    )
    current_pct = _coerce_float(latest.get("context_usage_pct")) or 0.0
    peak_pct = _coerce_float(peak.get("context_usage_pct")) or 0.0
    threshold_crossed = current_pct > _CONTEXT_USAGE_WARNING_PCT
    current_tokens = _coerce_int(latest.get("total_context_tokens")) or 0
    input_budget = _coerce_int(latest.get("input_budget_tokens"))
    usage = (
        f"{current_tokens:,} / {input_budget:,} tokens"
        if input_budget is not None
        else f"{current_tokens:,} tokens"
    )
    return TraceRow(
        key=f"policy:indicator:context-budget:{session_id}",
        kind="policy",
        title="Context Budget",
        preview=(
            f"current:{current_pct:.2f}% ({usage}) "
            f"peak:{peak_pct:.2f}%@T{peak_turn} "
            f"warning:>{_CONTEXT_USAGE_WARNING_PCT:g}%"
        ),
        content=_json_content(
            {
                "rules": ["budget-warning"],
                "indicator": (
                    "lookup('context_state', session.turn_count, 'context_usage_pct')"
                ),
                "warning_above_pct": _CONTEXT_USAGE_WARNING_PCT,
                "current": dict(latest),
                "peak": {"turn_index": peak_turn, **dict(peak)},
            }
        ),
        turn_index=latest_turn,
        cause="warning" if threshold_crossed else "within threshold",
        metadata=_metadata("context_budget", rules=("budget-warning",)),
    )


def _intervention_row(
    session_id: str,
    metrics: Mapping[str, object],
) -> TraceRow:
    signals = _string_sequence(metrics.get("signals"))
    observed = _string_sequence(metrics.get("observed_signals"))
    active = metrics.get("active") is True
    mutations = _coerce_int(metrics.get("mutations")) or 0
    effective_targets = _coerce_float(metrics.get("effective_targets")) or 0.0
    novel_files = _coerce_int(metrics.get("novel_files")) or 0
    calls_since_mutation = _coerce_int(metrics.get("calls_since_last_mutation")) or 0
    adaptive_patience = _coerce_int(metrics.get("adaptive_patience")) or 0
    signal_text = ",".join(signals) if signals else "none"
    return TraceRow(
        key=f"policy:indicator:intervention:{session_id}",
        kind="policy",
        title="Intervention Metrics",
        preview=(
            f"{'active' if active else 'inactive'} mutations:{mutations} "
            f"effective-targets:{effective_targets:.2f} novel-files:{novel_files} "
            f"validation-horizon:{calls_since_mutation}/{adaptive_patience} "
            f"signals:{signal_text}"
        ),
        content=_json_content(
            {
                "rules": [
                    "exploration-not-converging",
                    "mutation-target-drift",
                    "unvalidated-intervention",
                ],
                "metrics": dict(metrics),
            }
        ),
        cause=signal_text if signals else ("active" if active else "inactive"),
        metadata=_metadata(
            "intervention",
            signals=signals,
            observed_signals=observed,
        ),
    )


def _investigation_row(
    session_id: str,
    metrics: Mapping[str, object],
) -> TraceRow:
    evidence_counts = metrics.get("evidence_counts")
    counts = evidence_counts if isinstance(evidence_counts, Mapping) else {}
    reentries = _coerce_int(counts.get("unchanged_anchor_reentry")) or 0
    churn = _coerce_int(counts.get("unchanged_anchor_churn")) or 0
    replacements = _coerce_int(counts.get("created_artifact_replacement")) or 0
    cycles = _coerce_int(counts.get("unchanged_investigation_state_cycle")) or 0
    phases = _coerce_int(metrics.get("completed_phases")) or 0
    generation = _coerce_int(metrics.get("repository_generation")) or 0
    generation_reentries = _coerce_int(metrics.get("generation_reentries")) or 0
    adaptive_support = _coerce_int(metrics.get("adaptive_reentry_support")) or 0
    bash_support = _coerce_int(metrics.get("bash_support_observations")) or 0
    bash_candidates = _coerce_int(metrics.get("bash_path_candidates")) or 0
    evidence_total = reentries + churn + replacements + cycles
    return TraceRow(
        key=f"policy:indicator:investigation:{session_id}",
        kind="policy",
        title="Investigation Metrics",
        preview=(
            f"reentries:{generation_reentries}/{adaptive_support} churn:{churn} "
            f"replacements:{replacements} cycles:{cycles} phases:{phases} "
            f"bash-support:{bash_support} candidates:{bash_candidates} "
            f"repository-generation:{generation}"
        ),
        content=_json_content(
            {
                "rules": [
                    "unchanged-anchor-reentry",
                    "investigation-anchor-churn",
                    "created-artifact-replacement",
                    "unchanged-investigation-state-cycle",
                ],
                "metrics": dict(metrics),
            }
        ),
        cause=f"{evidence_total} signal(s)" if evidence_total else "no signals",
        metadata=_metadata("investigation", evidence_counts=dict(counts)),
    )


def _region_read_row(conn: Connection, session_id: str) -> TraceRow | None:
    state = IfgRegionState()
    for row in _rows(
        conn,
        """
        SELECT turn, tool_name, args_json, result_json, processed_json
        FROM policy_tool_events
        WHERE session_id = ?
          AND phase = 'post'
          AND tool_name IN ('read', 'edit', 'write')
        ORDER BY id ASC
        """,
        (session_id,),
    ):
        args = _loads(row["args_json"])
        if not isinstance(args, Mapping):
            continue
        state.record(
            str(row["tool_name"]),
            cast(ToolArgs, dict(args)),
            _region_result(
                _loads(row["result_json"]),
                _loads(row["processed_json"]),
            ),
            _coerce_int(row["turn"]) or 0,
        )

    query = RegionReadQuery(state)
    if query.count() == 0:
        return None
    metrics = query.summary(
        last=_REGION_READ_WINDOW,
        min_ratio=_REGION_READ_MIN_RATIO,
        min_lines=_REGION_READ_MIN_LINES,
    )
    repeated = _coerce_int(metrics.get("repeated_reads")) or 0
    overlap_ratio = _coerce_float(metrics.get("overlap_ratio")) or 0.0
    signaled = (
        repeated >= _REGION_READ_MIN_REPEATS
        and overlap_ratio >= _REGION_READ_MIN_OVERALL_RATIO
    )
    return TraceRow(
        key=f"policy:indicator:region-reads:{session_id}",
        kind="policy",
        title="Region Read Metrics",
        preview=(
            f"last:{_REGION_READ_WINDOW} reads:{metrics['reads']} "
            f"repeated:{repeated}/{_REGION_READ_MIN_REPEATS} "
            f"overlap:{overlap_ratio:.1%}/{_REGION_READ_MIN_OVERALL_RATIO:.0%} "
            f"overlap-lines:{metrics['overlap_lines']}"
        ),
        content=_json_content(
            {
                "rules": ["repeated-region-reading"],
                "thresholds": {
                    "window": _REGION_READ_WINDOW,
                    "minimum_per_read_overlap_ratio": _REGION_READ_MIN_RATIO,
                    "minimum_overlap_lines": _REGION_READ_MIN_LINES,
                    "minimum_repeated_reads": _REGION_READ_MIN_REPEATS,
                    "minimum_overall_overlap_ratio": (_REGION_READ_MIN_OVERALL_RATIO),
                },
                "metrics": metrics,
            }
        ),
        cause="signal" if signaled else "below threshold",
        metadata=_metadata(
            "region_reads",
            rule="repeated-region-reading",
        ),
    )


def _region_result(result: object, processed: object) -> dict[str, object]:
    normalized: dict[str, object] = {}
    text = _tool_result_text(result)
    if text is not None:
        normalized["text"] = text
    content_hash = _mapping_value(processed, "content_hash")
    if content_hash is not None:
        normalized["content_hash"] = content_hash
    if _mapping_value(processed, "is_error") is True:
        normalized["error"] = _mapping_value(processed, "error") or "tool error"
    return normalized


def _metadata(indicator: str, **values: object) -> dict[str, object]:
    return {
        "category": "summary",
        "aggregate": "policy_indicator",
        "indicator": indicator,
        **values,
    }


def _rows(
    conn: Connection,
    sql: str,
    params: Sequence[object] = (),
) -> list[RowMapping]:
    return list(conn.exec_driver_sql(sql, tuple(params)).mappings().all())


def _loads(raw: object) -> object:
    if raw is None or not isinstance(raw, str):
        return raw
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _mapping_value(value: object, key: str) -> object:
    return value.get(key) if isinstance(value, Mapping) else None


def _tool_result_text(result: object) -> str | None:
    if not isinstance(result, Mapping):
        return None
    text = result.get("text")
    if isinstance(text, str):
        return text
    content = result.get("content")
    if not isinstance(content, list):
        return None
    parts = [
        text
        for item in content
        if isinstance(item, Mapping) and isinstance((text := item.get("text")), str)
    ]
    return "".join(parts) if parts else None


def _json_content(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str)


def _coerce_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(str(value))
    except ValueError:
        return None


def _coerce_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value))
    except ValueError:
        return None


def _string_sequence(value: object) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(str(item) for item in value)


__all__ = ["load_policy_indicator_rows"]
