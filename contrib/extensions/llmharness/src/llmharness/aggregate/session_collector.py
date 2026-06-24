"""AgentM ClickHouse session trajectory -> :class:`CaseData`."""

from __future__ import annotations

from typing import Any

from agentm.core.observability import clickhouse

from .case import CaseData, CaseMeta


def _timestamp_ns(message: dict[str, Any]) -> int:
    raw = message.get("timestamp")
    if raw in (None, 0):
        payload = message.get("payload")
        if isinstance(payload, dict):
            raw = payload.get("timestamp")
    if raw in (None, 0):
        return 0
    if raw is None:
        return 0
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0
    # AgentM message timestamps are persisted as Unix seconds. Accept
    # nanosecond-shaped values as-is for forward compatibility.
    if value > 10_000_000_000:
        return int(value)
    return int(value * 1_000_000_000)


def _session_times(messages: list[dict[str, Any]]) -> tuple[int, int]:
    values = [ts for msg in messages if (ts := _timestamp_ns(msg))]
    if not values:
        return 0, 0
    return min(values), max(values)


def collect_session_case(
    *,
    session_id: str,
    trace_id: str | None = None,
    case_id: str | None = None,
    sample_id_override: str | None = None,
    dataset_name_override: str | None = None,
    dataset_path_override: str | None = None,
) -> CaseData:
    """Build a plain-session aggregate case from ClickHouse.

    ``sample_id_override`` becomes both ``sample_id`` and the default
    ``case_id`` when supplied. Otherwise the
    case id falls back to ``case_id`` and then ``session_id``.
    """

    url = clickhouse.get_url()
    if url is None:
        raise RuntimeError(
            "ClickHouse is unavailable; set AGENTM_CLICKHOUSE_URL or start localhost:8123"
        )

    messages = list(clickhouse.messages(url, session_id))
    if not messages:
        raise RuntimeError(f"no messages found for session {session_id}")

    info = clickhouse.info(url, session_id)
    header = info.get("header") if isinstance(info, dict) else None
    header_trace_id = None
    if isinstance(header, dict):
        header_trace_id = header.get("trace_id") or header.get("root_session")

    resolved_sample_id = sample_id_override
    resolved_case_id = resolved_sample_id or case_id or session_id
    started_at, ended_at = _session_times(messages)

    meta = CaseMeta(
        case_id=resolved_case_id,
        session_id=session_id,
        trace_id=trace_id or header_trace_id or "",
        sample_id=resolved_sample_id,
        dataset_name=dataset_name_override,
        dataset_path=dataset_path_override,
        started_at_ns=started_at,
        ended_at_ns=ended_at,
        extractor_firings=0,
        auditor_firings=0,
        surfaced_reminders=0,
        silent_verdicts=0,
    )
    return CaseData(meta=meta, main_agent_messages=messages)


def trace_ids_for_sessions(session_ids: set[str]) -> dict[str, str]:
    """Return ClickHouse trace ids for the requested sessions."""

    if not session_ids:
        return {}
    url = clickhouse.get_url()
    if url is None:
        raise RuntimeError(
            "ClickHouse is unavailable; set AGENTM_CLICKHOUSE_URL or start localhost:8123"
        )
    out: dict[str, str] = {}
    for row in clickhouse.index(url):
        sid = row.get("session_id")
        if sid in session_ids:
            trace_id = row.get("trace_id")
            if trace_id:
                out[str(sid)] = str(trace_id)
            if len(out) == len(session_ids):
                break
    return out


__all__ = ["collect_session_case", "trace_ids_for_sessions"]
