"""Export reminder-fork comparisons from ClickHouse sessions."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from agentm.core.observability import clickhouse

ForkRow = dict[str, str]


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False))
            fh.write("\n")


def _message_text(message: dict[str, Any]) -> str:
    payload = message.get("payload")
    if not isinstance(payload, dict):
        return ""
    content = payload.get("content")
    parts: list[str] = []
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
            elif block.get("type") == "tool_result":
                nested = block.get("content")
                if isinstance(nested, list):
                    for item in nested:
                        if isinstance(item, dict) and item.get("type") == "text":
                            parts.append(str(item.get("text") or ""))
    return "\n".join(part for part in parts if part)


def _message_timestamp_ns(message: dict[str, Any]) -> int:
    raw = message.get("timestamp")
    if raw in (None, 0):
        payload = message.get("payload")
        if isinstance(payload, dict):
            raw = payload.get("timestamp")
    if raw in (None, 0) or raw is None:
        return 0
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0
    if value > 10_000_000_000:
        return int(value)
    return int(value * 1_000_000_000)


def _header(info: dict[str, Any]) -> dict[str, Any]:
    header = info.get("header")
    return header if isinstance(header, dict) else {}


def _session_start_ns(info: dict[str, Any]) -> int:
    raw = _header(info).get("timestamp")
    if raw in (None, 0) or raw is None:
        return 0
    try:
        return int(float(raw) * 1_000_000_000)
    except (TypeError, ValueError):
        return 0


def _lineage(info: dict[str, Any]) -> dict[str, Any]:
    config = _header(info).get("config")
    if not isinstance(config, dict):
        return {}
    lineage = config.get("lineage")
    return lineage if isinstance(lineage, dict) else {}


def _experiment(info: dict[str, Any]) -> dict[str, Any]:
    config = _header(info).get("config")
    if not isinstance(config, dict):
        return {}
    experiment = config.get("experiment")
    return experiment if isinstance(experiment, dict) else {}


def _info_summary(info: dict[str, Any]) -> dict[str, Any]:
    header = _header(info)
    config = header.get("config")
    scenario = config.get("scenario") if isinstance(config, dict) else None
    fingerprint = info.get("fingerprint")
    return {
        "session_id": header.get("id"),
        "parent_session": header.get("parent_session"),
        "cwd": header.get("cwd"),
        "timestamp": header.get("timestamp"),
        "scenario": scenario,
        "lineage": _lineage(info),
        "experiment": _experiment(info),
        "fingerprint": fingerprint if isinstance(fingerprint, dict) else {},
    }


def _read_forks(path: Path) -> list[ForkRow]:
    with path.open(newline="", encoding="utf-8") as fh:
        return [
            {str(k): str(v or "") for k, v in row.items()}
            for row in csv.DictReader(fh, delimiter="\t")
        ]


def _score_rows_by_fork(run_dir: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(run_dir.glob("batch-*/*_score.csv")):
        with path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                sid = row.get("fork_session_id")
                if sid and sid not in out:
                    out[sid] = {
                        "path": str(path),
                        "row": {str(k): str(v or "") for k, v in row.items()},
                    }
    return out


def _exact_reminder_candidates(
    score_entry: dict[str, Any] | None,
    fork_info: dict[str, Any],
) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if score_entry is not None:
        row = score_entry.get("row")
        if isinstance(row, dict):
            text = row.get("reminder_text")
            if isinstance(text, str) and text:
                out.append(("score_csv", text))
    experiment_text = _experiment(fork_info).get("reminder_text")
    if isinstance(experiment_text, str) and experiment_text:
        out.append(("trace_experiment", experiment_text))
    return out


def _find_reminder_index(
    messages: list[dict[str, Any]],
    fork_info: dict[str, Any],
    score_entry: dict[str, Any] | None,
) -> tuple[int | None, str, str, str | None]:
    if score_entry is not None:
        row = score_entry.get("row")
        if isinstance(row, dict):
            reminder_id = row.get("fork_reminder_message_id")
            if isinstance(reminder_id, str) and reminder_id:
                for idx, message in enumerate(messages):
                    if message.get("id") == reminder_id:
                        return idx, _message_text(message), "score_message_id", reminder_id

    for source, text in _exact_reminder_candidates(score_entry, fork_info):
        for idx, message in enumerate(messages):
            if text and text in _message_text(message):
                msg_id = message.get("id")
                return idx, text, source, str(msg_id) if msg_id else None

    start_ns = _session_start_ns(fork_info)
    if start_ns:
        for idx, message in enumerate(messages):
            payload = message.get("payload")
            if not isinstance(payload, dict) or payload.get("role") != "user":
                continue
            ts = _message_timestamp_ns(message)
            if ts and ts < start_ns:
                continue
            text = _message_text(message).strip()
            if text:
                msg_id = message.get("id")
                return idx, text, "first_user_after_fork_start", str(msg_id) if msg_id else None

    for idx in range(len(messages) - 1, -1, -1):
        text = _message_text(messages[idx]).strip()
        if "<system_reminder>" in text or "Reminder for this fork" in text:
            msg_id = messages[idx].get("id")
            return idx, text, "last_reminder_like_user_message", str(msg_id) if msg_id else None

    return None, "", "missing", None


def _safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    return value.strip("-") or "unknown"


def _readme(meta: dict[str, Any]) -> str:
    raw_reminder = meta.get("reminder")
    raw_counts = meta.get("counts")
    raw_fork = meta.get("fork")
    reminder: dict[str, Any] = raw_reminder if isinstance(raw_reminder, dict) else {}
    counts: dict[str, Any] = raw_counts if isinstance(raw_counts, dict) else {}
    fork: dict[str, Any] = raw_fork if isinstance(raw_fork, dict) else {}
    return "\n".join(
        [
            f"# Fork `{meta.get('export_id')}`",
            "",
            f"- case_id: `{fork.get('case_id') or ''}`",
            f"- strategy: `{fork.get('strategy') or ''}`",
            f"- outcome: `{fork.get('outcome') or ''}`",
            f"- source_session_id: `{fork.get('source_session_id') or ''}`",
            f"- fork_session_id: `{fork.get('fork_session_id') or ''}`",
            f"- source_turn: `{fork.get('source_turn') or ''}`",
            f"- reminder_source: `{reminder.get('source') or ''}`",
            f"- reminder_message_id: `{reminder.get('message_id') or ''}`",
            f"- source_before/source_after: **{counts.get('source_before', 0)}** / **{counts.get('source_after', 0)}** messages",
            f"- fork_before/fork_after: **{counts.get('fork_before', 0)}** / **{counts.get('fork_after', 0)}** messages",
            "",
            "## Files",
            "",
            "| File | What |",
            "|---|---|",
            "| `meta.json` | fork metadata, lineage summary, split counts, score row if available |",
            "| `reminder.md` | exact reminder text recovered from score CSV or fork trajectory |",
            "| `source_before_fork.jsonl` | source session prefix copied into the fork |",
            "| `source_after_fork.jsonl` | original source continuation after the fork point |",
            "| `fork_before_reminder.jsonl` | fork session prefix before the injected reminder |",
            "| `fork_after_reminder.jsonl` | reminder plus branch continuation |",
            "| `source_full.jsonl` / `fork_full.jsonl` | complete source and fork trajectories |",
            "",
        ]
    )


def export_forks(
    *,
    run_dir: Path,
    forks_tsv: Path,
    out_dir: Path,
    include_full: bool = True,
) -> Path:
    """Export one review directory per fork row."""

    url = clickhouse.get_url()
    if url is None:
        raise RuntimeError(
            "ClickHouse is unavailable; set AGENTM_CLICKHOUSE_URL or start localhost:8123"
        )

    forks = _read_forks(forks_tsv)
    score_rows = _score_rows_by_fork(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict[str, Any]] = []
    message_cache: dict[str, list[dict[str, Any]]] = {}
    info_cache: dict[str, dict[str, Any]] = {}

    def messages_for(session_id: str) -> list[dict[str, Any]]:
        if session_id not in message_cache:
            message_cache[session_id] = list(clickhouse.messages(url, session_id))
        return message_cache[session_id]

    def info_for(session_id: str) -> dict[str, Any]:
        if session_id not in info_cache:
            info_cache[session_id] = clickhouse.info(url, session_id)
        return info_cache[session_id]

    for sequence, fork in enumerate(forks, start=1):
        source_session_id = fork["source_session_id"]
        fork_session_id = fork["fork_session_id"]
        source_messages = messages_for(source_session_id)
        fork_messages = messages_for(fork_session_id)
        source_info = info_for(source_session_id)
        fork_info = info_for(fork_session_id)
        score_entry = score_rows.get(fork_session_id)

        reminder_index, reminder_text, reminder_source, reminder_message_id = (
            _find_reminder_index(fork_messages, fork_info, score_entry)
        )
        split_at = reminder_index if reminder_index is not None else 0
        source_split_at = min(split_at, len(source_messages))

        source_before = source_messages[:source_split_at]
        source_after = source_messages[source_split_at:]
        fork_before = fork_messages[:split_at]
        fork_after = fork_messages[split_at:]

        export_id = (
            f"{sequence:03d}_"
            f"{_safe_name(fork['case_id'])}_"
            f"{_safe_name(fork['strategy'])}_"
            f"{fork_session_id[:8]}"
        )
        case_dir = out_dir / export_id
        case_dir.mkdir(parents=True, exist_ok=True)

        score_row = None
        score_path = None
        if score_entry is not None:
            raw_row = score_entry.get("row")
            score_row = raw_row if isinstance(raw_row, dict) else None
            raw_path = score_entry.get("path")
            score_path = raw_path if isinstance(raw_path, str) else None

        meta = {
            "export_id": export_id,
            "fork": fork,
            "source_session": _info_summary(source_info),
            "fork_session": _info_summary(fork_info),
            "reminder": {
                "message_id": reminder_message_id,
                "source": reminder_source,
                "text_hash": hashlib.sha256(reminder_text.encode()).hexdigest()
                if reminder_text
                else "",
                "text": reminder_text,
                "message_index": reminder_index,
            },
            "split": {
                "source_split_index": source_split_at,
                "fork_reminder_index": reminder_index,
                "source_fork_point_message_id": (
                    source_before[-1].get("id") if source_before else None
                ),
            },
            "counts": {
                "source_full": len(source_messages),
                "source_before": len(source_before),
                "source_after": len(source_after),
                "fork_full": len(fork_messages),
                "fork_before": len(fork_before),
                "fork_after": len(fork_after),
            },
            "score": {
                "path": score_path,
                "row": score_row,
            },
        }

        _write_json(case_dir / "meta.json", meta)
        (case_dir / "reminder.md").write_text(reminder_text, encoding="utf-8")
        _write_jsonl(case_dir / "source_before_fork.jsonl", source_before)
        _write_jsonl(case_dir / "source_after_fork.jsonl", source_after)
        _write_jsonl(case_dir / "fork_before_reminder.jsonl", fork_before)
        _write_jsonl(case_dir / "fork_after_reminder.jsonl", fork_after)
        if include_full:
            _write_jsonl(case_dir / "source_full.jsonl", source_messages)
            _write_jsonl(case_dir / "fork_full.jsonl", fork_messages)
        if score_row is not None:
            _write_json(case_dir / "score_row.json", score_row)
        (case_dir / "README.md").write_text(_readme(meta), encoding="utf-8")

        index_rows.append(
            {
                "export_id": export_id,
                "case_id": fork["case_id"],
                "strategy": fork["strategy"],
                "outcome": fork["outcome"],
                "source_session_id": source_session_id,
                "fork_session_id": fork_session_id,
                "source_turn": fork["source_turn"],
                "reminder_source": reminder_source,
                "reminder_message_id": reminder_message_id or "",
                "source_before": len(source_before),
                "source_after": len(source_after),
                "fork_before": len(fork_before),
                "fork_after": len(fork_after),
            }
        )

    with (out_dir / "index.tsv").open("w", encoding="utf-8", newline="") as fh:
        fieldnames = [
            "export_id",
            "case_id",
            "strategy",
            "outcome",
            "source_session_id",
            "fork_session_id",
            "source_turn",
            "reminder_source",
            "reminder_message_id",
            "source_before",
            "source_after",
            "fork_before",
            "fork_after",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(index_rows)

    return out_dir


__all__ = ["export_forks"]
