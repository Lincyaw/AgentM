"""Serialise :class:`CaseData` to the canonical case directory layout."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .case import CaseData, CaseLayout, FiringRecord


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[Any]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False))
            fh.write("\n")


def _firing_payload(fr: FiringRecord) -> dict[str, Any]:
    """Serialize one firing for on-disk persistence.

    ``raw_assistant_messages`` is included only when non-empty so consumers can
    use key presence as a capability check.
    """
    payload: dict[str, Any] = {
        "phase": fr.phase,
        "sequence": fr.sequence,
        "turn_index": fr.turn_index,
        "ts_ns": fr.ts_ns,
        "status": fr.status,
        "error": fr.error,
        "latency_ms": fr.latency_ms,
        "input": fr.input_payload,
        "output": fr.output,
    }
    if fr.raw_assistant_messages:
        payload["raw_assistant_messages"] = fr.raw_assistant_messages
    return payload


def _firing_summary(fr: FiringRecord) -> str:
    """One-line human summary for the trajectory timeline + README."""
    if fr.phase == "extractor":
        out = fr.output or {}
        return (
            f"extractor#{fr.sequence} turn={fr.turn_index} status={fr.status} "
            f"events={len(out.get('events') or [])} edges={len(out.get('edges') or [])}"
        )
    out = fr.output or {}
    surface = "REMIND" if out.get("surface_reminder") else "silent"
    return f"auditor#{fr.sequence} turn={fr.turn_index} status={fr.status} verdict={surface}"


def _message_ts_ns(message: dict[str, Any]) -> int:
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
    if value > 10_000_000_000:
        return int(value)
    return int(value * 1_000_000_000)


def _message_summary(message: dict[str, Any], sequence: int) -> str:
    payload = message.get("payload")
    payload = payload if isinstance(payload, dict) else {}
    role = str(payload.get("role") or message.get("type") or "?")
    content = payload.get("content")
    text_blocks = 0
    tool_calls = 0
    tool_results = 0
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                text_blocks += 1
            elif block_type == "tool_call":
                tool_calls += 1
            elif block_type == "tool_result":
                tool_results += 1
    details: list[str] = []
    if text_blocks:
        details.append(f"text={text_blocks}")
    if tool_calls:
        details.append(f"tool_calls={tool_calls}")
    if tool_results:
        details.append(f"tool_results={tool_results}")
    detail = " " + " ".join(details) if details else ""
    return f"message#{sequence} role={role}{detail}"


def _build_message_trajectory(case: CaseData) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, message in enumerate(case.main_agent_messages, start=1):
        rows.append(
            {
                "ts_ns": _message_ts_ns(message),
                "source": "main_agent",
                "sequence": idx,
                "turn_index": None,
                "summary": _message_summary(message, idx),
                "ref": "main_agent.jsonl",
                "line": idx,
            }
        )
    return rows


def _build_trajectory(case: CaseData, layout: CaseLayout) -> list[dict[str, Any]]:
    """Flat per-source timeline used for human review.

    Each row points back at the artefact path so a reviewer can grep
    the trajectory then open the file. Sorted by ``ts_ns``; firings
    tied on timestamp keep their natural ordering via ``sequence``.
    """
    if not case.extractor_firings and not case.auditor_firings:
        return _build_message_trajectory(case)

    rows: list[dict[str, Any]] = []
    for fr in case.extractor_firings:
        ref = layout.firing_path("extractor", fr.sequence, fr.turn_index)
        rows.append(
            {
                "ts_ns": fr.ts_ns,
                "source": "extractor",
                "sequence": fr.sequence,
                "turn_index": fr.turn_index,
                "summary": _firing_summary(fr),
                "ref": str(ref.relative_to(layout.root)),
            }
        )
    for fr in case.auditor_firings:
        ref = layout.firing_path("auditor", fr.sequence, fr.turn_index)
        rows.append(
            {
                "ts_ns": fr.ts_ns,
                "source": "auditor",
                "sequence": fr.sequence,
                "turn_index": fr.turn_index,
                "summary": _firing_summary(fr),
                "ref": str(ref.relative_to(layout.root)),
            }
        )
    rows.sort(key=lambda r: (r["ts_ns"], r["source"], r["sequence"]))
    return rows


def _build_readme(case: CaseData) -> str:
    m = case.meta
    lines = [
        f"# Case `{m.case_id}`",
        "",
        f"- session_id: `{m.session_id}`",
        f"- trace_id: `{m.trace_id or '—'}`",
        f"- sample_id: `{m.sample_id or '—'}`",
        f"- dataset: `{m.dataset_name or '—'}`",
        f"- main-agent messages: **{len(case.main_agent_messages)}**",
        f"- extractor firings: **{m.extractor_firings}**",
        f"- auditor firings: **{m.auditor_firings}** "
        f"(surfaced={m.surfaced_reminders}, silent={m.silent_verdicts})",
        f"- duration: {(m.ended_at_ns - m.started_at_ns) / 1e9:.2f}s",
        "",
        "## Layout",
        "",
        "| File | What |",
        "|---|---|",
        "| `meta.json` | summary fields you see above, machine-readable |",
        "| `main_agent.jsonl` | full main-agent message trajectory (one per line, AgentM-native shape) |",
        "| `extractor/NNN_turn_T.json` | one extractor firing's input + output |",
        "| `auditor/NNN_turn_T.json` | one auditor firing's input + verdict |",
        "| `context_index/after_extractor_NNN.json` | accumulated records + links after that firing |",
        "| `verdicts.jsonl` | verdict timeline (one verdict per line) |",
        "| `trajectory.jsonl` | flat review timeline pointing into the artefacts |",
        "",
        "## Firings",
        "",
    ]
    for fr in case.extractor_firings:
        lines.append(f"- {_firing_summary(fr)}")
    for fr in case.auditor_firings:
        lines.append(f"- {_firing_summary(fr)}")
    lines.append("")
    return "\n".join(lines)


def write_case(case: CaseData, out_dir: Path) -> Path:
    """Materialise a case under ``out_dir/<case_id>/`` and return that path.

    Existing files are overwritten; the directory is created if needed.
    """
    case_dir = out_dir / case.meta.case_id
    layout = CaseLayout(root=case_dir)

    case_dir.mkdir(parents=True, exist_ok=True)
    layout.extractor_dir.mkdir(exist_ok=True)
    layout.auditor_dir.mkdir(exist_ok=True)
    layout.index_dir.mkdir(exist_ok=True)

    _write_json(layout.meta_path, asdict(case.meta))
    _write_jsonl(layout.main_agent_path, case.main_agent_messages)
    _write_jsonl(layout.verdicts_path, case.verdicts)

    for fr in case.extractor_firings:
        path = layout.firing_path("extractor", fr.sequence, fr.turn_index)
        _write_json(path, _firing_payload(fr))
    for fr in case.auditor_firings:
        path = layout.firing_path("auditor", fr.sequence, fr.turn_index)
        _write_json(path, _firing_payload(fr))
    for snap in case.index_snapshots:
        _write_json(
            layout.snapshot_path(snap.after_extractor_firing),
            {
                "after_extractor_firing": snap.after_extractor_firing,
                "turn_index": snap.turn_index,
                "events": snap.events,
                "edges": snap.edges,
            },
        )

    _write_jsonl(layout.trajectory_path, _build_trajectory(case, layout))
    layout.readme_path.write_text(_build_readme(case), encoding="utf-8")

    return case_dir


__all__ = ["write_case"]
