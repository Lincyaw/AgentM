"""Utilities to export eval artifacts for a trajectory run."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

try:
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
except ImportError:
    JsonPlusSerializer = None  # type: ignore[assignment,misc]


def export_eval_result(trajectory_file: str, output_file: str | None = None) -> Path:
    """Export a consolidated eval artifact from one trajectory file.

    Output includes:
    - trajectory metadata (run_id/thread_id/checkpoint_db)
    - resolved case directory (if present in the incident prompt)
    - ground_truth from injection.json (if available)
    - final orchestrator text output (llm_end)
    - final verify verdict text (llm_end)
    - structured_response from checkpoints.db writes table (if available)
    """
    traj_path = Path(trajectory_file)
    if not traj_path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {trajectory_file}")

    meta, events = _load_trajectory(traj_path)
    case_dir = _extract_case_dir(events)
    ground_truth = _load_ground_truth(case_dir)
    final_orchestrator_output = _extract_last_llm_end(events, agent="orchestrator")
    final_verify_output = _extract_last_llm_end(events, agent="verify")
    structured_response = _load_structured_response_from_db(
        checkpoint_db=meta.get("checkpoint_db"),
        thread_id=meta.get("thread_id"),
    )

    payload: dict[str, Any] = {
        "trajectory_file": str(traj_path),
        "run_id": meta.get("run_id"),
        "thread_id": meta.get("thread_id"),
        "checkpoint_db": meta.get("checkpoint_db"),
        "case_dir": str(case_dir) if case_dir else None,
        "ground_truth": ground_truth,
        "final_orchestrator_output": final_orchestrator_output,
        "final_verify_output": final_verify_output,
        "structured_response": structured_response,
    }

    if output_file:
        out_path = Path(output_file)
    else:
        out_path = traj_path.with_suffix(".export.json")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out_path


def export_eval_batch(
    trajectory_dir: str,
    pattern: str = "*.jsonl",
    output_dir: str | None = None,
) -> tuple[list[Path], list[tuple[Path, str]]]:
    """Batch export eval artifacts for all matching trajectory files.

    Returns:
        (success_paths, failed_items)
        - success_paths: exported json paths
        - failed_items: list[(trajectory_path, error_message)]
    """
    base = Path(trajectory_dir)
    if not base.exists() or not base.is_dir():
        raise NotADirectoryError(f"Trajectory directory not found: {trajectory_dir}")

    targets = sorted(
        p for p in base.glob(pattern) if p.is_file() and p.name.endswith(".jsonl")
    )

    success: list[Path] = []
    failed: list[tuple[Path, str]] = []
    out_base = Path(output_dir) if output_dir else None

    for traj in targets:
        try:
            out_path: str | None = None
            if out_base is not None:
                out_path = str(out_base / f"{traj.stem}.export.json")
            exported = export_eval_result(str(traj), out_path)
            success.append(exported)
        except Exception as e:
            failed.append((traj, str(e)))

    return success, failed


def _load_trajectory(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return {}, []

    first = json.loads(lines[0])
    meta = first.get("_meta", {}) if isinstance(first, dict) else {}

    events: list[dict[str, Any]] = []
    for raw in lines[1:]:
        raw = raw.strip()
        if not raw:
            continue
        try:
            item = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            events.append(item)
    return meta, events


def _extract_case_dir(events: list[dict[str, Any]]) -> Path | None:
    pattern = re.compile(
        r"(?:telemetry data is stored in:|Data directory:)\s*`?([^`\n]+eval-data[^`\n]+)`?",
        re.IGNORECASE,
    )
    for event in events:
        data = event.get("data")
        if not isinstance(data, dict):
            continue
        messages = data.get("messages")
        if not isinstance(messages, list):
            continue
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if not isinstance(content, str):
                continue
            match = pattern.search(content)
            if match:
                candidate = Path(match.group(1).strip())
                if candidate.exists():
                    return candidate
                return candidate
    return None


def _load_ground_truth(case_dir: Path | None) -> dict[str, Any] | None:
    if case_dir is None:
        return None
    injection = case_dir / "injection.json"
    if not injection.exists():
        return None
    try:
        data = json.loads(injection.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    gt = data.get("ground_truth")
    return gt if isinstance(gt, dict) else None


def _extract_last_llm_end(events: list[dict[str, Any]], agent: str) -> str | None:
    result: str | None = None
    for event in events:
        if event.get("event_type") != "llm_end":
            continue
        agent_path = event.get("agent_path")
        if not isinstance(agent_path, list) or not agent_path:
            continue
        joined = ">".join(str(x) for x in agent_path)
        if agent == "orchestrator":
            if joined != "orchestrator":
                continue
        else:
            if not any(str(name).startswith(agent) for name in agent_path):
                continue
        data = event.get("data")
        if isinstance(data, dict) and isinstance(data.get("content"), str):
            result = data["content"]
    return result


def _load_structured_response_from_db(
    checkpoint_db: Any,
    thread_id: Any,
) -> Any:
    if not isinstance(checkpoint_db, str) or not checkpoint_db:
        return None
    if not isinstance(thread_id, str) or not thread_id:
        return None

    db_path = Path(checkpoint_db)
    if not db_path.exists():
        return None

    try:
        con = sqlite3.connect(str(db_path))
        cur = con.cursor()
        row = cur.execute(
            """
            SELECT type, value
            FROM writes
            WHERE thread_id = ? AND channel = 'structured_response'
            ORDER BY rowid DESC
            LIMIT 1
            """,
            (thread_id,),
        ).fetchone()
    except sqlite3.Error:
        return None
    finally:
        try:
            con.close()
        except Exception:
            pass

    if row is None:
        return None

    data_type, value = row
    try:
        serde = JsonPlusSerializer()
        return serde.loads_typed((data_type, value))
    except Exception:
        return None
