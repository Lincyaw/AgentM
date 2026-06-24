"""RCA adapter for the generic rescue-window branch runner."""

from __future__ import annotations

import json
import shlex
from collections import defaultdict
from pathlib import Path
from typing import Any


def build_export_commands(
    results_jsonl: Path,
    *,
    out_prefix: Path | None = None,
) -> list[str]:
    """Build existing RCA case-study exporter commands from branch results."""

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _iter_result_rows(results_jsonl):
        if row.get("status") != "succeeded":
            continue
        fork_session_id = row.get("fork_session_id")
        if not isinstance(fork_session_id, str) or not fork_session_id:
            continue
        baseline = row.get("baseline_session_id") or row.get("source_session_id")
        if isinstance(baseline, str) and baseline:
            groups[baseline].append(row)

    commands: list[str] = []
    for baseline, rows in sorted(groups.items()):
        first = rows[0]
        case_id = first.get("case_id")
        prefix = out_prefix or Path("runs/rescue-window/rca-score")
        if len(groups) > 1 or out_prefix is None:
            suffix = f"{case_id or 'case'}-{baseline[:8]}"
            resolved_prefix = prefix.parent / f"{prefix.name}-{suffix}"
        else:
            resolved_prefix = prefix
        parts = [
            "uv",
            "run",
            "--no-sync",
            "python",
            "scripts/export_reminder_case_study.py",
            "--baseline-session",
            baseline,
            "--include-baseline-row",
            "--out-prefix",
            str(resolved_prefix),
        ]
        if isinstance(case_id, str) and case_id:
            parts.extend(["--case-id", case_id])
        for row in sorted(rows, key=lambda item: str(item.get("branch_id") or "")):
            branch_id = str(row.get("branch_id") or row["fork_session_id"])
            parts.extend(["--variant", f"{branch_id}={row['fork_session_id']}"])
        commands.append(" ".join(shlex.quote(part) for part in parts))
    return commands


def _iter_result_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows
