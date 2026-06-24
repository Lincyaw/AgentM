"""Input helpers for session-id driven rescue-window experiments."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

__all__ = [
    "read_existing_source_ids",
    "read_session_file",
    "unique_preserve_order",
]


def read_session_file(path: Path, *, column: str) -> list[str]:
    """Read session ids from plain text, CSV, or TSV input."""

    lines = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not lines:
        return []

    first = lines[0]
    delimiter = "\t" if "\t" in first else "," if "," in first else None
    if delimiter is not None:
        header = next(csv.reader([first], delimiter=delimiter))
        if column in header:
            reader = csv.DictReader(lines, delimiter=delimiter)
            return [
                value
                for row in reader
                if (value := (row.get(column) or "").strip())
            ]
        if len(header) > 1:
            raise ValueError(
                f"{path} looks tabular but has no column {column!r}; "
                f"available columns: {', '.join(header)}"
            )

    ids: list[str] = []
    header_names = {column, "session_id", "baseline_session_id"}
    for line in lines:
        value = line.split()[0].strip()
        if value and value not in header_names:
            ids.append(value)
    return ids


def unique_preserve_order(values: list[str]) -> list[str]:
    """Deduplicate values without changing the first-seen order."""

    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def read_existing_source_ids(path: Path) -> set[str]:
    """Read already-processed source/baseline session ids from result JSONL."""

    if not path.exists():
        return set()
    ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload: Any = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        for key in ("baseline_session_id", "source_session_id"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                ids.add(value)
    return ids
