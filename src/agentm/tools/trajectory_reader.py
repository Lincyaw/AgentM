"""Trajectory query tool via jq.

Provides a thin wrapper around jq so the analysis orchestrator can
run arbitrary structured queries against trajectory files in either
JSONL (newline-delimited JSON) or plain JSON format.

Usage:
    reader = TrajectoryReader()
    reader.register("path/to/trajectory.jsonl")   # JSONL with _meta header
    reader.register("path/to/trajectory.json")    # plain JSON with _eval_meta
    reader.jq_query(thread_id, '. | length')
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class _FileFormat(Enum):
    JSONL = "jsonl"
    JSON = "json"


@dataclass(frozen=True)
class _RegisteredFile:
    path: Path
    fmt: _FileFormat


class TrajectoryReader:
    """Maps case IDs to trajectory files and runs jq queries against them."""

    def __init__(self) -> None:
        self._files: dict[str, _RegisteredFile] = {}

    def register(self, file_path: str | Path) -> str:
        """Register a trajectory file. Returns the case ID extracted from metadata.

        Supports two formats:
        - JSONL: first line contains ``{"_meta": {"thread_id": ...}, ...}``
        - JSON: root object contains ``"_eval_meta"`` or ``"trajectories"``
        """
        path = Path(file_path).resolve()
        fmt, case_id = _detect_format_and_id(path)
        self._files[case_id] = _RegisteredFile(path=path, fmt=fmt)
        return case_id

    def register_with_id(self, file_path: str | Path, case_id: str) -> str:
        """Register a file with an explicit ID (e.g. a DB id for batch imports).

        Format detection still occurs so that ``jq_query`` uses the correct
        invocation mode.
        """
        path = Path(file_path).resolve()
        fmt, _ = _detect_format_and_id(path)
        self._files[case_id] = _RegisteredFile(path=path, fmt=fmt)
        return case_id

    def jq_query(self, thread_id: str, expression: str, raw: bool = False) -> str:
        """Run a jq expression against a registered trajectory file.

        For JSONL files the input is a slurped array (``--slurp``).
        For plain JSON files the input is the root object directly.

        Args:
            thread_id: The case ID of the trajectory to query.
            expression: A jq expression.
            raw: If true, pass -r flag for raw string output.
        """
        entry = self._files.get(thread_id)
        if entry is None:
            return f"No trajectory registered for thread_id={thread_id!r}."

        cmd: list[str] = ["jq"]
        if entry.fmt == _FileFormat.JSONL:
            cmd.append("--slurp")
        if raw:
            cmd.append("-r")
        cmd.append(expression)
        cmd.append(str(entry.path))

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except FileNotFoundError:
            return "Error: jq is not installed. Install it with: apt install jq"
        except subprocess.TimeoutExpired:
            return "Error: jq query timed out after 30 seconds."

        if result.returncode != 0:
            return f"jq error: {result.stderr.strip()}"

        output = result.stdout.strip()
        if len(output) > 8000:
            return output[:8000] + f"\n... (truncated, {len(output)} chars total)"
        return output


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_default_reader: TrajectoryReader | None = None


def get_reader() -> TrajectoryReader:
    global _default_reader
    if _default_reader is None:
        _default_reader = TrajectoryReader()
    return _default_reader


def jq_query(thread_id: str, expression: str, raw: bool = False) -> str:
    """Run a jq expression against a registered trajectory file.

    Args:
        thread_id: The case ID of the trajectory to query.
        expression: A jq expression.
        raw: If true, pass -r flag for raw string output.
    """
    return get_reader().jq_query(thread_id, expression, raw)


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


def _detect_format_and_id(path: Path) -> tuple[_FileFormat, str]:
    """Detect file format and extract a case ID from the file.

    Uses file extension as primary signal (.json → JSON, .jsonl → JSONL),
    then falls back to content inspection for the case ID.

    Returns:
        A (format, case_id) tuple.
    """
    fallback_id = path.stem

    # Extension-based format detection (reliable even for pretty-printed JSON)
    if path.suffix == ".json":
        # Read full file to extract _eval_meta.id
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return _FileFormat.JSON, fallback_id
        if isinstance(data, dict):
            eval_meta = data.get("_eval_meta", {})
            if isinstance(eval_meta, dict) and "id" in eval_meta:
                return _FileFormat.JSON, str(eval_meta["id"])
        return _FileFormat.JSON, fallback_id

    # Default: JSONL — read first line for _meta.thread_id
    with open(path, encoding="utf-8") as f:
        first_line = f.readline()
    try:
        data = json.loads(first_line)
    except (json.JSONDecodeError, ValueError):
        return _FileFormat.JSONL, fallback_id
    if isinstance(data, dict) and "_meta" in data:
        meta = data["_meta"]
        thread_id = meta.get("thread_id", fallback_id) if isinstance(meta, dict) else fallback_id
        return _FileFormat.JSONL, thread_id
    return _FileFormat.JSONL, fallback_id
