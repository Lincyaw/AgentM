"""Trajectory JSONL query tool via jq.

Provides a thin wrapper around jq so the analysis orchestrator can
run arbitrary structured queries against trajectory JSONL files.

Usage:
    reader = TrajectoryReader()
    reader.register("path/to/trajectory.jsonl")
    reader.jq_query(thread_id, '. | length')
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


class TrajectoryReader:
    """Maps thread_ids to JSONL files and runs jq queries against them."""

    def __init__(self) -> None:
        self._paths: dict[str, Path] = {}

    def register(self, file_path: str | Path) -> str:
        """Register a JSONL file. Returns the thread_id from metadata."""
        path = Path(file_path).resolve()
        meta = _read_meta(path)
        thread_id = meta.get("thread_id", path.stem)
        self._paths[thread_id] = path
        return thread_id

    def jq_query(self, thread_id: str, expression: str, raw: bool = False) -> str:
        """Run a jq expression against a trajectory JSONL file.

        The file is a newline-delimited JSON (JSONL) where the first line
        is a metadata header (has a "_meta" key) and subsequent lines are
        trajectory events. jq is invoked with --slurp so the full file is
        available as an array.

        Args:
            thread_id: The thread ID of the trajectory to query.
            expression: A jq expression. The input is a slurped array of
                all JSON objects in the file.
            raw: If true, pass -r flag for raw string output.
        """
        path = self._paths.get(thread_id)
        if path is None:
            return f"No trajectory registered for thread_id={thread_id!r}."

        cmd = ["jq", "--slurp"]
        if raw:
            cmd.append("-r")
        cmd.append(expression)
        cmd.append(str(path))

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
        if len(output) > 50000:
            return output[:50000] + f"\n... (truncated, {len(output)} chars total)"
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
    """Run a jq expression against a trajectory JSONL file.

    The file is newline-delimited JSON (JSONL) where the first line is a
    metadata header (has a "_meta" key) and subsequent lines are trajectory
    events. jq is invoked with --slurp so the full file is available as
    an array.

    Args:
        thread_id: The thread ID of the trajectory to query.
        expression: A jq expression. The input is a slurped array of all
            JSON objects in the file.
        raw: If true, pass -r flag for raw string output.
    """
    return get_reader().jq_query(thread_id, expression, raw)


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


def _read_meta(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        first_line = f.readline()
    try:
        data = json.loads(first_line)
        return data.get("_meta", {})
    except (json.JSONDecodeError, KeyError):
        return {}
