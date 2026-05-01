"""Trajectory reader tool atom for single-pass RCA trajectory inspection."""

from __future__ import annotations

import contextvars
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from agentm.core.kernel import FunctionTool, TextContent, ToolResult
from agentm.extensions import ExtensionManifest
from agentm.harness.extension import ExtensionAPI


class _FileFormat(Enum):
    JSONL = "jsonl"
    JSON = "json"


@dataclass(frozen=True)
class _RegisteredFile:
    path: Path
    fmt: _FileFormat


class TrajectoryReader:
    """Maps case IDs to trajectory files and evaluates jq-like queries."""

    def __init__(self) -> None:
        self._files: dict[str, _RegisteredFile] = {}

    def register(self, file_path: str | Path) -> str:
        """Register a trajectory file and return the extracted case id."""
        path = Path(file_path).resolve()
        fmt, case_id = _detect_format_and_id(path)
        self._files[case_id] = _RegisteredFile(path=path, fmt=fmt)
        return case_id

    def read(self, thread_id: str, expression: str = ".", raw: bool = False) -> str:
        """Evaluate a small jq-like expression against a registered file."""
        entry = self._files.get(thread_id)
        if entry is None:
            return f"No trajectory registered for thread_id={thread_id!r}."

        try:
            data = _load_registered_file(entry)
            value = _evaluate_expression(data, expression)
        except Exception as exc:  # noqa: BLE001
            return f"jq error: {exc}"

        rendered = _render_value(value, raw=raw)
        if len(rendered) > 8000:
            return rendered[:8000] + f"\n... (truncated, {len(rendered)} chars total)"
        return rendered


_reader_var: contextvars.ContextVar[TrajectoryReader | None] = contextvars.ContextVar(
    "trajectory_reader", default=None
)


def get_reader() -> TrajectoryReader:
    reader = _reader_var.get()
    if reader is None:
        reader = TrajectoryReader()
        _reader_var.set(reader)
    return reader


def jq_query(thread_id: str, expression: str, raw: bool = False) -> str:
    return get_reader().read(thread_id, expression, raw)


MANIFEST = ExtensionManifest(
    name="trajectory_reader",
    description="Register a trajectory reader tool for RCA trajectory files.",
    registers=("tool:jq_query",),
    config_schema={
        "type": "object",
        "properties": {
            "paths": {
                "type": "array",
                "items": {"type": "string"},
            }
        },
        "additionalProperties": False,
    },
)


_PARAMETERS = {
    "type": "object",
    "properties": {
        "thread_id": {"type": "string"},
        "path": {"type": "string"},
        "expression": {"type": "string", "default": "."},
        "raw": {"type": "boolean", "default": False},
    },
    "anyOf": [
        {"required": ["thread_id"]},
        {"required": ["path"]},
    ],
    "additionalProperties": False,
}


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    reader = get_reader()
    for raw_path in config.get("paths", []):
        reader.register(raw_path)

    async def _jq_query(args: dict[str, Any]) -> ToolResult:
        thread_id = args.get("thread_id")
        if thread_id is None:
            thread_id = reader.register(str(args["path"]))
        text = reader.read(
            str(thread_id),
            str(args.get("expression", ".")),
            bool(args.get("raw", False)),
        )
        return ToolResult(content=[TextContent(type="text", text=text)])

    api.register_tool(
        FunctionTool(
            name="jq_query",
            description=(
                "Read a trajectory JSONL/JSON file and evaluate a small jq-like "
                "expression against it. Provide either a registered thread_id or "
                "a path to register on demand."
            ),
            parameters=_PARAMETERS,
            fn=_jq_query,
        )
    )


def _detect_format_and_id(path: Path) -> tuple[_FileFormat, str]:
    fallback_id = path.stem

    if path.suffix == ".json":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return _FileFormat.JSON, fallback_id
        if isinstance(data, dict):
            eval_meta = data.get("_eval_meta", {})
            if isinstance(eval_meta, dict) and "id" in eval_meta:
                return _FileFormat.JSON, str(eval_meta["id"])
        return _FileFormat.JSON, fallback_id

    try:
        with path.open(encoding="utf-8") as handle:
            first_line = handle.readline()
    except OSError:
        return _FileFormat.JSONL, fallback_id
    try:
        data = json.loads(first_line)
    except (json.JSONDecodeError, ValueError):
        return _FileFormat.JSONL, fallback_id
    if isinstance(data, dict) and "_meta" in data:
        meta = data["_meta"]
        thread_id = meta.get("thread_id", fallback_id) if isinstance(meta, dict) else fallback_id
        return _FileFormat.JSONL, str(thread_id)
    return _FileFormat.JSONL, fallback_id


def _load_registered_file(entry: _RegisteredFile) -> Any:
    if entry.fmt is _FileFormat.JSON:
        return json.loads(entry.path.read_text(encoding="utf-8"))

    rows: list[Any] = []
    with entry.path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # Preserve the old "don't crash on bad metadata / partial lines"
                # posture by skipping malformed records instead of aborting.
                continue
    return rows


def _evaluate_expression(data: Any, expression: str) -> Any:
    expr = expression.strip() or "."
    if expr == ".":
        return data
    if expr == ". | length":
        return len(data)

    cursor = data
    remaining = expr
    while remaining:
        if remaining == ".":
            return cursor
        if remaining.startswith(". | length"):
            cursor = len(cursor)
            remaining = remaining[len(". | length") :]
            continue
        if remaining.startswith(".["):
            end = remaining.find("]")
            if end == -1:
                raise ValueError(f"unsupported expression: {expression!r}")
            index_text = remaining[2:end]
            if not isinstance(cursor, list):
                raise ValueError("cannot index non-list value")
            cursor = cursor[int(index_text)]
            remaining = remaining[end + 1 :]
            continue
        if remaining.startswith("."):
            remaining = remaining[1:]
            if not remaining:
                return cursor
            key = remaining
            dot = remaining.find(".")
            bracket = remaining.find("[")
            split_points = [point for point in (dot, bracket) if point != -1]
            if split_points:
                key = remaining[: min(split_points)]
                remaining = remaining[len(key) :]
            else:
                remaining = ""
            if not isinstance(cursor, dict):
                raise ValueError("cannot access key on non-object value")
            cursor = cursor[key]
            continue
        raise ValueError(f"unsupported expression: {expression!r}")
    return cursor


def _render_value(value: Any, *, raw: bool) -> str:
    if raw and not isinstance(value, (dict, list)):
        return str(value)
    if isinstance(value, str):
        return value if raw else json.dumps(value)
    return json.dumps(value, indent=2, sort_keys=True, default=str)
