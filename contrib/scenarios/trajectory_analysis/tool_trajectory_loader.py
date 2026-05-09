"""Tool atom for the ``extensions.builtin.tool_trajectory_loader`` §7.1 row."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from agentm.core.abi import FunctionTool, TextContent, ToolResult
from agentm.core.abi.operations import FileOperations
from agentm.extensions import ExtensionManifest
from agentm.harness.extension import ExtensionAPI


MANIFEST = ExtensionManifest(
    name="tool_trajectory_loader",
    description="Register read-only trajectory analysis tools.",
    registers=(
        "tool:load_trajectory",
        "tool:summarize_trajectory",
        "tool:find_event",
        "tool:compare_trajectories",
    ),
    config_schema={
        "type": "object",
        "properties": {
            "file_ops": {"type": "object"},
        },
        "additionalProperties": True,
    },
)

_LOAD_PARAMETERS = {
    "type": "object",
    "properties": {"path": {"type": "string"}},
    "required": ["path"],
    "additionalProperties": False,
}

_SUMMARY_PARAMETERS = {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
    },
    "additionalProperties": False,
}

_FIND_PARAMETERS = {
    "type": "object",
    "properties": {
        "predicate": {"type": "string"},
        "path": {"type": "string"},
    },
    "required": ["predicate"],
    "additionalProperties": False,
}

_COMPARE_PARAMETERS = {
    "type": "object",
    "properties": {
        "a": {"type": "string"},
        "b": {"type": "string"},
    },
    "required": ["a", "b"],
    "additionalProperties": False,
}


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    file_ops = _coerce_file_ops(api, config.get("file_ops"))
    loaded: dict[str, list[dict[str, Any]]] = {}
    current_path: list[str | None] = [None]

    async def _load(args: dict[str, Any]) -> ToolResult:
        path = str(args["path"])
        # Narrow: ``read_file`` raises ``OSError`` on missing/permission,
        # ``_parse_jsonl`` raises ``json.JSONDecodeError`` /
        # ``UnicodeDecodeError`` on malformed payloads. Anything else
        # propagates.
        try:
            events = _parse_jsonl(await file_ops.read_file(path))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            return _error(f"Failed to load trajectory {path!r}: {exc}")
        loaded[path] = events
        current_path[0] = path
        return _json_result(
            {"path": path, "event_count": len(events), "loaded": True}
        )

    async def _summarize(args: dict[str, Any]) -> ToolResult:
        try:
            path, events = _resolve_loaded(loaded, current_path[0], args.get("path"))
        except ValueError as exc:
            return _error(f"Failed to summarize trajectory: {exc}")
        channels = Counter(str(item.get("channel", "unknown")) for item in events)
        return _json_result(
            {
                "path": path,
                "event_count": len(events),
                "channels": dict(sorted(channels.items())),
            }
        )

    async def _find(args: dict[str, Any]) -> ToolResult:
        predicate = str(args["predicate"])
        try:
            _path, events = _resolve_loaded(loaded, current_path[0], args.get("path"))
        except ValueError as exc:
            return _error(f"Failed to find trajectory event: {exc}")
        matches = [event for event in events if _matches(event, predicate)]
        return _json_result(matches)

    async def _compare(args: dict[str, Any]) -> ToolResult:
        try:
            left = loaded[str(args["a"])]
            right = loaded[str(args["b"])]
        except KeyError as exc:
            return _error(f"Unknown loaded trajectory: {exc}")

        left_channels = Counter(str(item.get("channel", "unknown")) for item in left)
        right_channels = Counter(str(item.get("channel", "unknown")) for item in right)
        return _json_result(
            {
                "a": str(args["a"]),
                "b": str(args["b"]),
                "event_count_delta": len(left) - len(right),
                "channel_count_delta": {
                    key: left_channels.get(key, 0) - right_channels.get(key, 0)
                    for key in sorted(set(left_channels) | set(right_channels))
                },
            }
        )

    api.register_tool(
        FunctionTool(
            name="load_trajectory",
            description="Load a trajectory JSONL file into memory.",
            parameters=_LOAD_PARAMETERS,
            fn=_load,
        )
    )
    api.register_tool(
        FunctionTool(
            name="summarize_trajectory",
            description="Summarize a loaded trajectory by event channel.",
            parameters=_SUMMARY_PARAMETERS,
            fn=_summarize,
        )
    )
    api.register_tool(
        FunctionTool(
            name="find_event",
            description="Find loaded trajectory events by simple predicate.",
            parameters=_FIND_PARAMETERS,
            fn=_find,
        )
    )
    api.register_tool(
        FunctionTool(
            name="compare_trajectories",
            description="Compare two loaded trajectories by event counts.",
            parameters=_COMPARE_PARAMETERS,
            fn=_compare,
        )
    )


def _coerce_file_ops(api: ExtensionAPI, candidate: Any) -> FileOperations:
    return candidate if candidate is not None else api.get_operations().file


def _parse_jsonl(data: bytes) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for raw_line in data.decode("utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            events.append(payload)
    return events


def _resolve_loaded(
    loaded: dict[str, list[dict[str, Any]]],
    current_path: str | None,
    requested_path: Any,
) -> tuple[str, list[dict[str, Any]]]:
    path = str(requested_path) if requested_path is not None else current_path
    if path is None or path not in loaded:
        raise ValueError("no trajectory loaded")
    return path, loaded[path]


def _matches(event: dict[str, Any], predicate: str) -> bool:
    key, separator, value = predicate.partition("=")
    if separator:
        target = event.get(key)
        return str(target) == value
    return predicate in json.dumps(event, default=str, sort_keys=True)


def _json_result(payload: Any) -> ToolResult:
    return ToolResult(
        content=[
            TextContent(
                type="text",
                text=json.dumps(payload, default=str, indent=2, sort_keys=True),
            )
        ]
    )


def _error(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)], is_error=True)
