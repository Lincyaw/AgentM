"""Tool atom for the ``extensions.builtin.tool_edit`` §7.1 row.

Enforces read-before-edit: a file must have been read via tool_read in
the current session before it can be edited. Supports two modes:

1. **String replacement** (``old_string`` + ``new_string``): classic
   search-replace, same as before.
2. **Line-range replacement** (``start_line`` + ``end_line`` + ``new_string``):
   replace lines [start, end] (1-based, inclusive) with new_string.
   Requires a prior read so the agent has seen the line numbers.
"""

from __future__ import annotations

from typing import Any, Final

from agentm.core.abi import FunctionTool, TextContent, ToolResult
from agentm.core.abi.tool import TOOL_RESULT_FORMAT_METADATA_KEY
from agentm.extensions import ExtensionManifest
from agentm.core.abi.extension import ExtensionAPI


MANIFEST = ExtensionManifest(
    name="tool_edit",
    description="Register the edit tool backed by ResourceWriter.",
    registers=("tool:edit",),
    config_schema={
        "type": "object",
        "properties": {
            "require_read": {
                "type": "boolean",
                "default": True,
                "description": "Require the file to be read before editing.",
            },
        },
        "additionalProperties": True,
    },
    requires=(),
)

_PARAMETERS: Final = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "File path to edit."},
        "old_string": {
            "type": "string",
            "description": "Exact text to find and replace. Mutually exclusive with start_line/end_line.",
        },
        "new_string": {
            "type": "string",
            "description": "Replacement text.",
        },
        "start_line": {
            "type": "integer",
            "description": "1-based start line for line-range replacement (inclusive). Use with end_line instead of old_string.",
        },
        "end_line": {
            "type": "integer",
            "description": "1-based end line for line-range replacement (inclusive).",
        },
        "replace_all": {"type": "boolean", "default": False},
        "rationale": {"type": "string", "default": "agent edit via tool_edit"},
    },
    "required": ["path", "new_string"],
    "additionalProperties": False,
}


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    writer = api.get_resource_writer()
    require_read = bool(config.get("require_read", True))
    read_files: set[str] = set()

    # Track reads from tool_read via the event bus.
    from agentm.core.abi import ToolCallEvent

    def _on_tool_call(event: ToolCallEvent) -> None:
        if event.name == "read":
            path = event.arguments.get("path")
            if path:
                read_files.add(str(path))

    api.on(ToolCallEvent.CHANNEL, _on_tool_call)

    async def _execute(args: dict[str, Any]) -> ToolResult:
        path = str(args["path"])
        new_string = str(args["new_string"])
        old_string = args.get("old_string")
        start_line = args.get("start_line")
        end_line = args.get("end_line")
        replace_all = bool(args.get("replace_all", False))
        rationale = str(args.get("rationale", "agent edit via tool_edit"))

        if require_read and path not in read_files:
            return _error(
                f"You must read {path!r} before editing it. "
                "Use the read tool first so you can see the exact content and line numbers."
            )

        has_old = old_string is not None and old_string != ""
        has_lines = start_line is not None and end_line is not None

        if has_old and has_lines:
            return _error("Provide either old_string OR start_line/end_line, not both.")
        if not has_old and not has_lines:
            return _error("Provide old_string or start_line + end_line.")

        try:
            original = (await writer.read(path)).decode("utf-8", errors="replace")

            if has_lines:
                return await _line_range_replace(
                    writer, path, original, int(start_line), int(end_line),
                    new_string, rationale,
                )
            else:
                return await _string_replace(
                    writer, path, original, str(old_string), new_string,
                    replace_all, rationale,
                )
        except Exception as exc:
            return _error(f"Failed to edit {path!r}: {exc}")

    api.register_tool(
        FunctionTool(
            name="edit",
            description=(
                "Edit a UTF-8 text file. Two modes:\n"
                "1. String replacement: provide old_string + new_string.\n"
                "2. Line-range replacement: provide start_line + end_line + new_string "
                "(1-based, inclusive). You MUST read the file first to see line numbers."
            ),
            parameters=_PARAMETERS,
            fn=_execute,
            metadata={"file_op": "edit", TOOL_RESULT_FORMAT_METADATA_KEY: "diff"},
        )
    )


async def _string_replace(
    writer: Any,
    path: str,
    original: str,
    old_string: str,
    new_string: str,
    replace_all: bool,
    rationale: str,
) -> ToolResult:
    occurrences = original.count(old_string)
    if occurrences == 0:
        return _error(f"String not found in {path!r}: {old_string!r}")
    if not replace_all and occurrences != 1:
        return _error(
            f"String is not unique in {path!r}: found {occurrences} matches"
        )
    updated = (
        original.replace(old_string, new_string)
        if replace_all
        else original.replace(old_string, new_string, 1)
    )
    result = await writer.replace(
        path, original.encode("utf-8"), updated.encode("utf-8"), rationale=rationale,
    )
    if result.error is not None:
        return _error(result.error)
    return _ok(f"Updated {path!r}")


async def _line_range_replace(
    writer: Any,
    path: str,
    original: str,
    start: int,
    end: int,
    new_string: str,
    rationale: str,
) -> ToolResult:
    lines = original.splitlines(keepends=True)
    total = len(lines)
    if start < 1 or end < start or start > total:
        return _error(
            f"Invalid line range [{start}, {end}] for {path!r} ({total} lines). "
            "Lines are 1-based."
        )
    end = min(end, total)
    before = lines[: start - 1]
    after = lines[end:]
    if new_string and not new_string.endswith("\n"):
        new_string += "\n"
    updated = "".join(before) + new_string + "".join(after)
    result = await writer.replace(
        path, original.encode("utf-8"), updated.encode("utf-8"), rationale=rationale,
    )
    if result.error is not None:
        return _error(result.error)
    return _ok(f"Replaced lines {start}-{end} in {path!r}")


def _ok(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)])


def _error(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)], is_error=True)
