"""Tool atom for the ``extensions.builtin.tool_write`` §7.1 row.

Enforces read-before-write for existing files: a file that already exists
must have been read via tool_read before it can be overwritten. New files
(path does not exist yet) can be written freely. This prevents blind
overwrites that discard content the agent has never seen.
"""

from __future__ import annotations

import os
from typing import Any, Final

from agentm.core.abi import FunctionTool, TextContent, ToolResult
from agentm.core.lib.read_state import get_read_state, record_read
from agentm.extensions import ExtensionManifest
from agentm.core.abi.extension import ExtensionAPI


MANIFEST = ExtensionManifest(
    name="tool_write",
    description="Register the write tool backed by ResourceWriter.",
    registers=("tool:write",),
    config_schema={
        "type": "object",
        "properties": {
            "require_read": {
                "type": "boolean",
                "default": True,
                "description": "Require existing files to be read before overwriting.",
            },
        },
        "additionalProperties": True,
    },
    requires=(),
)

_PARAMETERS: Final = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "File path to write."},
        "content": {"type": "string", "description": "Content to write."},
        "rationale": {"type": "string", "default": "agent write via tool_write"},
    },
    "required": ["path", "content"],
    "additionalProperties": False,
}


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    writer = api.get_resource_writer()
    require_read = bool(config.get("require_read", True))

    async def _execute(args: dict[str, Any]) -> ToolResult:
        path = str(args["path"])
        content = str(args["content"])
        rationale = str(args.get("rationale", "agent write via tool_write"))

        normalized = os.path.normpath(path)

        # Check if the file already exists — existing files need a prior read.
        file_exists = False
        try:
            await writer.read(path)
            file_exists = True
        except Exception:
            pass

        if file_exists and require_read and get_read_state(normalized) is None:
            return _error(
                f"File {path!r} already exists. Read it first before overwriting "
                "so you can see its current content. Use the read tool, then write."
            )

        try:
            result = await writer.write(
                path,
                content.encode("utf-8"),
                rationale=rationale,
            )
            if result.error is not None:
                return _error(result.error)

            total_lines = content.count("\n") + (1 if content else 0)
            record_read(normalized, total_lines=total_lines, is_partial=False)

            action = "Updated" if file_exists else "Created"
            return _ok(f"{action} {path!r} ({len(content)} bytes)")
        except Exception as exc:
            return _error(f"Failed to write {path!r}: {exc}")

    api.register_tool(
        FunctionTool(
            name="write",
            description=(
                "Write a UTF-8 text file to disk. For existing files, you MUST "
                "read the file first — use this tool only for creating new files "
                "or for complete rewrites after reading."
            ),
            parameters=_PARAMETERS,
            fn=_execute,
            metadata={"file_op": "write"},
        )
    )


def _ok(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)])


def _error(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)], is_error=True)
