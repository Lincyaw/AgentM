"""Tool atom for the ``extensions.builtin.tool_write`` §7.1 row."""

from __future__ import annotations

from typing import Any

from agentm.core.kernel import FunctionTool, TextContent, ToolResult
from agentm.core.operations import FileOperations, LocalFileOperations
from agentm.extensions import ExtensionManifest
from agentm.harness.extension import ExtensionAPI


MANIFEST = ExtensionManifest(
    name="tool_write",
    description="Register the write tool backed by FileOperations.",
    registers=("tool:write",),
    config_schema={
        "type": "object",
        "properties": {
            "file_ops": {"type": "object"},
        },
        "additionalProperties": True,
    },
)

_PARAMETERS = {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "content": {"type": "string"},
    },
    "required": ["path", "content"],
    "additionalProperties": False,
}


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    file_ops = _coerce_file_ops(config.get("file_ops"))

    async def _execute(args: dict[str, Any]) -> ToolResult:
        path = str(args["path"])
        content = str(args["content"])
        try:
            await file_ops.write_file(path, content.encode("utf-8"))
            return _ok(f"Wrote {len(content)} bytes to {path!r}")
        except Exception as exc:
            return _error(f"Failed to write {path!r}: {exc}")

    api.register_tool(
        FunctionTool(
            name="write",
            description="Write a UTF-8 text file to disk.",
            parameters=_PARAMETERS,
            fn=_execute,
        )
    )


def _coerce_file_ops(candidate: Any) -> FileOperations:
    return candidate if candidate is not None else LocalFileOperations()


def _ok(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)])


def _error(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)], is_error=True)
