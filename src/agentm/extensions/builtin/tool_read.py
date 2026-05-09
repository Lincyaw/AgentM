"""Tool atom for the ``extensions.builtin.tool_read`` §7.1 row."""

from __future__ import annotations

from typing import Any, Final

from agentm.core.abi import FunctionTool, TextContent, ToolResult
from agentm.core.abi.operations import FileOperations
from agentm.extensions import ExtensionManifest
from agentm.harness.extension import ExtensionAPI


MANIFEST = ExtensionManifest(
    name="tool_read",
    description="Register the read tool backed by FileOperations.",
    registers=("tool:read",),
    config_schema={
        "type": "object",
        "properties": {
            "file_ops": {"type": "object"},
        },
        "additionalProperties": True,
    },
    requires=(),  # Leaf tool atom: consumes Operations via ExtensionAPI.
)

_PARAMETERS: Final = {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "offset": {"type": "integer", "default": 0},
        "limit": {"type": "integer", "default": 2000},
    },
    "required": ["path"],
    "additionalProperties": False,
}


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    file_ops = _coerce_file_ops(api, config.get("file_ops"))

    async def _execute(args: dict[str, Any]) -> ToolResult:
        path = str(args["path"])
        offset = int(args.get("offset", 0))
        limit = int(args.get("limit", 2000))
        try:
            data = await file_ops.read_file(path)
            lines = data.decode("utf-8", errors="replace").splitlines()
            if limit < 0:
                sliced = lines[offset:]
            else:
                sliced = lines[offset : offset + limit]
            return _ok("\n".join(sliced))
        except Exception as exc:
            return _error(f"Failed to read {path!r}: {exc}")

    api.register_tool(
        FunctionTool(
            name="read",
            description="Read a UTF-8 text file from disk by line range.",
            parameters=_PARAMETERS,
            fn=_execute,
            metadata={"file_op": "read"},
        )
    )


def _coerce_file_ops(api: ExtensionAPI, candidate: Any) -> FileOperations:
    return candidate if candidate is not None else api.get_operations().file


def _ok(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)])


def _error(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)], is_error=True)
