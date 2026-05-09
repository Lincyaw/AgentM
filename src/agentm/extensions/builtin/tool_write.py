"""Tool atom for the ``extensions.builtin.tool_write`` §7.1 row."""

from __future__ import annotations

from typing import Any, Final

from agentm.core.abi import FunctionTool, TextContent, ToolResult
from agentm.extensions import ExtensionManifest
from agentm.harness.extension import ExtensionAPI


MANIFEST = ExtensionManifest(
    name="tool_write",
    description="Register the write tool backed by ResourceWriter.",
    registers=("tool:write",),
    config_schema={
        "type": "object",
        "properties": {},
        "additionalProperties": True,
    },
    requires=(),  # Leaf tool atom: consumes ResourceWriter via ExtensionAPI.
)

_PARAMETERS: Final = {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "content": {"type": "string"},
        "rationale": {"type": "string", "default": "agent write via tool_write"},
    },
    "required": ["path", "content"],
    "additionalProperties": False,
}


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    del config
    writer = api.get_resource_writer()

    async def _execute(args: dict[str, Any]) -> ToolResult:
        path = str(args["path"])
        content = str(args["content"])
        rationale = str(args.get("rationale", "agent write via tool_write"))
        try:
            result = await writer.write(
                path,
                content.encode("utf-8"),
                rationale=rationale,
            )
            if result.error is not None:
                return _error(result.error)
            return _ok(f"Wrote {len(content)} bytes to {path!r}")
        except Exception as exc:
            return _error(f"Failed to write {path!r}: {exc}")

    api.register_tool(
        FunctionTool(
            name="write",
            description="Write a UTF-8 text file to disk.",
            parameters=_PARAMETERS,
            fn=_execute,
            metadata={"file_op": "write"},
        )
    )


def _ok(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)])


def _error(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)], is_error=True)
