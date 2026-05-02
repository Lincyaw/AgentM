"""Tool atom for the ``extensions.builtin.tool_edit`` §7.1 row."""

from __future__ import annotations

from typing import Any

from agentm.core.abi import FunctionTool, TextContent, ToolResult
from agentm.core.abi.operations import FileOperations
from agentm.extensions import ExtensionManifest
from agentm.harness.extension import ExtensionAPI


MANIFEST = ExtensionManifest(
    name="tool_edit",
    description="Register the edit tool backed by FileOperations.",
    registers=("tool:edit",),
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
        "old_string": {"type": "string"},
        "new_string": {"type": "string"},
        "replace_all": {"type": "boolean", "default": False},
        "rationale": {"type": "string", "default": "agent edit via tool_edit"},
    },
    "required": ["path", "old_string", "new_string"],
    "additionalProperties": False,
}


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    file_ops = _coerce_file_ops(api, config.get("file_ops"))
    writer = api.get_resource_writer()

    async def _execute(args: dict[str, Any]) -> ToolResult:
        path = str(args["path"])
        old_string = str(args["old_string"])
        new_string = str(args["new_string"])
        replace_all = bool(args.get("replace_all", False))
        rationale = str(args.get("rationale", "agent edit via tool_edit"))

        if not old_string:
            return _error("old_string must not be empty")

        try:
            original = (await file_ops.read_file(path)).decode(
                "utf-8", errors="replace"
            )
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
                path,
                original.encode("utf-8"),
                updated.encode("utf-8"),
                rationale=rationale,
            )
            if result.error is not None:
                return _error(result.error)
            return _ok(f"Updated {path!r}")
        except Exception as exc:
            return _error(f"Failed to edit {path!r}: {exc}")

    api.register_tool(
        FunctionTool(
            name="edit",
            description="Replace text in a UTF-8 text file.",
            parameters=_PARAMETERS,
            fn=_execute,
        )
    )


def _coerce_file_ops(api: ExtensionAPI, candidate: Any) -> FileOperations:
    return candidate if candidate is not None else api.get_operations().file


def _ok(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)])


def _error(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)], is_error=True)
