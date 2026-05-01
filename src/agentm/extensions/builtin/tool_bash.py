"""Tool atom for the ``extensions.builtin.tool_bash`` §7.1 row."""

from __future__ import annotations

import json
from typing import Any

from agentm.core.abi import FunctionTool, TextContent, ToolResult
from agentm.core.abi.operations import BashOperations
from agentm.extensions import ExtensionManifest
from agentm.harness.extension import ExtensionAPI


MANIFEST = ExtensionManifest(
    name="tool_bash",
    description="Register the bash tool backed by BashOperations.",
    registers=("tool:bash",),
    config_schema={
        "type": "object",
        "properties": {
            "bash_ops": {"type": "object"},
        },
        "additionalProperties": True,
    },
)

_PARAMETERS = {
    "type": "object",
    "properties": {
        "cmd": {"type": "string"},
        "timeout": {"type": "number", "default": 120.0},
    },
    "required": ["cmd"],
    "additionalProperties": False,
}


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    bash_ops = _coerce_bash_ops(api, config.get("bash_ops"))

    async def _execute(args: dict[str, Any]) -> ToolResult:
        cmd = str(args["cmd"])
        timeout = float(args.get("timeout", 120.0))
        try:
            result = await bash_ops.exec(cmd, cwd=api.cwd, timeout=timeout)
        except Exception as exc:
            return _error(f"Failed to run command {cmd!r}: {exc}")

        payload = {
            "exit_code": result.exit_code,
            "stdout": result.stdout.decode("utf-8", errors="replace"),
            "stderr": result.stderr.decode("utf-8", errors="replace"),
            "timed_out": result.timed_out,
        }
        is_error = result.exit_code != 0 or result.timed_out
        text = json.dumps(payload, default=str, indent=2, sort_keys=True)
        return ToolResult(
            content=[TextContent(type="text", text=text)],
            is_error=is_error,
        )

    api.register_tool(
        FunctionTool(
            name="bash",
            description="Execute a shell command in the session cwd.",
            parameters=_PARAMETERS,
            fn=_execute,
        )
    )


def _coerce_bash_ops(api: ExtensionAPI, candidate: Any) -> BashOperations:
    return candidate if candidate is not None else api.get_operations().bash


def _error(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)], is_error=True)
