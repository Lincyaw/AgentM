"""Tool atom for the ``extensions.builtin.tool_bash`` §7.1 row."""

from __future__ import annotations

import time
from typing import Any, Final

from pydantic import BaseModel

from agentm.core.abi import (
    BashOperations,
    ExtensionAPI,
    FunctionTool,
    TextContent,
    ToolResult,
)
from agentm.extensions import ExtensionManifest

_DEFAULT_TIMEOUT_SECONDS: Final[float] = 120.0

class ToolBashConfig(BaseModel):
    model_config = {"extra": "allow"}

    bash_ops: Any = None
    default_timeout: float = _DEFAULT_TIMEOUT_SECONDS

MANIFEST = ExtensionManifest(
    name="tool_bash",
    description="Register the bash tool backed by BashOperations.",
    registers=("tool:bash",),
    config_schema=ToolBashConfig,
    requires=(),  # Leaf tool atom: consumes Operations via ExtensionAPI.
)

_PARAMETERS: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "cmd": {"type": "string"},
        "timeout": {"type": "number"},
    },
    "required": ["cmd"],
    "additionalProperties": False,
}

def install(api: ExtensionAPI, config: ToolBashConfig) -> None:
    bash_ops = _coerce_bash_ops(api, config.bash_ops)
    default_timeout = float(config.default_timeout)

    parameters = {
        **_PARAMETERS,
        "properties": {
            **_PARAMETERS["properties"],
            "timeout": {"type": "number", "default": default_timeout},
        },
    }

    async def _execute(args: dict[str, Any]) -> ToolResult:
        cmd = str(args["cmd"])
        timeout = float(args.get("timeout", default_timeout))
        t0 = time.monotonic()
        try:
            result = await bash_ops.exec(cmd, cwd=api.cwd, timeout=timeout)
        except Exception as exc:
            return _error(f"Failed to run command {cmd!r}: {exc}")
        wall_time = round(time.monotonic() - t0, 1)

        stdout = result.stdout.decode("utf-8", errors="replace")
        stderr = result.stderr.decode("utf-8", errors="replace")
        is_error = result.exit_code != 0 or result.timed_out

        sections: list[str] = []
        sections.append(f"Exit code: {result.exit_code}")
        sections.append(f"Wall time: {wall_time}s")
        if result.timed_out:
            sections.append(f"TIMED OUT after {timeout}s")

        stdout_lines = stdout.count("\n") + (1 if stdout else 0)
        stderr_lines = stderr.count("\n") + (1 if stderr else 0)
        sections.append(f"Stdout lines: {stdout_lines}")
        if stderr_lines:
            sections.append(f"Stderr lines: {stderr_lines}")

        if stdout:
            sections.append(f"Stdout:\n{stdout}")
        if stderr:
            sections.append(f"Stderr:\n{stderr}")

        text = "\n".join(sections)
        return ToolResult(
            content=[TextContent(type="text", text=text)],
            is_error=is_error,
        )

    api.register_tool(
        FunctionTool(
            name="bash",
            description="Execute a shell command in the session cwd.",
            parameters=parameters,
            fn=_execute,
        )
    )

def _coerce_bash_ops(api: ExtensionAPI, candidate: Any) -> BashOperations:
    return candidate if candidate is not None else api.get_operations().bash

def _error(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)], is_error=True)
