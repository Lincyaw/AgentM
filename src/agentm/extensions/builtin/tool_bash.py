"""Tool atom for the ``extensions.builtin.tool_bash`` §7.1 row."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Final

from loguru import logger
from pydantic import BaseModel

from agentm.core.abi import (
    BashOperations,
    ExtensionAPI,
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


class _ToolBashRuntime:
    def __init__(self, api: ExtensionAPI, config: ToolBashConfig) -> None:
        self._api = api
        self._bash_ops = _coerce_bash_ops(api, config.bash_ops)
        self._default_timeout = float(config.default_timeout)

    def install(self) -> None:
        self._api.register_tool(
            _BashTool(
                api=self._api,
                bash_ops=self._bash_ops,
                default_timeout=self._default_timeout,
                parameters=self._parameters(),
            )
        )

    def _parameters(self) -> dict[str, Any]:
        return {
            **_PARAMETERS,
            "properties": {
                **_PARAMETERS["properties"],
                "timeout": {"type": "number", "default": self._default_timeout},
            },
        }


def install(api: ExtensionAPI, config: ToolBashConfig) -> None:
    _ToolBashRuntime(api, config).install()


class _BashTool:
    name = "bash"
    description = "Execute a shell command in the session cwd."

    def __init__(
        self,
        *,
        api: ExtensionAPI,
        bash_ops: BashOperations,
        default_timeout: float,
        parameters: dict[str, Any],
    ) -> None:
        self.parameters = parameters
        self._api = api
        self._bash_ops = bash_ops
        self._default_timeout = default_timeout

    async def execute(
        self,
        args: dict[str, Any],
        *,
        signal: asyncio.Event | None = None,
    ) -> ToolResult:
        cmd = str(args["cmd"])
        timeout = float(args.get("timeout", self._default_timeout))
        t0 = time.monotonic()
        try:
            result = await self._bash_ops.exec(
                cmd, cwd=self._api.cwd, timeout=timeout, signal=signal
            )
        except Exception as exc:
            logger.debug("tool_bash: exec failed for {!r}: {}", cmd, exc)
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


def _coerce_bash_ops(api: ExtensionAPI, candidate: Any) -> BashOperations:
    return candidate if candidate is not None else api.get_operations().bash


def _error(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)], is_error=True)
