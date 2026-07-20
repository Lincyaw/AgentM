# code-health: ignore-file[AM025] -- atom tools validate untyped tool, config, and service payloads
"""Builtin ``tool_purpose`` atom: require intent on tool calls.

Injects a ``purpose`` parameter into tool schemas sent to the model, then
strips that synthetic argument before delegating to the real tool.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Final

from pydantic import BaseModel, ConfigDict

from agentm.core.abi import (
    AtomAPI,
    AtomInstallPriority,
    BeforeSendEvent,
    CancelSignal,
    Tool,
    ToolExecutionCapabilities,
    ToolExecutionRequest,
    ToolExecutor,
    ToolOutcome,
    ToolResult,
)
from agentm.core.lib.tool_executor import DirectToolExecutor
from agentm.extensions import ExtensionManifest

_FIELD_NAME: Final[str] = "purpose"
_FIELD_SCHEMA: Final[dict[str, object]] = {
    "type": "string",
    "description": "Why you are calling this tool (a few words).",
}


class ToolPurposeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    exclude: list[str] = []


MANIFEST = ExtensionManifest(
    name="tool_purpose",
    description="Add a purpose parameter to every tool so the model states intent.",
    registers=("event:before_send", "executor:tool_purpose"),
    config_schema=ToolPurposeConfig,
    requires=(),
    # Install after background_exec so purpose stripping is the outermost
    # execution-boundary adapter.
    priority=AtomInstallPriority.CONTEXT + 200,
)


@dataclass(slots=True)
class _PurposeExecutor:
    _executor: ToolExecutor
    _injected_tools: set[str]

    def capabilities(self) -> ToolExecutionCapabilities:
        return self._executor.capabilities()

    async def execute(
        self,
        request: ToolExecutionRequest,
        *,
        signal: CancelSignal | None = None,
    ) -> ToolResult | ToolOutcome:
        if request.tool.name not in self._injected_tools:
            return await self._executor.execute(request, signal=signal)
        if _FIELD_NAME not in request.args:
            return await self._executor.execute(request, signal=signal)
        clean_args = dict(request.args)
        clean_args.pop(_FIELD_NAME, None)
        return await self._executor.execute(
            replace(request, args=clean_args),
            signal=signal,
        )


class _ToolPurposeRuntime:
    def __init__(self, api: AtomAPI, exclude: set[str]) -> None:
        self._api = api
        self._exclude = exclude
        self._injected_tools: set[str] = set()

    def install(self) -> None:
        inner = self._api.get_tool_executor() or DirectToolExecutor()
        self._api.register_tool_executor(
            _PurposeExecutor(inner, self._injected_tools),
            replace=True,
        )
        self._api.on(BeforeSendEvent.CHANNEL, self.on_before_send)

    def on_before_send(self, event: BeforeSendEvent) -> None:
        for tool in event.tools:
            self._inject(tool)

    def _inject(self, tool: Tool) -> None:
        if tool.name in self._exclude:
            return
        parameters = tool.parameters
        properties = parameters.get("properties")
        if not isinstance(properties, dict):
            return
        if _FIELD_NAME in properties:
            return
        properties[_FIELD_NAME] = dict(_FIELD_SCHEMA)
        required = parameters.get("required")
        if isinstance(required, list) and _FIELD_NAME not in required:
            required.append(_FIELD_NAME)
        elif isinstance(required, tuple):
            updated = list(required)
            if _FIELD_NAME not in updated:
                updated.append(_FIELD_NAME)
            parameters["required"] = updated
        elif not isinstance(required, list):
            parameters["required"] = [_FIELD_NAME]
        self._injected_tools.add(tool.name)


def install(api: AtomAPI, config: ToolPurposeConfig) -> None:
    _ToolPurposeRuntime(api, {str(name) for name in config.exclude}).install()
