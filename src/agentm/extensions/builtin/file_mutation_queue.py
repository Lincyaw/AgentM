"""Builtin ``file_mutation_queue`` atom per extension-as-scenario §7."""

from __future__ import annotations

import asyncio
import os
from typing import Any

from pydantic import BaseModel

from agentm.core.abi import (
    AgentStartEvent,
    ExtensionAPI,
    ExtensionLoadError,
    Tool,
    ToolOutcome,
    ToolResult,
)
from agentm.extensions import ExtensionManifest


class FileMutationQueueConfig(BaseModel):
    tools: list[str] = ["edit", "write"]


MANIFEST = ExtensionManifest(
    name="file_mutation_queue",
    description="Serializes file-mutation tools with per-path asyncio locks.",
    registers=("event:agent_start",),
    config_schema=FileMutationQueueConfig,
    requires=("file_tools",),
)

_PATH_KEYS = ("path", "file_path", "filepath", "target")


class _QueuedTool:
    def __init__(self, wrapped: Tool, locks: dict[str, asyncio.Lock]) -> None:
        self._wrapped = wrapped
        self._locks = locks
        self.name = wrapped.name
        self.description = wrapped.description
        self.parameters = wrapped.parameters

    async def execute(
        self,
        args: dict[str, Any],
        *,
        signal: asyncio.Event | None = None,
    ) -> ToolResult | ToolOutcome:
        lock = self._locks.setdefault(_normalize_path(args), asyncio.Lock())
        async with lock:
            return await self._wrapped.execute(args, signal=signal)


def _normalize_path(args: dict[str, Any]) -> str:
    raw = None
    for key in _PATH_KEYS:
        value = args.get(key)
        if isinstance(value, str) and value:
            raw = value
            break
    if raw is None:
        return "<missing-path>"
    return os.path.abspath(raw)


class _FileMutationQueueRuntime:
    def __init__(self, api: ExtensionAPI, config: FileMutationQueueConfig) -> None:
        self._api = api
        self._target_names = tuple(config.tools)
        self._locks: dict[str, asyncio.Lock] = {}
        self._wrapped_names: set[str] = set()

    def install(self) -> None:
        self._api.on(AgentStartEvent.CHANNEL, self.on_agent_start)

    def on_agent_start(self, _: AgentStartEvent) -> ExtensionLoadError | None:
        tools_by_name = {
            tool.name: (index, tool) for index, tool in enumerate(self._api.tools)
        }
        missing = [name for name in self._target_names if name not in tools_by_name]
        if missing:
            return ExtensionLoadError(
                __name__,
                RuntimeError(
                    "file_mutation_queue must load after mutation tools; missing "
                    + ", ".join(sorted(missing))
                ),
            )
        for name in self._target_names:
            if name in self._wrapped_names:
                continue
            index, tool = tools_by_name[name]
            if isinstance(tool, _QueuedTool):
                self._wrapped_names.add(name)
                continue
            self._api.tools[index] = _QueuedTool(tool, self._locks)
            self._wrapped_names.add(name)
        return None


def install(api: ExtensionAPI, config: FileMutationQueueConfig) -> None:
    _FileMutationQueueRuntime(api, config).install()
