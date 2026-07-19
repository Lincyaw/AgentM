"""Builtin ``file_mutation_queue`` atom — serialize file-mutation tools.

Concurrent tool batches can run ``edit`` / ``write`` against the same path
at once and corrupt it. This atom installs a wrapping :class:`ToolExecutor`
that takes a per-path :class:`asyncio.Lock` around the configured mutation
tools, so two calls targeting the same file run one after another while
calls to different paths (or other tools) stay parallel.

The execution boundary is the correct seam for this: it intercepts every
tool call regardless of whether the tool list was later filtered or which
provider assembled it, and it composes with any host-registered executor by
delegating to it.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from pydantic import BaseModel

from agentm.core.abi import (
    AtomAPI,
    AtomInstallPriority,
    CancelSignal,
    ToolExecutionCapabilities,
    ToolExecutionRequest,
    ToolExecutor,
    ToolOutcome,
    ToolResult,
)
from agentm.core.lib.tool_executor import DirectToolExecutor
from agentm.extensions import ExtensionManifest


class FileMutationQueueConfig(BaseModel):
    tools: list[str] = ["edit", "write"]


MANIFEST = ExtensionManifest(
    name="file_mutation_queue",
    description="Serializes file-mutation tools with per-path asyncio locks.",
    registers=("executor:file_mutation_queue",),
    config_schema=FileMutationQueueConfig,
    requires=(),
    priority=AtomInstallPriority.TOOL,
)

_PATH_KEYS = ("path", "file_path", "filepath", "target")


def _normalize_path(args: Any) -> str:
    raw = None
    for key in _PATH_KEYS:
        value = args.get(key) if hasattr(args, "get") else None
        if isinstance(value, str) and value:
            raw = value
            break
    if raw is None:
        return "<missing-path>"
    return os.path.abspath(raw)


class _QueuedExecutor:
    """Wrapping executor that serializes mutation tools per target path."""

    def __init__(self, inner: ToolExecutor, target_names: tuple[str, ...]) -> None:
        self._inner = inner
        self._targets = frozenset(target_names)
        self._locks: dict[str, asyncio.Lock] = {}

    def capabilities(self) -> ToolExecutionCapabilities:
        return self._inner.capabilities()

    async def execute(
        self,
        request: ToolExecutionRequest,
        *,
        signal: CancelSignal | None = None,
    ) -> ToolResult | ToolOutcome:
        if request.tool.name not in self._targets:
            return await self._inner.execute(request, signal=signal)
        lock = self._locks.setdefault(_normalize_path(request.args), asyncio.Lock())
        async with lock:
            return await self._inner.execute(request, signal=signal)


class _FileMutationQueueRuntime:
    def __init__(self, api: AtomAPI, config: FileMutationQueueConfig) -> None:
        self._api = api
        self._target_names = tuple(config.tools)

    def install(self) -> None:
        inner = self._api.get_tool_executor() or DirectToolExecutor()
        wrapper = _QueuedExecutor(inner, self._target_names)
        self._api.register_tool_executor(wrapper, replace=True)


def install(api: AtomAPI, config: FileMutationQueueConfig) -> None:
    _FileMutationQueueRuntime(api, config).install()
