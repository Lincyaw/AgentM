"""Tool atom for the ``extensions.builtin.tool_ls`` search-tools row."""

from __future__ import annotations

import asyncio
import os
import stat
from typing import Any, Protocol

from agentm.core.kernel import TextContent, Tool, ToolResult
from agentm.core.path_utils import resolve_to_cwd
from agentm.core.text_truncate import DEFAULT_MAX_BYTES, format_size, truncate_head
from agentm.extensions import ExtensionManifest
from agentm.harness.extension import ExtensionAPI

MANIFEST = ExtensionManifest(
    name="tool_ls",
    description="Register the ls tool backed by LsOperations.",
    registers=("tool:ls",),
    config_schema={
        "type": "object",
        "properties": {"ops": {"type": "object"}},
        "additionalProperties": True,
    },
)

_PARAMETERS = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "default": "."},
        "limit": {"type": "integer", "default": 500},
    },
    "additionalProperties": False,
}


class LsOperations(Protocol):
    async def exists(self, path: str) -> bool: ...

    async def stat(self, path: str) -> os.stat_result: ...

    async def listdir(self, path: str) -> list[str]: ...


class _LocalLsOperations:
    async def exists(self, path: str) -> bool:
        return await asyncio.to_thread(os.path.exists, path)

    async def stat(self, path: str) -> os.stat_result:
        return await asyncio.to_thread(os.stat, path)

    async def listdir(self, path: str) -> list[str]:
        def _scan() -> list[str]:
            with os.scandir(path) as entries:
                return [entry.name for entry in entries]

        return await asyncio.to_thread(_scan)


class _LsTool(Tool):
    name = "ls"
    description = "List directory contents with '/' suffixes for directories."
    parameters = _PARAMETERS

    def __init__(self, cwd: str, ops: LsOperations) -> None:
        self._cwd = cwd
        self._ops = ops

    async def execute(
        self,
        args: dict[str, Any],
        *,
        signal: asyncio.Event | None = None,
        on_update: Any = None,
    ) -> ToolResult:
        del on_update
        if signal is not None and signal.is_set():
            raise Exception("Operation aborted")
        path = resolve_to_cwd(str(args.get("path", ".")), self._cwd)
        limit = max(1, int(args.get("limit", 500)))
        if not await self._ops.exists(path):
            raise Exception(f"Path not found: {path}")
        if not _is_dir(await self._ops.stat(path)):
            raise Exception(f"Not a directory: {path}")

        lines: list[str] = []
        limit_hit = False
        for name in sorted(await self._ops.listdir(path), key=lambda item: (item.lower(), item)):
            if signal is not None and signal.is_set():
                raise Exception("Operation aborted")
            if len(lines) >= limit:
                limit_hit = True
                break
            entry_path = os.path.join(path, name)
            lines.append(name + ("/" if _is_dir(await self._ops.stat(entry_path)) else ""))

        output = "\n".join(lines) if lines else "(empty directory)"
        trunc = truncate_head(output, max_lines=10**9, max_bytes=DEFAULT_MAX_BYTES)
        text = trunc.content
        notes: list[str] = []
        if limit_hit:
            notes.append(f"{limit} entries limit reached")
        if trunc.truncated:
            notes.append(f"{format_size(DEFAULT_MAX_BYTES)} limit reached")
        if notes:
            text += f"\n[Truncated: {', '.join(notes)}]"
        return ToolResult(content=[TextContent(type="text", text=text)])


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    api.register_tool(_LsTool(api.cwd, config.get("ops") or _LocalLsOperations()))


def _is_dir(info: os.stat_result) -> bool:
    return stat.S_ISDIR(info.st_mode)
