"""Tool atom for the ``extensions.builtin.tool_ls`` search-tools row."""

from __future__ import annotations

import asyncio
import os
from typing import Any, Final

from agentm.core.abi import TextContent, Tool, ToolResult
from agentm.core.abi.operations import FileOperations
from agentm.core.lib.path_utils import resolve_to_cwd
from agentm.core.lib.text_truncate import DEFAULT_MAX_BYTES, format_size, truncate_head
from agentm.extensions import ExtensionManifest
from agentm.harness.extension import ExtensionAPI

MANIFEST = ExtensionManifest(
    name="tool_ls",
    description="Register the ls tool backed by FileOperations.",
    registers=("tool:ls",),
    config_schema={
        "type": "object",
        "properties": {"file_ops": {"type": "object"}},
        "additionalProperties": True,
    },
    requires=(),  # Leaf tool atom: consumes Operations via ExtensionAPI.
)

_PARAMETERS: Final = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "default": "."},
        "limit": {"type": "integer", "default": 500},
    },
    "additionalProperties": False,
}


class _LsTool(Tool):
    name = "ls"
    description = "List directory contents with '/' suffixes for directories."
    parameters = _PARAMETERS

    def __init__(self, cwd: str, file_ops: FileOperations) -> None:
        self._cwd = cwd
        self._file_ops = file_ops

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
        if not await self._file_ops.access(path):
            raise Exception(f"Path not found: {path}")
        if not await self._file_ops.is_dir(path):
            raise Exception(f"Not a directory: {path}")

        lines: list[str] = []
        limit_hit = False
        for name in sorted(await self._file_ops.list_dir(path), key=lambda item: (item.lower(), item)):
            if signal is not None and signal.is_set():
                raise Exception("Operation aborted")
            if len(lines) >= limit:
                limit_hit = True
                break
            entry_path = os.path.join(path, name)
            suffix = "/" if await self._file_ops.is_dir(entry_path) else ""
            lines.append(name + suffix)

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


def _coerce_file_ops(api: ExtensionAPI, candidate: Any) -> FileOperations:
    return candidate if candidate is not None else api.get_operations().file


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    api.register_tool(_LsTool(api.cwd, _coerce_file_ops(api, config.get("file_ops"))))
