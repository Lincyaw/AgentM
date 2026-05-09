"""Tool atom for the ``extensions.builtin.tool_find`` search-tools row."""

from __future__ import annotations

import asyncio
import os
from typing import Any, Final

import pathspec

from agentm.core.abi import TextContent, Tool, ToolResult
from agentm.core.abi.operations import FileOperations
from agentm.core.lib.path_utils import load_gitignore_patterns, resolve_to_cwd
from agentm.core.lib.text_truncate import DEFAULT_MAX_BYTES, format_size, truncate_head
from agentm.extensions import ExtensionManifest
from agentm.harness.extension import ExtensionAPI

MANIFEST = ExtensionManifest(
    name="tool_find",
    description="Register the find tool backed by FileOperations.",
    registers=("tool:find",),
    config_schema={
        "type": "object",
        "properties": {"file_ops": {"type": "object"}},
        "additionalProperties": True,
    },
)

_PARAMETERS: Final = {
    "type": "object",
    "properties": {
        "pattern": {"type": "string"},
        "path": {"type": "string", "default": "."},
        "limit": {"type": "integer", "default": 1000},
    },
    "required": ["pattern"],
    "additionalProperties": False,
}


class _FindTool(Tool):
    name = "find"
    description = "Find files and directories by glob pattern."
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
        pattern = str(args["pattern"])
        root = resolve_to_cwd(str(args.get("path", ".")), self._cwd)
        limit = max(1, int(args.get("limit", 1000)))
        if not await self._file_ops.access(root):
            raise Exception(f"Path not found: {root}")

        results, limit_hit = await _find_with_file_ops(
            self._file_ops,
            pattern,
            root,
            limit,
            signal,
        )
        if not results:
            return ToolResult(content=[TextContent(type="text", text="No files found matching pattern")])
        joined = "\n".join(results)
        trunc = truncate_head(joined, max_lines=10**9, max_bytes=DEFAULT_MAX_BYTES)
        output = trunc.content
        notes: list[str] = []
        if limit_hit:
            notes.append(f"{limit} results limit reached")
        if trunc.truncated:
            notes.append(f"{format_size(DEFAULT_MAX_BYTES)} limit reached")
        if notes:
            output += f"\n[Truncated: {', '.join(notes)}]"
        return ToolResult(content=[TextContent(type="text", text=output)])


async def _find_with_file_ops(
    file_ops: FileOperations,
    pattern: str,
    root: str,
    limit: int,
    signal: asyncio.Event | None,
) -> tuple[list[str], bool]:
    ignore = _ignore_spec(root, [".git/", "node_modules/"])
    match_spec = _compile_pattern(pattern)
    results: list[str] = []

    async def _walk(path: str, rel_dir: str) -> bool:
        if signal is not None and signal.is_set():
            raise Exception("Operation aborted")
        for name in sorted(await file_ops.list_dir(path), key=lambda item: (item.lower(), item)):
            rel_path = f"{rel_dir}/{name}".strip("/")
            child = os.path.join(path, name)
            is_dir = await file_ops.is_dir(child)
            if ignore.match_file(rel_path) or (is_dir and ignore.match_file(rel_path + "/")):
                continue
            candidate = rel_path + "/" if is_dir else rel_path
            if match_spec.match_file(rel_path):
                results.append(candidate)
                if len(results) >= limit:
                    return True
            if is_dir and await _walk(child, rel_path):
                return True
        return False

    root_is_dir = await file_ops.is_dir(root)
    if not root_is_dir:
        name = os.path.basename(root)
        if match_spec.match_file(name):
            return [name], False
        return [], False
    limit_hit = await _walk(root, "")
    return results, limit_hit


def _compile_pattern(pattern: str) -> pathspec.GitIgnoreSpec:
    # Bare patterns (no slash) match basename anywhere in the tree.
    expanded = pattern if "/" in pattern else f"**/{pattern}"
    return pathspec.GitIgnoreSpec.from_lines([expanded.lstrip("/")])


def _ignore_spec(root: str, extra: list[str]) -> pathspec.PathSpec:
    return pathspec.GitIgnoreSpec.from_lines(load_gitignore_patterns(root, extra=extra))


def _coerce_file_ops(api: ExtensionAPI, candidate: Any) -> FileOperations:
    return candidate if candidate is not None else api.get_operations().file


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    api.register_tool(_FindTool(api.cwd, _coerce_file_ops(api, config.get("file_ops"))))
