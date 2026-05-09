"""Tool atom for the ``extensions.builtin.tool_grep`` search-tools row."""

from __future__ import annotations

import asyncio
import os
import re
from typing import Any, Final

import pathspec

from agentm.core.abi import TextContent, Tool, ToolResult
from agentm.core.abi.operations import FileOperations
from agentm.core.lib.path_utils import load_gitignore_patterns, resolve_to_cwd
from agentm.core.lib.text_truncate import (
    DEFAULT_MAX_BYTES,
    GREP_MAX_LINE_LENGTH,
    format_size,
    truncate_head,
    truncate_line,
)
from agentm.extensions import ExtensionManifest
from agentm.harness.extension import ExtensionAPI

MANIFEST = ExtensionManifest(
    name="tool_grep",
    description="Register the grep tool backed by FileOperations.",
    registers=("tool:grep",),
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
        "path": {"type": "string"},
        "glob": {"type": "string"},
        "ignore_case": {"type": "boolean", "default": False},
        "literal": {"type": "boolean", "default": False},
        "context": {"type": "integer", "default": 0},
        "limit": {"type": "integer", "default": 100},
    },
    "required": ["pattern"],
    "additionalProperties": False,
}


class _GrepTool(Tool):
    name = "grep"
    description = "Search files for regex or literal matches, honoring .gitignore."
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
        search_path = resolve_to_cwd(str(args.get("path", ".")), self._cwd)
        glob = str(args["glob"]) if args.get("glob") is not None else None
        context = max(0, int(args.get("context", 0)))
        limit = max(1, int(args.get("limit", 100)))
        is_dir = await self._classify_path(search_path)
        hits = await _grep_with_file_ops(
            self._file_ops,
            pattern,
            search_path,
            glob,
            bool(args.get("ignore_case", False)),
            bool(args.get("literal", False)),
            limit,
            is_dir,
            signal,
        )
        lines, line_cut = await _render(
            self._file_ops,
            search_path,
            is_dir,
            hits[:limit],
            context,
            self._cwd,
        )
        if not lines:
            return ToolResult(content=[TextContent(type="text", text="No matches found")])
        trunc = truncate_head("\n".join(lines), max_lines=10**9, max_bytes=DEFAULT_MAX_BYTES)
        text = trunc.content
        notes: list[str] = []
        if len(hits) >= limit:
            notes.append(f"{limit} matches limit reached")
        if trunc.truncated:
            notes.append(f"{format_size(DEFAULT_MAX_BYTES)} limit reached")
        if line_cut:
            notes.append("some lines truncated")
        if notes:
            text += f"\n[Truncated: {', '.join(notes)}]"
        return ToolResult(content=[TextContent(type="text", text=text)])

    async def _classify_path(self, search_path: str) -> bool:
        if not await self._file_ops.access(search_path):
            raise Exception(f"Path not found: {search_path}")
        return await self._file_ops.is_dir(search_path)


async def _grep_with_file_ops(
    file_ops: FileOperations,
    pattern: str,
    search_path: str,
    glob: str | None,
    ignore_case: bool,
    literal: bool,
    limit: int,
    is_dir: bool,
    signal: asyncio.Event | None,
) -> list[tuple[str, int]]:
    compiled = re.compile(
        re.escape(pattern) if literal else pattern,
        re.IGNORECASE if ignore_case else 0,
    )
    files = [search_path] if not is_dir else await _list_search_files(file_ops, search_path, glob, signal)
    hits: list[tuple[str, int]] = []
    for file_path in files:
        if signal is not None and signal.is_set():
            raise Exception("Operation aborted")
        try:
            text = (await file_ops.read_file(file_path)).decode("utf-8", errors="replace")
        except Exception:
            continue
        for line_no, line in enumerate(_split_lines(text), start=1):
            if compiled.search(line):
                hits.append((file_path, line_no))
                if len(hits) >= limit:
                    return hits
    return hits


async def _list_search_files(
    file_ops: FileOperations,
    root: str,
    glob: str | None,
    signal: asyncio.Event | None,
) -> list[str]:
    ignore = pathspec.GitIgnoreSpec.from_lines(load_gitignore_patterns(root, extra=[".git/"]))
    glob_spec = _compile_glob(glob) if glob else None
    files: list[str] = []

    async def _walk(path: str, rel_dir: str) -> None:
        if signal is not None and signal.is_set():
            raise Exception("Operation aborted")
        for name in sorted(await file_ops.list_dir(path), key=lambda item: (item.lower(), item)):
            rel_path = f"{rel_dir}/{name}".strip("/")
            child = os.path.join(path, name)
            is_dir = await file_ops.is_dir(child)
            if ignore.match_file(rel_path) or (is_dir and ignore.match_file(rel_path + "/")):
                continue
            if is_dir:
                await _walk(child, rel_path)
                continue
            if glob_spec is not None and not glob_spec.match_file(rel_path):
                continue
            files.append(child)

    await _walk(root, "")
    return files


async def _render(
    file_ops: FileOperations,
    root: str,
    is_dir: bool,
    hits: list[tuple[str, int]],
    context: int,
    cwd: str,
) -> tuple[list[str], bool]:
    cache: dict[str, list[str]] = {}
    output: list[str] = []
    line_cut = False
    for offset, (file_path, line_no) in enumerate(sorted(hits, key=lambda item: (item[0], item[1]))):
        lines = cache.get(file_path)
        if lines is None:
            lines = _split_lines((await file_ops.read_file(file_path)).decode("utf-8", errors="replace"))
            cache[file_path] = lines
        if context and offset:
            output.append("--")
        rel = (
            os.path.relpath(file_path, root).replace(os.sep, "/")
            if is_dir
            else os.path.relpath(file_path, cwd).replace(os.sep, "/")
        )
        for current in range(max(1, line_no - context), min(len(lines), line_no + context) + 1):
            text, cut = truncate_line(lines[current - 1], GREP_MAX_LINE_LENGTH)
            line_cut = line_cut or cut
            marker = ":" if current == line_no else "-"
            output.append(f"{rel}{marker}{current}{marker} {text}")
    return output, line_cut


def _compile_glob(glob: str) -> pathspec.GitIgnoreSpec:
    # Bare patterns (no slash) match basename anywhere in the tree.
    pattern = glob if "/" in glob else f"**/{glob}"
    return pathspec.GitIgnoreSpec.from_lines([pattern.lstrip("/")])


def _split_lines(text: str) -> list[str]:
    return text.replace("\r\n", "\n").replace("\r", "\n").split("\n")


def _coerce_file_ops(api: ExtensionAPI, candidate: Any) -> FileOperations:
    return candidate if candidate is not None else api.get_operations().file


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    api.register_tool(_GrepTool(api.cwd, _coerce_file_ops(api, config.get("file_ops"))))
