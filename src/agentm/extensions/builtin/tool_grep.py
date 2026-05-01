"""Tool atom for the ``extensions.builtin.tool_grep`` search-tools row."""

from __future__ import annotations

import asyncio
import fnmatch
import json
import os
from pathlib import PurePosixPath
import re
import shutil
from typing import Any, Protocol

import pathspec

from agentm.core.kernel import TextContent, Tool, ToolResult
from agentm.core.operations import LocalFileOperations
from agentm.core.path_utils import load_gitignore_patterns, resolve_to_cwd
from agentm.core.text_truncate import DEFAULT_MAX_BYTES, GREP_MAX_LINE_LENGTH, format_size, truncate_head, truncate_line
from agentm.extensions import ExtensionManifest
from agentm.harness.extension import ExtensionAPI

MANIFEST = ExtensionManifest(
    name="tool_grep",
    description="Register the grep tool backed by rg or stdlib fallback.",
    registers=("tool:grep",),
    config_schema={"type": "object", "properties": {"ops": {"type": "object"}}, "additionalProperties": True},
)

_PARAMETERS = {
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


class GrepOperations(Protocol):
    async def is_directory(self, path: str) -> bool: ...

    async def read_file(self, path: str) -> str: ...


class _LocalGrepOperations:
    def __init__(self) -> None:
        self._file_ops = LocalFileOperations()

    async def is_directory(self, path: str) -> bool:
        return await asyncio.to_thread(os.path.isdir, path)

    async def read_file(self, path: str) -> str:
        data = await self._file_ops.read_file(path)
        return data.decode("utf-8", errors="replace")


class _GrepTool(Tool):
    name = "grep"
    description = "Search files for regex or literal matches, honoring .gitignore."
    parameters = _PARAMETERS

    def __init__(self, cwd: str, ops: GrepOperations | None) -> None:
        self._cwd = cwd
        self._ops = ops or _LocalGrepOperations()
        self._has_custom_ops = ops is not None

    async def execute(self, args: dict[str, Any], *, signal: asyncio.Event | None = None, on_update: Any = None) -> ToolResult:
        del on_update
        if signal is not None and signal.is_set():
            raise Exception("Operation aborted")
        pattern = str(args["pattern"])
        search_path = resolve_to_cwd(str(args.get("path", ".")), self._cwd)
        glob = str(args["glob"]) if args.get("glob") is not None else None
        context = max(0, int(args.get("context", 0)))
        limit = max(1, int(args.get("limit", 100)))
        is_dir, prefetched_text = await _classify_path(self._ops, search_path)
        rg_path = shutil.which("rg")
        if rg_path and not self._has_custom_ops:
            hits = await _grep_with_rg(rg_path, pattern, search_path, glob, bool(args.get("ignore_case", False)), bool(args.get("literal", False)), limit, signal)
            fallback = False
        else:
            hits = await _grep_fallback(
                self._ops,
                pattern,
                search_path,
                glob,
                bool(args.get("ignore_case", False)),
                bool(args.get("literal", False)),
                limit,
                is_dir,
                prefetched_text,
                signal,
            )
            fallback = True
        lines, line_cut = await _render(
            self._ops,
            search_path,
            is_dir,
            hits[:limit],
            context,
            prefetched_text,
            self._cwd,
        )
        if not lines:
            text = "No matches found"
            if fallback:
                text += "\n[rg unavailable; used slower stdlib fallback. Install ripgrep via your package manager for faster searches.]"
            return ToolResult(content=[TextContent(type="text", text=text)])
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


async def _grep_with_rg(
    rg_path: str,
    pattern: str,
    search_path: str,
    glob: str | None,
    ignore_case: bool,
    literal: bool,
    limit: int,
    signal: asyncio.Event | None,
) -> list[tuple[str, int]]:
    args = [rg_path, "--json", "--line-number", "--color=never", "--hidden"]
    if ignore_case:
        args.append("--ignore-case")
    if literal:
        args.append("--fixed-strings")
    if glob:
        args.extend(["--glob", glob])
    args.extend([pattern, search_path])
    proc = await asyncio.create_subprocess_exec(*args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    assert proc.stdout is not None and proc.stderr is not None
    hits: list[tuple[str, int]] = []
    while raw := await proc.stdout.readline():
        if signal is not None and signal.is_set():
            proc.terminate()
            await proc.wait()
            raise Exception("Operation aborted")
        event = json.loads(raw.decode("utf-8", errors="replace"))
        if event.get("type") != "match":
            continue
        data = event["data"]
        hits.append((data["path"]["text"], int(data["line_number"])))
        if len(hits) >= limit:
            proc.terminate()
            break
    stderr = await proc.stderr.read()
    await proc.wait()
    if proc.returncode not in (0, 1, -15):
        raise Exception(stderr.decode("utf-8", errors="replace").strip() or "ripgrep search failed")
    return hits


async def _grep_fallback(
    ops: GrepOperations,
    pattern: str,
    search_path: str,
    glob: str | None,
    ignore_case: bool,
    literal: bool,
    limit: int,
    is_dir: bool,
    prefetched_text: str | None,
    signal: asyncio.Event | None,
) -> list[tuple[str, int]]:
    compiled = re.compile(re.escape(pattern) if literal else pattern, re.IGNORECASE if ignore_case else 0)
    files = [search_path] if not is_dir else await _list_search_files(ops, search_path, glob)
    hits: list[tuple[str, int]] = []
    for file_path in files:
        if signal is not None and signal.is_set():
            raise Exception("Operation aborted")
        try:
            text = prefetched_text if file_path == search_path and prefetched_text is not None else await ops.read_file(file_path)
        except Exception:
            continue
        for line_no, line in enumerate(_split_lines(text), start=1):
            if compiled.search(line):
                hits.append((file_path, line_no))
                if len(hits) >= limit:
                    return hits
    return hits


async def _classify_path(ops: GrepOperations, search_path: str) -> tuple[bool, str | None]:
    try:
        if await ops.is_directory(search_path):
            return True, None
        return False, await ops.read_file(search_path)
    except Exception as exc:
        raise Exception(f"Path not found: {search_path}") from exc


async def _list_search_files(
    ops: GrepOperations,
    root: str,
    glob: str | None,
) -> list[str]:
    walk_files = getattr(ops, "walk_files", None)
    if callable(walk_files):
        return sorted(await walk_files(root, glob=glob))
    ignore = pathspec.PathSpec.from_lines("gitignore", load_gitignore_patterns(root, extra=[".git/"]))
    return await asyncio.to_thread(_walk_files, root, glob, ignore)


async def _render(
    ops: GrepOperations,
    root: str,
    is_dir: bool,
    hits: list[tuple[str, int]],
    context: int,
    prefetched_text: str | None,
    cwd: str,
) -> tuple[list[str], bool]:
    cache: dict[str, list[str]] = (
        {root: _split_lines(prefetched_text)} if prefetched_text is not None and not is_dir else {}
    )
    output: list[str] = []
    line_cut = False
    for offset, (file_path, line_no) in enumerate(sorted(hits, key=lambda item: (item[0], item[1]))):
        lines = cache.get(file_path)
        if lines is None:
            lines = _split_lines(await ops.read_file(file_path))
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


def _walk_files(root: str, glob: str | None, ignore: pathspec.PathSpec) -> list[str]:
    files: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root, topdown=True, followlinks=True):
        rel_dir = os.path.relpath(dirpath, root).replace(os.sep, "/")
        rel_dir = "" if rel_dir == "." else rel_dir
        dirnames[:] = [name for name in dirnames if not _ignored(ignore, rel_dir, name)]
        for filename in filenames:
            rel_path = f"{rel_dir}/{filename}".strip("/")
            if ignore.match_file(rel_path) or (glob and not _glob_matches(glob, rel_path, filename)):
                continue
            files.append(os.path.join(dirpath, filename))
    return sorted(files)


def _ignored(spec: pathspec.PathSpec, rel_dir: str, name: str) -> bool:
    rel_path = f"{rel_dir}/{name}".strip("/")
    return spec.match_file(rel_path) or spec.match_file(rel_path + "/")


def _glob_matches(glob: str, rel_path: str, name: str) -> bool:
    if "/" not in glob:
        return fnmatch.fnmatch(name, glob)
    adjusted = glob if glob.startswith("**/") or glob.startswith("/") else f"**/{glob}"
    return PurePosixPath(rel_path).match(adjusted.lstrip("/"))


def _split_lines(text: str) -> list[str]:
    return text.replace("\r\n", "\n").replace("\r", "\n").split("\n")


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    api.register_tool(_GrepTool(api.cwd, config.get("ops")))
