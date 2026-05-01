"""Tool atom for the ``extensions.builtin.tool_find`` search-tools row."""

from __future__ import annotations

import asyncio
import fnmatch
import os
from pathlib import PurePosixPath
import shutil
from typing import Any, Protocol

import pathspec

from agentm.core.kernel import TextContent, Tool, ToolResult
from agentm.core.path_utils import resolve_to_cwd
from agentm.core.text_truncate import DEFAULT_MAX_BYTES, format_size, truncate_head
from agentm.extensions import ExtensionManifest
from agentm.harness.extension import ExtensionAPI

MANIFEST = ExtensionManifest(
    name="tool_find",
    description="Register the find tool backed by fd or stdlib fallback.",
    registers=("tool:find",),
    config_schema={
        "type": "object",
        "properties": {"ops": {"type": "object"}},
        "additionalProperties": True,
    },
)

_PARAMETERS = {
    "type": "object",
    "properties": {
        "pattern": {"type": "string"},
        "path": {"type": "string"},
        "limit": {"type": "integer", "default": 1000},
    },
    "required": ["pattern"],
    "additionalProperties": False,
}


class FindOperations(Protocol):
    async def exists(self, path: str) -> bool: ...

    async def glob(
        self,
        pattern: str,
        cwd: str,
        *,
        ignore: list[str],
        limit: int,
    ) -> list[str]: ...


class _LocalFindOperations:
    async def exists(self, path: str) -> bool:
        return await asyncio.to_thread(os.path.exists, path)

    async def glob(
        self,
        pattern: str,
        cwd: str,
        *,
        ignore: list[str],
        limit: int,
    ) -> list[str]:
        del pattern, ignore, limit
        raise NotImplementedError


class _FindTool(Tool):
    name = "find"
    description = "Find files and directories by glob pattern."
    parameters = _PARAMETERS

    def __init__(self, cwd: str, ops: FindOperations | None) -> None:
        self._cwd = cwd
        self._ops = ops or _LocalFindOperations()
        self._has_custom_ops = ops is not None

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
        if not await self._ops.exists(root):
            raise Exception(f"Path not found: {root}")

        if self._has_custom_ops:
            found = await self._ops.glob(
                pattern,
                root,
                ignore=[".git/", "node_modules/"],
                limit=limit,
            )
            results = sorted(_normalize_custom_paths(found, root))
            limit_hit = len(results) >= limit
        else:
            results, limit_hit = await _find_local(pattern, root, limit, signal)

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


async def _find_local(
    pattern: str,
    root: str,
    limit: int,
    signal: asyncio.Event | None,
) -> tuple[list[str], bool]:
    fd_path = shutil.which("fd")
    if fd_path:
        return await _find_with_fd(fd_path, pattern, root, limit, signal)
    return await asyncio.to_thread(_find_fallback, pattern, root, limit, signal is not None and signal.is_set())


async def _find_with_fd(
    fd_path: str,
    pattern: str,
    root: str,
    limit: int,
    signal: asyncio.Event | None,
) -> tuple[list[str], bool]:
    args = [
        fd_path,
        "--glob",
        "--color=never",
        "--hidden",
        "--no-require-git",
        "--max-results",
        str(limit),
    ]
    effective = pattern
    if "/" in pattern:
        args.append("--full-path")
        if not pattern.startswith("/") and not pattern.startswith("**/") and pattern != "**":
            effective = f"**/{pattern}"
    args.extend([effective, root])
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    wait_task = asyncio.create_task(proc.communicate())
    if signal is not None:
        signal_task = asyncio.create_task(signal.wait())
        done, _ = await asyncio.wait({wait_task, signal_task}, return_when=asyncio.FIRST_COMPLETED)
        if signal_task in done:
            proc.terminate()
            await wait_task
            raise Exception("Operation aborted")
        signal_task.cancel()
        await asyncio.gather(signal_task, return_exceptions=True)
    stdout, stderr = await wait_task
    if proc.returncode not in (0, 1):
        message = stderr.decode("utf-8", errors="replace").strip() or "fd search failed"
        raise Exception(message)
    results = sorted(
        line.strip().replace(os.sep, "/")
        for line in stdout.decode("utf-8", errors="replace").splitlines()
        if line.strip()
    )
    return results, len(results) >= limit


def _find_fallback(
    pattern: str,
    root: str,
    limit: int,
    aborted: bool,
) -> tuple[list[str], bool]:
    if aborted:
        raise Exception("Operation aborted")
    results: list[str] = []
    ignore = _ignore_spec(root, [".git/", "node_modules/"])
    limit_hit = False
    for dirpath, dirnames, filenames in os.walk(root, topdown=True, followlinks=True):
        rel_dir = os.path.relpath(dirpath, root)
        rel_dir_posix = "" if rel_dir == "." else rel_dir.replace(os.sep, "/")
        kept_dirs: list[str] = []
        for dirname in dirnames:
            rel_path = f"{rel_dir_posix}/{dirname}".strip("/")
            if ignore.match_file(rel_path) or ignore.match_file(rel_path + "/"):
                continue
            kept_dirs.append(dirname)
            if _matches(pattern, rel_path, dirname):
                results.append(rel_path + "/")
                if len(results) >= limit:
                    limit_hit = True
                    dirnames[:] = kept_dirs
                    return sorted(results), limit_hit
        dirnames[:] = kept_dirs
        for filename in filenames:
            rel_path = f"{rel_dir_posix}/{filename}".strip("/")
            if ignore.match_file(rel_path):
                continue
            if _matches(pattern, rel_path, filename):
                results.append(rel_path)
                if len(results) >= limit:
                    limit_hit = True
                    return sorted(results), limit_hit
    return sorted(results), limit_hit


def _matches(pattern: str, rel_path: str, name: str) -> bool:
    if "/" not in pattern:
        return fnmatch.fnmatch(name, pattern)
    adjusted = pattern
    if not pattern.startswith("/") and not pattern.startswith("**/") and pattern != "**":
        adjusted = f"**/{pattern}"
    return PurePosixPath(rel_path).match(adjusted.lstrip("/"))


def _ignore_spec(root: str, extra: list[str]) -> pathspec.PathSpec:
    patterns = list(extra)
    for dirpath, _dirnames, filenames in os.walk(root):
        if ".gitignore" not in filenames:
            continue
        prefix = os.path.relpath(dirpath, root).replace(os.sep, "/")
        if prefix == ".":
            prefix = ""
        with open(os.path.join(dirpath, ".gitignore"), encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if prefix:
                    line = f"{prefix}/{line.lstrip('/')}" if not line.startswith("/") else f"{prefix}{line}"
                patterns.append(line)
    return pathspec.PathSpec.from_lines("gitignore", patterns)


def _normalize_custom_paths(paths: list[str], root: str) -> list[str]:
    normalized: list[str] = []
    for value in paths:
        if os.path.isabs(value):
            rel = os.path.relpath(value, root)
        else:
            rel = value
        normalized.append(rel.replace(os.sep, "/"))
    return normalized


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    api.register_tool(_FindTool(api.cwd, config.get("ops")))
