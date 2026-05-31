"""Contrib tool atom: file-finding glob.

Ported from Claude Code's GlobTool — wraps Python's ``glob.glob`` with
recursive support, result limiting, and directory filtering.
"""

from __future__ import annotations

import glob as _glob_mod
import os
from typing import Any, Final

from agentm.core.abi import FunctionTool, TextContent, ToolResult
from agentm.extensions import ExtensionManifest
from agentm.core.abi.extension import ExtensionAPI


MANIFEST = ExtensionManifest(
    name="tool_glob",
    description="Register the glob tool for file-pattern matching.",
    registers=("tool:glob",),
    config_schema={
        "type": "object",
        "properties": {},
        "additionalProperties": True,
    },
    requires=(),
)

_PARAMETERS: Final = {
    "type": "object",
    "properties": {
        "pattern": {
            "type": "string",
            "description": "Glob pattern to match files against (e.g. '*.py', '**/*.yaml').",
        },
        "path": {
            "type": "string",
            "description": (
                "Directory to search in. Defaults to the session working "
                "directory if omitted."
            ),
        },
        "limit": {
            "type": "integer",
            "default": 100,
            "description": "Maximum number of results to return. Default 100.",
        },
    },
    "required": ["pattern"],
    "additionalProperties": False,
}

# Directories that are always pruned from results.
_SKIP_DIRS: Final[frozenset[str]] = frozenset({
    ".git",
    "node_modules",
    "__pycache__",
})


def _should_skip(path: str) -> bool:
    """Return True if any path component is in the skip set."""
    parts = path.replace(os.sep, "/").split("/")
    return bool(_SKIP_DIRS.intersection(parts))


def _glob_files(
    pattern: str,
    root: str,
    limit: int,
) -> tuple[list[str], bool]:
    """Run glob and return (relative_paths, truncated).

    Paths are relative to *root*, sorted alphabetically. Only regular
    files are included (directories are filtered out).
    """
    search_pattern = os.path.join(root, pattern)
    raw = _glob_mod.glob(search_pattern, recursive=True)

    results: list[str] = []
    for entry in raw:
        if not os.path.isfile(entry):
            continue
        rel = os.path.relpath(entry, root)
        if _should_skip(rel):
            continue
        results.append(rel)

    results.sort()

    truncated = len(results) > limit
    if truncated:
        results = results[:limit]

    return results, truncated


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    del config
    cwd = api.cwd

    async def _execute(args: dict[str, Any]) -> ToolResult:
        pattern = str(args["pattern"])
        search_root = str(args.get("path") or cwd)
        limit = int(args.get("limit", 100))

        if not os.path.isdir(search_root):
            return _error(
                f"Directory does not exist: {search_root!r}. "
                f"Current working directory is {cwd!r}."
            )

        try:
            filenames, truncated = _glob_files(pattern, search_root, limit)
        except Exception as exc:
            return _error(f"Glob failed: {exc}")

        if not filenames:
            return _ok("No files found")

        lines = filenames.copy()
        if truncated:
            lines.append(
                "(Results are truncated. Consider using a more specific "
                "path or pattern.)"
            )
        return _ok("\n".join(lines))

    api.register_tool(
        FunctionTool(
            name="glob",
            description=(
                "Find files by glob pattern. Returns matching filenames "
                "relative to the search directory."
            ),
            parameters=_PARAMETERS,
            fn=_execute,
            metadata={"file_op": "glob"},
        )
    )


def _ok(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)])


def _error(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)], is_error=True)
