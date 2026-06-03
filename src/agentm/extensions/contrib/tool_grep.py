"""Tool atom: content search via ``grep -rn`` or ``rg`` (ripgrep).

Ported from Claude Code's ``GrepTool``. Wraps the system grep (or ripgrep
when available) behind a ``tool:grep`` registration so the agent can search
file contents by regex pattern.

Backend selection: ``rg`` is preferred when found on ``$PATH`` because it
respects ``.gitignore`` by default and is significantly faster on large
trees. Falls back to POSIX ``grep -rn`` transparently.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Any, Final

from agentm.core.abi import FunctionTool, TextContent, ToolResult
from agentm.core.abi.extension import ExtensionAPI
from agentm.extensions import ExtensionManifest


MANIFEST = ExtensionManifest(
    name="tool_grep",
    description="Register a content-search tool backed by grep/ripgrep.",
    registers=("tool:grep",),
    config_schema={
        "type": "object",
        "properties": {
            "default_limit": {
                "type": "integer",
                "description": "Max result lines when caller omits limit.",
            },
        },
        "additionalProperties": True,
    },
    requires=(),
)

_PARAMETERS: Final = {
    "type": "object",
    "properties": {
        "pattern": {
            "type": "string",
            "description": "Regex pattern to search for in file contents.",
        },
        "path": {
            "type": "string",
            "description": (
                "Directory (or file) to search. Defaults to the session "
                "working directory."
            ),
        },
        "glob": {
            "type": "string",
            "description": (
                'File-name filter pattern (e.g. "*.py", "*.{ts,tsx}"). '
                "Passed as --include (grep) or --glob (rg)."
            ),
        },
        "output_mode": {
            "type": "string",
            "enum": ["content", "files_with_matches", "count"],
            "description": (
                '"content" shows matching lines with line numbers, '
                '"files_with_matches" lists file paths only, '
                '"count" shows per-file match counts. Default: "content".'
            ),
        },
        "case_insensitive": {
            "type": "boolean",
            "description": "Case-insensitive matching. Default: false.",
        },
        "limit": {
            "type": "integer",
            "description": "Max result lines returned. Default: 250.",
        },
        "context_lines": {
            "type": "integer",
            "description": "Lines of context around each match (-C flag).",
        },
    },
    "required": ["pattern"],
    "additionalProperties": False,
}

_DEFAULT_LIMIT: Final[int] = 250

_EXCLUDED_DIRS: Final[tuple[str, ...]] = (
    ".git",
    ".svn",
    ".hg",
    "node_modules",
    "__pycache__",
)


# ---------------------------------------------------------------------------
# Command builders
# ---------------------------------------------------------------------------

def _has_ripgrep() -> bool:
    return shutil.which("rg") is not None


def build_rg_command(
    pattern: str,
    path: str,
    *,
    glob: str | None = None,
    output_mode: str = "content",
    case_insensitive: bool = False,
    context_lines: int | None = None,
) -> list[str]:
    """Build an ``rg`` argument list from the tool parameters."""
    cmd: list[str] = ["rg", "--hidden"]

    for d in _EXCLUDED_DIRS:
        cmd.extend(["--glob", f"!{d}"])

    # Max column width prevents minified / base64 blobs from flooding output.
    cmd.extend(["--max-columns", "500"])

    if case_insensitive:
        cmd.append("-i")

    if output_mode == "files_with_matches":
        cmd.append("-l")
    elif output_mode == "count":
        cmd.append("-c")
    else:
        # content mode -- show line numbers
        cmd.append("-n")

    if context_lines is not None and output_mode == "content":
        cmd.extend(["-C", str(context_lines)])

    if glob:
        cmd.extend(["--glob", glob])

    # Protect patterns starting with '-' from being parsed as flags.
    if pattern.startswith("-"):
        cmd.extend(["-e", pattern])
    else:
        cmd.append(pattern)

    cmd.append(path)
    return cmd


def build_grep_command(
    pattern: str,
    path: str,
    *,
    glob: str | None = None,
    output_mode: str = "content",
    case_insensitive: bool = False,
    context_lines: int | None = None,
) -> list[str]:
    """Build a POSIX ``grep`` argument list from the tool parameters."""
    cmd: list[str] = ["grep", "-r"]

    for d in _EXCLUDED_DIRS:
        cmd.extend(["--exclude-dir", d])

    if case_insensitive:
        cmd.append("-i")

    if output_mode == "files_with_matches":
        cmd.append("-l")
    elif output_mode == "count":
        cmd.append("-c")
    else:
        cmd.append("-n")

    if context_lines is not None and output_mode == "content":
        cmd.extend(["-C", str(context_lines)])

    if glob:
        cmd.extend(["--include", glob])

    # Protect patterns starting with '-'.
    if pattern.startswith("-"):
        cmd.extend(["-e", pattern])
    else:
        cmd.append(pattern)

    cmd.append(path)
    return cmd


def build_command(
    pattern: str,
    path: str,
    *,
    glob: str | None = None,
    output_mode: str = "content",
    case_insensitive: bool = False,
    context_lines: int | None = None,
    use_ripgrep: bool | None = None,
) -> list[str]:
    """Select ``rg`` or ``grep`` and return the full argument list."""
    if use_ripgrep is None:
        use_ripgrep = _has_ripgrep()

    builder = build_rg_command if use_ripgrep else build_grep_command
    return builder(
        pattern,
        path,
        glob=glob,
        output_mode=output_mode,
        case_insensitive=case_insensitive,
        context_lines=context_lines,
    )


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def relativize_paths(lines: list[str], base: str) -> list[str]:
    """Convert absolute paths at the start of each line to relative."""
    prefix = base.rstrip(os.sep) + os.sep
    out: list[str] = []
    for line in lines:
        if line.startswith(prefix):
            line = line[len(prefix):]
        out.append(line)
    return out


def parse_output(
    raw: str,
    *,
    base_path: str,
    output_mode: str,
    limit: int,
) -> str:
    """Post-process raw grep/rg stdout into the final tool response."""
    if not raw.strip():
        return "No matches found."

    lines = raw.splitlines()
    lines = relativize_paths(lines, base_path)

    # For count mode, drop zero-count lines that grep emits.
    if output_mode == "count":
        lines = [ln for ln in lines if not ln.endswith(":0")]

    truncated = len(lines) > limit
    lines = lines[:limit]

    result = "\n".join(lines)
    if truncated:
        result += f"\n\n[Results truncated at {limit} lines]"
    return result


# ---------------------------------------------------------------------------
# Extension install
# ---------------------------------------------------------------------------

def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    default_limit = int(config.get("default_limit", _DEFAULT_LIMIT))
    cwd = api.cwd

    async def _execute(args: dict[str, Any]) -> ToolResult:
        pattern: str = args["pattern"]
        path: str = args.get("path", cwd)
        glob_filter: str | None = args.get("glob")
        output_mode: str = args.get("output_mode", "content")
        case_insensitive: bool = bool(args.get("case_insensitive", False))
        limit: int = int(args.get("limit", default_limit))
        context_lines: int | None = args.get("context_lines")

        # Resolve path relative to cwd when not absolute.
        if not os.path.isabs(path):
            path = os.path.normpath(os.path.join(cwd, path))

        if not os.path.exists(path):
            return _error(f"Path does not exist: {path!r}")

        cmd = build_command(
            pattern,
            path,
            glob=glob_filter,
            output_mode=output_mode,
            case_insensitive=case_insensitive,
            context_lines=context_lines,
        )

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            return _error("Search timed out after 30 seconds.")
        except FileNotFoundError:
            return _error(
                "Neither 'rg' nor 'grep' found on PATH. "
                "Install ripgrep or ensure grep is available."
            )

        # grep/rg exit 1 = no matches (not an error).
        if proc.returncode not in (0, 1):
            stderr = proc.stderr.strip()
            return _error(f"Search failed (exit {proc.returncode}): {stderr}")

        text = parse_output(
            proc.stdout,
            base_path=path,
            output_mode=output_mode,
            limit=limit,
        )
        return _ok(text)

    api.register_tool(
        FunctionTool(
            name="grep",
            description=(
                "Search file contents with a regex pattern. Uses ripgrep "
                "(rg) when available, falls back to grep."
            ),
            parameters=_PARAMETERS,
            fn=_execute,
        )
    )


def _ok(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)])


def _error(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)], is_error=True)
