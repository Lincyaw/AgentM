"""Grouped file-I/O tool atom: ``read``, ``write``, ``edit``, ``glob``, ``grep``.

Merges the former single-tool atoms into one §11-compliant module. The
LLM-facing tool names are unchanged.

Read — aligned with Claude Code's FileReadTool behavior: no hardcoded
line cap, max-file-size gate (default 256 KB), partial-view tracking for
downstream edit/write safety.

Write — enforces Claude-Code-style safety gates: read-before-write for
existing files, file-modified-since-read detection, post-write read_state
update.

Edit — enforces read-before-edit, supports string replacement and
line-range replacement modes, file-modified-since-read detection,
post-edit read_state update.

Glob — file-pattern matching via ``find`` through BashOperations, with
directory exclusion for .git/node_modules/__pycache__.

Grep — content search via grep/ripgrep through BashOperations, with
output parsing and path relativization.
"""

from __future__ import annotations

import fnmatch
import os
import shlex
from pathlib import Path, PurePath
from typing import Any, Final

from loguru import logger
from pydantic import BaseModel, ConfigDict

from agentm.core.abi import (
    BashOperations,
    ExtensionAPI,
    FileOperations,
    FunctionTool,
    TOOL_RESULT_FORMAT_METADATA_KEY,
    TextContent,
    ToolResult,
)
from agentm.core.lib import (
    content_hash_for,
    file_modified_since_read,
    get_read_state,
    record_read,
)
from agentm.extensions import ExtensionManifest

# ---------------------------------------------------------------------------
# MANIFEST
# ---------------------------------------------------------------------------

_ALL_TOOLS: Final[frozenset[str]] = frozenset({"read", "write", "edit", "glob", "grep"})


class FileToolsConfig(BaseModel):
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    file_ops: Any = None
    tools: list[str] | None = None
    allow_globs: list[str] | None = None
    deny_globs: list[str] | None = None
    max_size_bytes: int = 262_144
    require_read: bool = True
    default_limit: int = 250

MANIFEST = ExtensionManifest(
    name="file_tools",
    description="Register the read, write, edit, glob, and grep tools for file I/O.",
    registers=("tool:read", "tool:write", "tool:edit", "tool:glob", "tool:grep"),
    config_schema=FileToolsConfig,
    requires=(),
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _ok(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)])

def _error(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)], is_error=True)

# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------

# 256 KB — matches Claude Code's MAX_OUTPUT_SIZE (0.25 * 1024 * 1024).
_DEFAULT_MAX_SIZE_BYTES: Final[int] = 262_144

_BINARY_EXTENSIONS: Final[frozenset[str]] = frozenset({
    # Video
    ".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv",
    # Audio
    ".mp3", ".wav", ".flac", ".aac", ".ogg", ".wma", ".m4a",
    # Image
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp", ".ico", ".svg",
    # Archive
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
    # Binary / native
    ".bin", ".exe", ".dll", ".so", ".dylib", ".o", ".a", ".pyc", ".class",
    # Documents
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    # Database
    ".sqlite", ".db",
})

def _check_binary(path: str) -> str | None:
    """Return an error string if *path* looks like a binary file, else None."""
    ext = PurePath(path).suffix.lower()
    if ext in _BINARY_EXTENSIONS:
        return (
            f"Cannot read binary file {path!r} ({ext} format). "
            "Use bash to inspect metadata (e.g. `file <path>`, `ls -la <path>`) "
            "or process it with appropriate tools."
        )
    return None

def _coerce_file_ops(api: ExtensionAPI, candidate: Any) -> FileOperations:
    return candidate if candidate is not None else api.get_operations().file

def _coerce_globs(value: Any, cwd: str) -> tuple[str, ...]:
    """Anchor relative glob patterns against the session ``cwd``."""
    if not isinstance(value, list):
        return ()
    out: list[str] = []
    for raw in value:
        if not isinstance(raw, str) or not raw:
            continue
        if os.path.isabs(raw):
            out.append(raw)
        else:
            out.append(os.path.normpath(os.path.join(cwd, raw)))
    return tuple(out)

def _resolved(path: str) -> str:
    """Resolve to absolute, symlink-collapsed path for matching."""
    try:
        return str(Path(path).expanduser().resolve(strict=False))
    except (OSError, RuntimeError):
        return os.path.abspath(os.path.expanduser(path))

def _matches_any(path: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)

def _check_path_allowed(
    path: str,
    allow: tuple[str, ...],
    deny: tuple[str, ...],
) -> str | None:
    resolved = _resolved(path)
    if allow and not _matches_any(resolved, allow):
        return (
            f"Access denied: {path!r} is outside the configured allow_globs "
            f"({list(allow)}). Adjust the scenario manifest if this access "
            "should be permitted."
        )
    if deny and _matches_any(resolved, deny):
        return (
            f"Access denied: {path!r} matches a configured deny_glob "
            f"({list(deny)})."
        )
    return None

_READ_PARAMETERS: Final = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "Absolute path to the file to read.",
        },
        "offset": {
            "type": "integer",
            "description": (
                "1-based line number to start reading from. "
                "Only provide if the file is too large to read at once."
            ),
        },
        "limit": {
            "type": "integer",
            "description": (
                "Number of lines to read. "
                "Only provide if the file is too large to read at once."
            ),
        },
    },
    "required": ["path"],
    "additionalProperties": False,
}

# ---------------------------------------------------------------------------
# Edit helpers
# ---------------------------------------------------------------------------

_EDIT_PARAMETERS: Final = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "File path to edit."},
        "old_string": {
            "type": "string",
            "description": "Exact text to find and replace. Mutually exclusive with start_line/end_line.",
        },
        "new_string": {
            "type": "string",
            "description": "Replacement text.",
        },
        "start_line": {
            "type": "integer",
            "description": "1-based start line for line-range replacement (inclusive).",
        },
        "end_line": {
            "type": "integer",
            "description": "1-based end line for line-range replacement (inclusive).",
        },
        "replace_all": {"type": "boolean", "default": False},
        "rationale": {"type": "string", "default": "agent edit via file_tools"},
    },
    "required": ["path", "new_string"],
    "additionalProperties": False,
}

_CONTEXT_LINES = 4
_MAX_UNINTENDED_SHRINK_LINES = 5

def _check_shrinkage(original: str, updated: str, old_len: int, new_len: int) -> str | None:
    """Reject edits that delete far more content than the replacement explains."""
    expected_delta = new_len - old_len
    actual_delta = len(updated) - len(original)
    unintended = expected_delta - actual_delta
    if unintended <= 0:
        return None
    lost_lines = original.count("\n") - updated.count("\n")
    if lost_lines > _MAX_UNINTENDED_SHRINK_LINES:
        return (
            f"Edit rejected: this replacement would delete {lost_lines} lines "
            f"beyond the matched region. This usually means old_string matched "
            f"more content than intended. Re-read the file and use a more precise "
            f"old_string, or use start_line/end_line for the exact range."
        )
    return None

_QUOTE_MAP: Final[dict[str, str]] = {
    "‘": "'",  # left single curly
    "’": "'",  # right single curly
    "“": '"',  # left double curly
    "”": '"',  # right double curly
}

def _normalize_quotes(s: str) -> str:
    for curly, straight in _QUOTE_MAP.items():
        s = s.replace(curly, straight)
    return s

def _snippet_around(content: str, start_line: int, end_line: int) -> str:
    """Return a snippet of *content* showing +-CONTEXT_LINES around [start, end] with line numbers."""
    lines = content.splitlines()
    total = len(lines)
    snippet_start = max(0, start_line - 1 - _CONTEXT_LINES)
    snippet_end = min(total, end_line + _CONTEXT_LINES)
    numbered = [
        f"{snippet_start + i + 1}\t{line}"
        for i, line in enumerate(lines[snippet_start:snippet_end])
    ]
    return "\n".join(numbered)

async def _update_read_state_after_edit(
    normalized_path: str, file_ops: FileOperations
) -> None:
    """Refresh read_state for *normalized_path* after a successful edit."""
    old = get_read_state(normalized_path)
    total_lines = old.total_lines if old else 0
    is_partial = old.is_partial if old else False
    try:
        fs = await file_ops.stat(normalized_path)
        mtime_ns = fs.mtime_ns
    except OSError:
        mtime_ns = 0
    try:
        raw = await file_ops.read_file(normalized_path)
        chash = content_hash_for(raw)
        total_lines = raw.decode("utf-8", errors="replace").count("\n") + 1
    except OSError:
        chash = ""
    record_read(
        normalized_path,
        total_lines=total_lines,
        is_partial=is_partial,
        mtime_ns=mtime_ns,
        content_hash=chash,
    )

def _strip_line_whitespace(s: str) -> str:
    """Strip leading/trailing whitespace from each line, preserving newlines."""
    return "\n".join(line.strip() for line in s.split("\n"))

def _find_actual_string(file_content: str, search: str) -> str | None:
    """Find *search* in *file_content* with progressive fallbacks.

    Aligned with Claude Code's ``findActualString``:
    1. Exact match
    2. Quote-normalized match (curly -> straight)
    3. Whitespace-trimmed per-line match

    Always returns the ACTUAL string from the file, not the search string.
    """
    # 1. Exact match
    if search in file_content:
        return search

    # 2. Quote-normalized match
    norm_search = _normalize_quotes(search)
    norm_file = _normalize_quotes(file_content)
    idx = norm_file.find(norm_search)
    if idx != -1:
        return file_content[idx : idx + len(norm_search)]

    # 3. Whitespace-trimmed per-line match
    stripped_search = _strip_line_whitespace(search)
    stripped_file = _strip_line_whitespace(file_content)
    idx = stripped_file.find(stripped_search)
    if idx != -1:
        orig_lines = file_content.split("\n")
        prefix = stripped_file[:idx]
        start_line = prefix.count("\n")
        search_line_count = stripped_search.count("\n") + 1
        matched_lines = orig_lines[start_line : start_line + search_line_count]
        return "\n".join(matched_lines)

    return None

# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------

_WRITE_PARAMETERS: Final = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "File path to write."},
        "content": {
            "type": "string",
            "description": "The full content to write.",
        },
        "rationale": {
            "type": "string",
            "default": "agent write via file_tools",
        },
    },
    "required": ["path", "content"],
    "additionalProperties": False,
}

# ---------------------------------------------------------------------------
# Glob helpers
# ---------------------------------------------------------------------------

_GLOB_PARAMETERS: Final = {
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

_GLOB_SKIP_DIRS: Final[frozenset[str]] = frozenset({
    ".git",
    "node_modules",
    "__pycache__",
})

def _should_skip_glob(path: str) -> bool:
    """Return True if any path component is in the skip set."""
    parts = path.replace(os.sep, "/").split("/")
    return bool(_GLOB_SKIP_DIRS.intersection(parts))

# ---------------------------------------------------------------------------
# Grep helpers
# ---------------------------------------------------------------------------

_GREP_PARAMETERS: Final = {
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

_GREP_DEFAULT_LIMIT: Final[int] = 250

_GREP_EXCLUDED_DIRS: Final[tuple[str, ...]] = (
    ".git",
    ".svn",
    ".hg",
    "node_modules",
    "__pycache__",
)

def build_rg_command(
    pattern: str,
    path: str,
    *,
    glob_filter: str | None = None,
    output_mode: str = "content",
    case_insensitive: bool = False,
    context_lines: int | None = None,
) -> list[str]:
    """Build an ``rg`` argument list from the tool parameters."""
    cmd: list[str] = ["rg", "--hidden"]

    for d in _GREP_EXCLUDED_DIRS:
        cmd.extend(["--glob", f"!{d}"])

    cmd.extend(["--max-columns", "500"])

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

    if glob_filter:
        cmd.extend(["--glob", glob_filter])

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
    glob_filter: str | None = None,
    output_mode: str = "content",
    case_insensitive: bool = False,
    context_lines: int | None = None,
) -> list[str]:
    """Build a POSIX ``grep`` argument list from the tool parameters."""
    cmd: list[str] = ["grep", "-r"]

    for d in _GREP_EXCLUDED_DIRS:
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

    if glob_filter:
        cmd.extend(["--include", glob_filter])

    if pattern.startswith("-"):
        cmd.extend(["-e", pattern])
    else:
        cmd.append(pattern)

    cmd.append(path)
    return cmd

def build_grep_or_rg_command(
    pattern: str,
    path: str,
    *,
    glob_filter: str | None = None,
    output_mode: str = "content",
    case_insensitive: bool = False,
    context_lines: int | None = None,
    use_ripgrep: bool | None = None,
) -> list[str]:
    """Select ``rg`` or ``grep`` and return the full argument list.

    When *use_ripgrep* is None the caller must probe availability externally
    (via BashOperations) before calling.
    """
    builder = build_rg_command if use_ripgrep else build_grep_command
    return builder(
        pattern,
        path,
        glob_filter=glob_filter,
        output_mode=output_mode,
        case_insensitive=case_insensitive,
        context_lines=context_lines,
    )

def relativize_paths(lines: list[str], base: str) -> list[str]:
    """Convert absolute paths at the start of each line to relative."""
    prefix = base.rstrip(os.sep) + os.sep
    out: list[str] = []
    for line in lines:
        if line.startswith(prefix):
            line = line[len(prefix):]
        out.append(line)
    return out

def parse_grep_output(
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

    if output_mode == "count":
        lines = [ln for ln in lines if not ln.endswith(":0")]

    truncated = len(lines) > limit
    lines = lines[:limit]

    result = "\n".join(lines)
    if truncated:
        result += f"\n\n[Results truncated at {limit} lines]"
    return result

# ---------------------------------------------------------------------------
# install()
# ---------------------------------------------------------------------------

def install(api: ExtensionAPI, config: FileToolsConfig) -> None:
    # Lazy-resolve file_ops and writer: at install time, the Operations
    # bundle or ResourceWriter may not be registered yet (depends on atom
    # load order). Deferring to first tool invocation avoids an install-time
    # ordering dependency while keeping requires=() (any Operations
    # provider is acceptable, not just the local backend).
    _file_ops_cfg = config.file_ops
    _file_ops_cache: list[FileOperations] = []

    def _get_file_ops() -> FileOperations:
        if not _file_ops_cache:
            _file_ops_cache.append(_coerce_file_ops(api, _file_ops_cfg))
        return _file_ops_cache[0]

    _writer_cache: list[Any] = []

    def _get_writer() -> Any:
        if not _writer_cache:
            _writer_cache.append(api.get_resource_writer())
        return _writer_cache[0]

    _bash_ops_cache: list[BashOperations] = []

    def _get_bash_ops() -> BashOperations:
        if not _bash_ops_cache:
            _bash_ops_cache.append(api.get_operations().bash)
        return _bash_ops_cache[0]

    enabled_tools = frozenset(config.tools) if config.tools is not None else _ALL_TOOLS
    allow_globs = _coerce_globs(config.allow_globs, api.cwd)
    deny_globs = _coerce_globs(config.deny_globs, api.cwd)
    max_size_bytes: int = config.max_size_bytes
    require_read = config.require_read

    # --- read tool --------------------------------------------------------

    async def _read_execute(args: dict[str, Any]) -> ToolResult:
        path = str(args["path"])
        raw_offset = args.get("offset")
        raw_limit = args.get("limit")

        gate_error = _check_path_allowed(path, allow_globs, deny_globs)
        if gate_error is not None:
            return _error(gate_error)

        binary_error = _check_binary(path)
        if binary_error is not None:
            return _error(binary_error)

        try:
            data = await _get_file_ops().read_file(path)
        except Exception as exc:
            return _error(f"Failed to read {path!r}: {exc}")

        # --- max-size gate (checked on raw bytes, before decode) ---
        file_size = len(data)
        caller_wants_range = raw_offset is not None or raw_limit is not None
        if file_size > max_size_bytes and not caller_wants_range:
            return _error(
                f"File content ({file_size} bytes) exceeds maximum "
                f"allowed size ({max_size_bytes} bytes). "
                "Use offset and limit parameters to read specific "
                "portions of the file."
            )

        try:
            all_lines = data.decode("utf-8", errors="replace").splitlines()
            total = len(all_lines)

            # Offset: 1-based when provided, 0 means "from beginning".
            offset = max(0, int(raw_offset) - 1) if raw_offset is not None else 0
            limit = int(raw_limit) if raw_limit is not None else None

            if limit is not None and limit > 0:
                sliced = all_lines[offset : offset + limit]
            else:
                sliced = all_lines[offset:]

            is_partial = (
                offset > 0
                or (limit is not None and limit > 0 and offset + limit < total)
            )

            record_read(path, total_lines=total, is_partial=is_partial)

            numbered = [
                f"{offset + i + 1}\t{line}"
                for i, line in enumerate(sliced)
            ]

            if is_partial:
                end_line = offset + len(sliced)
                header = f"(showing lines {offset + 1}-{end_line} of {total})"
            else:
                header = f"({total} lines total)"

            return _ok(header + "\n" + "\n".join(numbered))
        except Exception as exc:
            return _error(f"Failed to read {path!r}: {exc}")

    if "read" in enabled_tools:
        api.register_tool(
            FunctionTool(
                name="read",
                description=(
                    "Read a UTF-8 text file from disk. "
                    "By default reads the entire file. "
                    f"Files larger than {max_size_bytes} bytes require "
                    "offset and limit parameters."
                ),
                parameters=_READ_PARAMETERS,
                fn=_read_execute,
                metadata={"file_op": "read"},
            )
        )

    # --- write tool -------------------------------------------------------

    async def _write_execute(args: dict[str, Any]) -> ToolResult:
        path = str(args["path"])
        content = str(args["content"])
        rationale = str(args.get("rationale", "agent write via file_tools"))

        normalized = os.path.normpath(path)

        # Determine if the file already exists on disk.
        writer = _get_writer()
        file_exists = False
        try:
            await writer.read(path)
            file_exists = True
        except Exception as exc:
            # Read failed → treat as not-yet-existing (covers not-found and
            # unreadable paths alike); the write below will surface hard errors.
            logger.debug("file_tools: existence probe read({!r}) failed: {}", path, exc)

        if file_exists and require_read:
            rs = get_read_state(normalized)

            # Gate 1: must have been read at all.
            if rs is None:
                return _error(
                    f"File {path!r} already exists. Read it first before "
                    "overwriting so you can see its current content. "
                    "Use the read tool, then write."
                )

            # Gate 2: must have been a full read (no offset/limit).
            if rs.is_partial:
                return _error(
                    f"You read {path!r} with offset/limit (partial view). "
                    "Read the full file before overwriting."
                )

            # Gate 3: mtime must not have changed since the read.
            recorded_mtime = getattr(rs, "mtime_ns", None)
            if recorded_mtime is not None:
                try:
                    fs = await _get_file_ops().stat(normalized)
                    current_mtime: int | None = fs.mtime_ns
                except OSError:
                    current_mtime = None
                if current_mtime is not None and current_mtime != recorded_mtime:
                    return _error(
                        "File has been modified since you read it. "
                        "Read it again before writing."
                    )

        try:
            result = await writer.write(
                path,
                content.encode("utf-8"),
                rationale=rationale,
            )
            if result.error is not None:
                return _error(result.error)

            # Post-write: update read_state so subsequent edit calls
            # see the file as freshly read (full content, not partial).
            total_lines = content.count("\n") + (1 if content else 0)
            record_kwargs: dict[str, Any] = {
                "total_lines": total_lines,
                "is_partial": False,
            }
            try:
                disk_stat = await _get_file_ops().stat(normalized)
                record_kwargs["mtime_ns"] = disk_stat.mtime_ns
            except OSError as exc:
                # mtime is an optimisation for read-state tracking; omit it if
                # the post-write stat fails rather than failing the write.
                logger.debug("file_tools: post-write stat({}) failed: {}", normalized, exc)
            record_read(normalized, **record_kwargs)

            action = "Updated" if file_exists else "Created"
            byte_count = len(content.encode("utf-8"))
            return _ok(f"{action} {path!r} ({byte_count} bytes)")
        except Exception as exc:
            return _error(f"Failed to write {path!r}: {exc}")

    if "write" in enabled_tools:
        api.register_tool(
            FunctionTool(
                name="write",
                description=(
                    "Write a UTF-8 text file. For existing files, you MUST read "
                    "the full file first. Prefer the edit tool for modifying "
                    "existing files — use write only for new files or complete "
                    "rewrites."
                ),
                parameters=_WRITE_PARAMETERS,
                fn=_write_execute,
                metadata={"file_op": "write"},
            )
        )

    # --- edit tool --------------------------------------------------------

    async def _string_replace(
        path: str,
        original: str,
        old_string: str,
        new_string: str,
        replace_all: bool,
        rationale: str,
    ) -> ToolResult:
        actual = _find_actual_string(original, old_string)
        if actual is None:
            return _error(f"String not found in {path!r}: {old_string!r}")
        occurrences = original.count(actual)
        if not replace_all and occurrences != 1:
            return _error(
                f"String is not unique in {path!r}: found {occurrences} matches"
            )
        updated = (
            original.replace(actual, new_string)
            if replace_all
            else original.replace(actual, new_string, 1)
        )
        shrinkage = _check_shrinkage(original, updated, len(actual), len(new_string))
        if shrinkage:
            return _error(shrinkage)
        result = await _get_writer().replace(
            path, original.encode("utf-8"), updated.encode("utf-8"), rationale=rationale,
        )
        if result.error is not None:
            return _error(result.error)
        before_lines = original[: original.index(actual)].count("\n")
        new_lines_count = new_string.count("\n") + 1
        snippet = _snippet_around(updated, before_lines + 1, before_lines + new_lines_count)
        return _ok(f"Updated {path!r}:\n{snippet}")

    async def _line_range_replace(
        path: str,
        original: str,
        start: int,
        end: int,
        new_string: str,
        rationale: str,
    ) -> ToolResult:
        lines = original.splitlines(keepends=True)
        total = len(lines)
        if start < 1 or end < start or start > total:
            return _error(
                f"Invalid line range [{start}, {end}] for {path!r} ({total} lines). "
                "Lines are 1-based."
            )
        end = min(end, total)
        before = lines[: start - 1]
        after = lines[end:]
        if new_string and not new_string.endswith("\n"):
            new_string += "\n"
        updated = "".join(before) + new_string + "".join(after)
        replaced_len = sum(len(ln) for ln in lines[start - 1 : end])
        shrinkage = _check_shrinkage(original, updated, replaced_len, len(new_string))
        if shrinkage:
            return _error(shrinkage)
        result = await _get_writer().replace(
            path, original.encode("utf-8"), updated.encode("utf-8"), rationale=rationale,
        )
        if result.error is not None:
            return _error(result.error)
        new_line_count = new_string.count("\n") + 1
        snippet = _snippet_around(updated, start, start + new_line_count - 1)
        return _ok(f"Replaced lines {start}-{end} in {path!r}:\n{snippet}")

    async def _edit_execute(args: dict[str, Any]) -> ToolResult:
        path = str(args["path"])
        new_string = str(args["new_string"])
        old_string = args.get("old_string")
        start_line = args.get("start_line")
        end_line = args.get("end_line")
        replace_all = bool(args.get("replace_all", False))
        rationale = str(args.get("rationale", "agent edit via file_tools"))

        normalized = os.path.normpath(path)
        state = get_read_state(normalized)
        if require_read and state is None:
            return _error(
                f"You must read {path!r} before editing it. "
                "Use the read tool first so you can see the exact content and line numbers."
            )

        # File-modified-since-read detection (aligned with Claude Code)
        if state is not None and file_modified_since_read(normalized):
            return _error(
                f"File has been modified since you last read it. "
                f"Read {path!r} again before editing."
            )

        has_old = old_string is not None and old_string != ""
        has_lines = start_line is not None and end_line is not None

        if has_old and has_lines:
            return _error("Provide either old_string OR start_line/end_line, not both.")
        if not has_old and not has_lines:
            return _error("Provide old_string or start_line + end_line.")

        try:
            original = (await _get_writer().read(path)).decode("utf-8", errors="replace")

            if start_line is not None and end_line is not None:
                result = await _line_range_replace(
                    path, original, int(start_line), int(end_line),
                    new_string, rationale,
                )
            else:
                result = await _string_replace(
                    path, original, str(old_string), new_string,
                    replace_all, rationale,
                )

            # Post-edit: update read_state so subsequent edits don't
            # false-positive on "modified since read".
            if not result.is_error:
                await _update_read_state_after_edit(normalized, _get_file_ops())

            return result
        except Exception as exc:
            logger.opt(exception=True).warning("edit tool failed for {}: {}", path, exc)
            return _error(f"Failed to edit {path!r}: {exc}")

    if "edit" in enabled_tools:
        api.register_tool(
            FunctionTool(
                name="edit",
                description=(
                    "Edit a UTF-8 text file. Two modes:\n"
                    "1. String replacement: provide old_string + new_string.\n"
                    "2. Line-range replacement: provide start_line + end_line + new_string "
                    "(1-based, inclusive). You MUST read the file first to see line numbers."
                ),
                parameters=_EDIT_PARAMETERS,
                fn=_edit_execute,
                metadata={"file_op": "edit", TOOL_RESULT_FORMAT_METADATA_KEY: "diff"},
            )
        )

    # --- glob tool -------------------------------------------------------

    cwd = api.cwd

    async def _glob_execute(args: dict[str, Any]) -> ToolResult:
        pattern = str(args["pattern"])
        search_root = str(args.get("path") or cwd)
        limit = int(args.get("limit", 100))

        bash_ops = _get_bash_ops()

        # Verify directory exists via bash (works in sandbox too).
        check = await bash_ops.exec(
            f"test -d {shlex.quote(search_root)} && echo yes || echo no",
            cwd=cwd,
            timeout=10,
        )
        if check.stdout.decode().strip() != "yes":
            return _error(
                f"Directory does not exist: {search_root!r}. "
                f"Current working directory is {cwd!r}."
            )

        # Build a find command with pruning for skip dirs.
        # -path '*/<dir>/*' -prune keeps those dirs out of results.
        prune_parts: list[str] = []
        for d in sorted(_GLOB_SKIP_DIRS):
            prune_parts.append(f"-path {shlex.quote('*/' + d)} -prune")
        prune_expr = " -o ".join(prune_parts)

        # Extra limit+1 to detect truncation.
        find_cmd = (
            f"find {shlex.quote(search_root)} "
            f"\\( {prune_expr} \\) -o "
            f"-type f -name {shlex.quote(pattern)} -print "
            f"| head -n {limit + 1} | sort"
        )

        try:
            result = await bash_ops.exec(find_cmd, cwd=cwd, timeout=30)
        except Exception as exc:
            return _error(f"Glob failed: {exc}")

        raw = result.stdout.decode().strip()
        if not raw:
            return _ok("No files found")

        all_paths = raw.split("\n")

        # Filter skip dirs (belt-and-suspenders; find prunes, but sort
        # can re-interleave pruned entries on some edge cases).
        filtered: list[str] = []
        for p in all_paths:
            rel = os.path.relpath(p, search_root)
            if _should_skip_glob(rel):
                continue
            filtered.append(rel)

        truncated = len(filtered) > limit
        if truncated:
            filtered = filtered[:limit]

        if not filtered:
            return _ok("No files found")

        filtered.sort()
        lines = filtered.copy()
        if truncated:
            lines.append(
                "(Results are truncated. Consider using a more specific "
                "path or pattern.)"
            )
        return _ok("\n".join(lines))

    if "glob" in enabled_tools:
        api.register_tool(
            FunctionTool(
                name="glob",
                description=(
                    "Find files by glob pattern. Returns matching filenames "
                    "relative to the search directory."
                ),
                parameters=_GLOB_PARAMETERS,
                fn=_glob_execute,
                metadata={"file_op": "glob"},
            )
        )

    # --- grep tool -------------------------------------------------------

    grep_default_limit = config.default_limit

    # Cache ripgrep availability (probed once on first grep invocation).
    _rg_available: list[bool | None] = [None]

    async def _probe_ripgrep() -> bool:
        if _rg_available[0] is not None:
            return _rg_available[0]
        try:
            result = await _get_bash_ops().exec(
                "which rg", cwd=cwd, timeout=5,
            )
            _rg_available[0] = result.exit_code == 0
        except Exception:
            _rg_available[0] = False
        return _rg_available[0]  # type: ignore[return-value]

    async def _grep_execute(args: dict[str, Any]) -> ToolResult:
        pattern: str = args["pattern"]
        path: str = args.get("path", cwd)
        glob_filter: str | None = args.get("glob")
        output_mode: str = args.get("output_mode", "content")
        case_insensitive: bool = bool(args.get("case_insensitive", False))
        limit: int = int(args.get("limit", grep_default_limit))
        context_lines: int | None = args.get("context_lines")

        if not os.path.isabs(path):
            path = os.path.normpath(os.path.join(cwd, path))

        bash_ops = _get_bash_ops()

        # Verify path exists.
        check = await bash_ops.exec(
            f"test -e {shlex.quote(path)} && echo yes || echo no",
            cwd=cwd,
            timeout=10,
        )
        if check.stdout.decode().strip() != "yes":
            return _error(f"Path does not exist: {path!r}")

        use_rg = await _probe_ripgrep()
        cmd = build_grep_or_rg_command(
            pattern,
            path,
            glob_filter=glob_filter,
            output_mode=output_mode,
            case_insensitive=case_insensitive,
            context_lines=context_lines,
            use_ripgrep=use_rg,
        )

        cmd_str = shlex.join(cmd)
        try:
            proc = await bash_ops.exec(cmd_str, cwd=cwd, timeout=30)
        except Exception as exc:
            return _error(f"Search failed: {exc}")

        # grep/rg exit 1 = no matches (not an error).
        if proc.exit_code not in (0, 1):
            stderr = proc.stderr.decode().strip()
            return _error(f"Search failed (exit {proc.exit_code}): {stderr}")

        text = parse_grep_output(
            proc.stdout.decode(),
            base_path=path,
            output_mode=output_mode,
            limit=limit,
        )
        return _ok(text)

    if "grep" in enabled_tools:
        api.register_tool(
            FunctionTool(
                name="grep",
                description=(
                    "Search file contents with a regex pattern. Uses ripgrep "
                    "(rg) when available, falls back to grep."
                ),
                parameters=_GREP_PARAMETERS,
                fn=_grep_execute,
            )
        )
