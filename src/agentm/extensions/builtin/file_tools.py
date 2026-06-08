"""Grouped file-I/O tool atom: ``read``, ``write``, and ``edit``.

Merges the former single-tool atoms ``tool_read``, ``tool_write``, and
``tool_edit`` into one §11-compliant module. The LLM-facing tool names
are unchanged (``read``, ``write``, ``edit``).

Read — aligned with Claude Code's FileReadTool behavior: no hardcoded
line cap, max-file-size gate (default 256 KB), partial-view tracking for
downstream edit/write safety.

Write — enforces Claude-Code-style safety gates: read-before-write for
existing files, file-modified-since-read detection, post-write read_state
update.

Edit — enforces read-before-edit, supports string replacement and
line-range replacement modes, file-modified-since-read detection,
post-edit read_state update.
"""

from __future__ import annotations

import fnmatch
import inspect
import os
from pathlib import Path, PurePath
from typing import Any, Final

from agentm.core.abi import FunctionTool, TextContent, ToolResult
from agentm.core.abi.tool import TOOL_RESULT_FORMAT_METADATA_KEY
from agentm.core.abi.operations import FileOperations
from agentm.core.lib.read_state import (
    content_hash_for,
    file_modified_since_read,
    get_read_state,
    record_read,
)
from agentm.extensions import ExtensionManifest
from agentm.core.abi.extension import ExtensionAPI


# ---------------------------------------------------------------------------
# MANIFEST
# ---------------------------------------------------------------------------

MANIFEST = ExtensionManifest(
    name="file_tools",
    description="Register the read, write, and edit tools for file I/O.",
    registers=("tool:read", "tool:write", "tool:edit"),
    config_schema={
        "type": "object",
        "properties": {
            "file_ops": {"type": "object"},
            "allow_globs": {
                "type": "array",
                "items": {"type": "string"},
            },
            "deny_globs": {
                "type": "array",
                "items": {"type": "string"},
            },
            "max_size_bytes": {
                "type": "integer",
                "default": 262_144,
            },
            "require_read": {
                "type": "boolean",
                "default": True,
                "description": "Require existing files to be read before overwriting/editing.",
            },
        },
        "additionalProperties": True,
    },
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


def _update_read_state_after_edit(normalized_path: str) -> None:
    """Refresh read_state for *normalized_path* after a successful edit."""
    old = get_read_state(normalized_path)
    total_lines = old.total_lines if old else 0
    is_partial = old.is_partial if old else False
    try:
        stat = os.stat(normalized_path)
        mtime_ns = stat.st_mtime_ns
    except OSError:
        mtime_ns = 0
    try:
        with open(normalized_path, "rb") as fh:
            raw = fh.read()
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
# install()
# ---------------------------------------------------------------------------

def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    # Lazy-resolve file_ops and writer: at install time, the Operations
    # bundle or ResourceWriter may not be registered yet (depends on atom
    # load order). Deferring to first tool invocation avoids an install-time
    # ordering dependency while keeping requires=() (any Operations
    # provider is acceptable, not just operations_local).
    _file_ops_cfg = config.get("file_ops")
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

    allow_globs = _coerce_globs(config.get("allow_globs"), api.cwd)
    deny_globs = _coerce_globs(config.get("deny_globs"), api.cwd)
    max_size_bytes: int = int(
        config.get("max_size_bytes", _DEFAULT_MAX_SIZE_BYTES)
    )
    require_read = bool(config.get("require_read", True))

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
        except Exception:
            pass

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
                    current_mtime = os.stat(normalized).st_mtime_ns
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
                disk_mtime = os.stat(normalized).st_mtime_ns
                sig = inspect.signature(record_read)
                if "mtime_ns" in sig.parameters:
                    record_kwargs["mtime_ns"] = disk_mtime
            except OSError:
                pass
            record_read(normalized, **record_kwargs)

            action = "Updated" if file_exists else "Created"
            byte_count = len(content.encode("utf-8"))
            return _ok(f"{action} {path!r} ({byte_count} bytes)")
        except Exception as exc:
            return _error(f"Failed to write {path!r}: {exc}")

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
                _update_read_state_after_edit(normalized)

            return result
        except Exception as exc:
            return _error(f"Failed to edit {path!r}: {exc}")

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
