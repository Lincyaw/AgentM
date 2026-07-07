"""Grouped file-I/O tool atom: ``read``, ``write``, ``edit``.

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
"""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path, PurePath
from typing import Any, Final

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator

from agentm.core.abi import (
    ExtensionAPI,
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

_ALL_TOOLS: Final[frozenset[str]] = frozenset({"read", "write", "edit"})


class FileToolsConfig(BaseModel):
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    tools: list[str] | None = None
    allow_globs: list[str] | None = None
    deny_globs: list[str] | None = None
    max_size_bytes: int = 262_144
    require_read: bool = True
    default_limit: int = 250

    @field_validator("tools")
    @classmethod
    def _validate_tools(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        requested = frozenset(value)
        unknown = requested - _ALL_TOOLS
        if unknown:
            allowed = ", ".join(sorted(_ALL_TOOLS))
            bad = ", ".join(sorted(unknown))
            raise ValueError(
                f"unknown file_tools tool(s): {bad}; allowed tools: {allowed}"
            )
        return value


MANIFEST = ExtensionManifest(
    name="file_tools",
    description="Register the read, write, and edit tools for guarded file I/O.",
    registers=("tool:read", "tool:write", "tool:edit"),
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


_PATH_ALIASES: Final[tuple[str, ...]] = ("file_path",)


def _required_string_arg(
    args: dict[str, Any],
    key: str,
    tool_name: str,
    *,
    aliases: tuple[str, ...] = (),
    allow_empty: bool = False,
    hint: str,
) -> tuple[str | None, ToolResult | None]:
    """Return a required string argument or a model-facing tool error."""
    supplied_name = next((name for name in (key, *aliases) if name in args), None)
    if supplied_name is None:
        alias_text = ""
        if aliases:
            alias_text = f" Accepted aliases: {', '.join(repr(a) for a in aliases)}."
        return (
            None,
            _error(
                f"Invalid {tool_name} call: missing required argument {key!r}."
                f"{alias_text} Use {hint}."
            ),
        )

    value = args[supplied_name]
    if not isinstance(value, str):
        return (
            None,
            _error(
                f"Invalid {tool_name} call: argument {supplied_name!r} must be a "
                f"string, got {type(value).__name__}. Use {hint}."
            ),
        )
    if not allow_empty and value == "":
        return (
            None,
            _error(
                f"Invalid {tool_name} call: argument {supplied_name!r} must not "
                f"be empty. Use {hint}."
            ),
        )
    return value, None


def _read_state_path(path: str, cwd: str) -> str:
    """Return the stable key used for read-before-write/edit state."""
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(cwd, path))


# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------

# 256 KB — matches Claude Code's MAX_OUTPUT_SIZE (0.25 * 1024 * 1024).
_DEFAULT_MAX_SIZE_BYTES: Final[int] = 262_144

_BINARY_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {
        # Video
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".webm",
        ".flv",
        ".wmv",
        # Audio
        ".mp3",
        ".wav",
        ".flac",
        ".aac",
        ".ogg",
        ".wma",
        ".m4a",
        # Image
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".tiff",
        ".webp",
        ".ico",
        ".svg",
        # Archive
        ".zip",
        ".tar",
        ".gz",
        ".bz2",
        ".xz",
        ".7z",
        ".rar",
        # Binary / native
        ".bin",
        ".exe",
        ".dll",
        ".so",
        ".dylib",
        ".o",
        ".a",
        ".pyc",
        ".class",
        # Documents
        ".pdf",
        ".doc",
        ".docx",
        ".xls",
        ".xlsx",
        ".ppt",
        ".pptx",
        # Database
        ".sqlite",
        ".db",
    }
)


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
        return f"Access denied: {path!r} matches a configured deny_glob ({list(deny)})."
    return None


class _ReadArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(
        description="Path to the file to read (absolute, or relative to the session cwd)."
    )
    offset: int | None = Field(
        default=None,
        description=(
            "1-based line number to start reading from. Providing offset "
            "and/or limit lifts the size gate on large files."
        ),
    )
    limit: int | None = Field(
        default=None,
        description=(
            "Number of lines to read. Providing offset and/or limit lifts "
            "the size gate on large files."
        ),
    )


# ---------------------------------------------------------------------------
# Edit helpers
# ---------------------------------------------------------------------------


class _EditArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(description="File path to edit.")
    old_string: str | None = Field(
        default=None,
        description="Exact text to find and replace. Mutually exclusive with start_line/end_line.",
    )
    new_string: str = Field(description="Replacement text.")
    start_line: int | None = Field(
        default=None,
        description="1-based start line for line-range replacement (inclusive).",
    )
    end_line: int | None = Field(
        default=None,
        description="1-based end line for line-range replacement (inclusive).",
    )
    replace_all: bool = Field(
        default=False,
        description=(
            "Replace every occurrence of old_string. Without this, "
            "old_string must match exactly once or the edit is rejected."
        ),
    )
    rationale: str = Field(default="agent edit via file_tools")


_CONTEXT_LINES = 4
_MAX_UNINTENDED_SHRINK_LINES = 5


def _check_shrinkage(
    original: str, updated: str, old_len: int, new_len: int
) -> str | None:
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
    path: str,
    state_key: str,
    writer: Any,
) -> None:
    """Refresh read_state after a successful edit.

    ``path`` is the path the agent used (the writer backend resolves it
    against its own working directory, e.g. the sandbox workspace);
    ``state_key`` is the host-side bookkeeping key. Reading back with the
    state key would send a host path to a remote backend — wrong file space.
    """
    old = get_read_state(state_key)
    total_lines = old.total_lines if old else 0
    is_partial = old.is_partial if old else False
    chash = old.content_hash if old else ""
    try:
        raw = await writer.read(path)
        chash = content_hash_for(raw)
        total_lines = raw.decode("utf-8", errors="replace").count("\n") + 1
    except (OSError, FileNotFoundError) as exc:
        logger.warning(
            "file_tools: post-edit read({}) failed: {}", path, exc
        )
    record_read(
        state_key,
        total_lines=total_lines,
        is_partial=is_partial,
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


class _WriteArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(description="File path to write.")
    content: str = Field(description="The full content to write.")
    rationale: str = Field(default="agent write via file_tools")


# ---------------------------------------------------------------------------
# install()
# ---------------------------------------------------------------------------


def install(api: ExtensionAPI, config: FileToolsConfig) -> None:
    _FileToolsRuntime(api=api, config=config).install()


def _enabled_tools(configured: list[str] | None) -> frozenset[str]:
    if configured is None:
        return _ALL_TOOLS
    enabled_tools = frozenset(configured)
    unknown = enabled_tools - _ALL_TOOLS
    if unknown:
        allowed = ", ".join(sorted(_ALL_TOOLS))
        requested = ", ".join(sorted(unknown))
        raise ValueError(
            f"unknown file_tools tool(s): {requested}; allowed tools: {allowed}"
        )
    return enabled_tools


class _FileToolsRuntime:
    """Owns file_tools registration and per-session handler state."""

    def __init__(self, *, api: ExtensionAPI, config: FileToolsConfig) -> None:
        self._api = api
        self._writer_cache: list[Any] = []
        self._enabled_tools = _enabled_tools(config.tools)
        self._allow_globs = _coerce_globs(config.allow_globs, api.cwd)
        self._deny_globs = _coerce_globs(config.deny_globs, api.cwd)
        self._max_size_bytes = config.max_size_bytes
        self._require_read = config.require_read

    def install(self) -> None:
        if "read" in self._enabled_tools:
            self._register_read()
        if "write" in self._enabled_tools:
            self._register_write()
        if "edit" in self._enabled_tools:
            self._register_edit()

    def _get_writer(self) -> Any:
        if not self._writer_cache:
            self._writer_cache.append(self._api.get_resource_writer())
        return self._writer_cache[0]

    def _register_read(self) -> None:
        self._api.register_tool(
            FunctionTool(
                name="read",
                description=(
                    "Read a UTF-8 text file from disk. "
                    "By default reads the entire file. "
                    f"Files larger than {self._max_size_bytes} bytes require "
                    "offset and/or limit parameters. "
                    "Output starts with a header line giving the total or "
                    "shown line range, followed by 1-based line-numbered "
                    "content (`N\\tcontent`) — the same numbers the edit "
                    "tool's start_line/end_line refer to. "
                    "Known binary formats (images, archives, pdf, ...) are "
                    "rejected; inspect those via bash instead."
                ),
                parameters=_ReadArgs,
                fn=self._read_execute,
                metadata={"file_op": "read"},
            )
        )

    async def _read_execute(self, args: dict[str, Any]) -> ToolResult:
        path, arg_error = _required_string_arg(
            args,
            "path",
            "read",
            aliases=_PATH_ALIASES,
            hint='{"path": "..."}',
        )
        if arg_error is not None:
            return arg_error
        assert path is not None

        gate_error = _check_path_allowed(path, self._allow_globs, self._deny_globs)
        if gate_error is not None:
            return _error(gate_error)

        binary_error = _check_binary(path)
        if binary_error is not None:
            return _error(binary_error)

        try:
            data = await self._get_writer().read(path)
        except Exception as exc:
            logger.debug("file_tools: read failed for {}: {}", path, exc)
            return _error(f"Failed to read {path!r}: {exc}")

        raw_offset = args.get("offset")
        raw_limit = args.get("limit")
        file_size = len(data)
        caller_wants_range = raw_offset is not None or raw_limit is not None
        if file_size > self._max_size_bytes and not caller_wants_range:
            return _error(
                f"File content ({file_size} bytes) exceeds maximum "
                f"allowed size ({self._max_size_bytes} bytes). "
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

            is_partial = offset > 0 or (
                limit is not None and limit > 0 and offset + limit < total
            )

            record_read(
                _read_state_path(path, self._api.cwd),
                total_lines=total,
                is_partial=is_partial,
                content_hash=content_hash_for(data),
            )

            numbered = [f"{offset + i + 1}\t{line}" for i, line in enumerate(sliced)]

            if is_partial:
                end_line = offset + len(sliced)
                header = f"(showing lines {offset + 1}-{end_line} of {total})"
            else:
                header = f"({total} lines total)"

            return _ok(header + "\n" + "\n".join(numbered))
        except Exception as exc:
            logger.debug("file_tools: read decode failed for {}: {}", path, exc)
            return _error(f"Failed to read {path!r}: {exc}")

    def _register_write(self) -> None:
        self._api.register_tool(
            FunctionTool(
                name="write",
                description=(
                    "Write a UTF-8 text file. For existing files, you MUST read "
                    "the full file first. Prefer the edit tool for modifying "
                    "existing files — use write only for new files or complete "
                    "rewrites."
                ),
                parameters=_WriteArgs,
                fn=self._write_execute,
                metadata={"file_op": "write"},
            )
        )

    async def _write_execute(self, args: dict[str, Any]) -> ToolResult:
        path, arg_error = _required_string_arg(
            args,
            "path",
            "write",
            aliases=_PATH_ALIASES,
            hint='{"path": "...", "content": "..."}',
        )
        if arg_error is not None:
            return arg_error
        assert path is not None

        content, arg_error = _required_string_arg(
            args,
            "content",
            "write",
            allow_empty=True,
            hint='{"path": "...", "content": "..."}',
        )
        if arg_error is not None:
            return arg_error
        assert content is not None
        rationale = str(args.get("rationale", "agent write via file_tools"))

        read_state_path = _read_state_path(path, self._api.cwd)
        writer = self._get_writer()
        try:
            file_exists = await writer.exists(path)
        except Exception as exc:
            # Unknown probe failure (gateway hiccup, transport error): do
            # NOT assume the file is absent — that would silently disable
            # the read-before-overwrite guard and allow a blind overwrite
            # of an existing file. Fail the write instead, in plain words
            # the agent can act on.
            logger.warning("file_tools: exists({!r}) probe failed: {}", path, exc)
            return _error(
                f"Could not check whether {path!r} already exists "
                f"(temporary file-system error: {exc}). Nothing was "
                "written. Retry the write; if it keeps failing, read the "
                "file first to confirm its state."
            )

        if file_exists and self._require_read:
            rs = get_read_state(read_state_path)
            if rs is None:
                return _error(
                    f"File {path!r} already exists. Read it first before "
                    "overwriting so you can see its current content. "
                    "Use the read tool, then write."
                )
            if rs.is_partial:
                return _error(
                    f"You read {path!r} with offset/limit (partial view). "
                    "Read the full file before overwriting."
                )
            if rs.content_hash:
                try:
                    current_data = await writer.read(path)
                    current_hash = content_hash_for(current_data)
                except (OSError, FileNotFoundError):
                    current_hash = None
                if current_hash is not None and current_hash != rs.content_hash:
                    return _error(
                        "File has been modified since you read it. "
                        "Read it again before writing."
                    )

        try:
            content_bytes = content.encode("utf-8")
            result = await writer.write(path, content_bytes, rationale=rationale)
            if result.error is not None:
                return _error(result.error)

            total_lines = content.count("\n") + (1 if content else 0)
            record_read(
                read_state_path,
                total_lines=total_lines,
                is_partial=False,
                content_hash=content_hash_for(content_bytes),
            )

            action = "Updated" if file_exists else "Created"
            byte_count = len(content.encode("utf-8"))
            return _ok(f"{action} {path!r} ({byte_count} bytes)")
        except Exception as exc:
            logger.debug("file_tools: write failed for {}: {}", path, exc)
            return _error(f"Failed to write {path!r}: {exc}")

    def _register_edit(self) -> None:
        self._api.register_tool(
            FunctionTool(
                name="edit",
                description=(
                    "Edit a UTF-8 text file. Two modes:\n"
                    "1. String replacement: provide old_string + new_string. "
                    "old_string must match exactly once unless replace_all "
                    "is set.\n"
                    "2. Line-range replacement: provide start_line + end_line "
                    "+ new_string (1-based, inclusive).\n"
                    "You MUST read the file first (both modes); an edit whose "
                    "prior read is stale or partial is rejected. An edit that "
                    "deletes many more lines than the match explains is also "
                    "rejected — use line-range mode for large intentional "
                    "deletions. Returns a line-numbered diff snippet of the "
                    "changed region."
                ),
                parameters=_EditArgs,
                fn=self._edit_execute,
                metadata={"file_op": "edit", TOOL_RESULT_FORMAT_METADATA_KEY: "diff"},
            )
        )

    async def _string_replace(
        self,
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
        result = await self._get_writer().replace(
            path,
            original.encode("utf-8"),
            updated.encode("utf-8"),
            rationale=rationale,
        )
        if result.error is not None:
            return _error(result.error)
        before_lines = original[: original.index(actual)].count("\n")
        new_lines_count = new_string.count("\n") + 1
        snippet = _snippet_around(
            updated, before_lines + 1, before_lines + new_lines_count
        )
        return _ok(f"Updated {path!r}:\n{snippet}")

    async def _line_range_replace(
        self,
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
        result = await self._get_writer().replace(
            path,
            original.encode("utf-8"),
            updated.encode("utf-8"),
            rationale=rationale,
        )
        if result.error is not None:
            return _error(result.error)
        new_line_count = new_string.count("\n") + 1
        snippet = _snippet_around(updated, start, start + new_line_count - 1)
        return _ok(f"Replaced lines {start}-{end} in {path!r}:\n{snippet}")

    async def _edit_execute(self, args: dict[str, Any]) -> ToolResult:
        path, arg_error = _required_string_arg(
            args,
            "path",
            "edit",
            aliases=_PATH_ALIASES,
            hint='{"path": "...", "old_string": "...", "new_string": "..."}',
        )
        if arg_error is not None:
            return arg_error
        assert path is not None

        new_string, arg_error = _required_string_arg(
            args,
            "new_string",
            "edit",
            allow_empty=True,
            hint='{"path": "...", "old_string": "...", "new_string": "..."}',
        )
        if arg_error is not None:
            return arg_error
        assert new_string is not None

        old_string = args.get("old_string")
        start_line = args.get("start_line")
        end_line = args.get("end_line")
        replace_all = bool(args.get("replace_all", False))
        rationale = str(args.get("rationale", "agent edit via file_tools"))

        read_state_path = _read_state_path(path, self._api.cwd)
        state = get_read_state(read_state_path)
        if self._require_read and state is None:
            return _error(
                f"You must read {path!r} before editing it. "
                "Use the read tool first so you can see the exact content and line numbers."
            )

        if state is not None and file_modified_since_read(read_state_path):
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
            original = (await self._get_writer().read(path)).decode(
                "utf-8", errors="replace"
            )

            if start_line is not None and end_line is not None:
                result = await self._line_range_replace(
                    path,
                    original,
                    int(start_line),
                    int(end_line),
                    new_string,
                    rationale,
                )
            else:
                result = await self._string_replace(
                    path,
                    original,
                    str(old_string),
                    new_string,
                    replace_all,
                    rationale,
                )

            if not result.is_error:
                await _update_read_state_after_edit(
                    path, read_state_path, self._get_writer()
                )

            return result
        except Exception as exc:
            logger.opt(exception=True).warning("edit tool failed for {}: {}", path, exc)
            return _error(f"Failed to edit {path!r}: {exc}")
