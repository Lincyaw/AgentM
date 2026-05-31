"""Tool atom for the ``extensions.builtin.tool_edit`` §7.1 row.

Enforces read-before-edit: a file must have been read via tool_read in
the current session before it can be edited. Supports two modes:

1. **String replacement** (``old_string`` + ``new_string``): classic
   search-replace, same as before.
2. **Line-range replacement** (``start_line`` + ``end_line`` + ``new_string``):
   replace lines [start, end] (1-based, inclusive) with new_string.
   Requires a prior read so the agent has seen the line numbers.

Aligned with Claude Code's ``FileEditTool``: file-modified-since-read
detection, improved whitespace-tolerant string matching, and post-edit
read-state update.
"""

from __future__ import annotations

import os
from typing import Any, Final

from agentm.core.abi import FunctionTool, TextContent, ToolResult
from agentm.core.abi.tool import TOOL_RESULT_FORMAT_METADATA_KEY
from agentm.core.lib.read_state import (
    content_hash_for,
    file_modified_since_read,
    get_read_state,
    record_read,
)
from agentm.extensions import ExtensionManifest
from agentm.core.abi.extension import ExtensionAPI


MANIFEST = ExtensionManifest(
    name="tool_edit",
    description="Register the edit tool backed by ResourceWriter.",
    registers=("tool:edit",),
    config_schema={
        "type": "object",
        "properties": {
            "require_read": {
                "type": "boolean",
                "default": True,
                "description": "Require the file to be read before editing.",
            },
        },
        "additionalProperties": True,
    },
    requires=(),
)

_PARAMETERS: Final = {
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
        "rationale": {"type": "string", "default": "agent edit via tool_edit"},
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
    """Return a snippet of *content* showing ±CONTEXT_LINES around [start, end] with line numbers."""
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
    """Refresh read_state for *normalized_path* after a successful edit.

    Without this, the next edit to the same file would trip the
    file-modified-since-read guard because our own write bumped the mtime.
    """
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


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    writer = api.get_resource_writer()
    require_read = bool(config.get("require_read", True))

    async def _execute(args: dict[str, Any]) -> ToolResult:
        path = str(args["path"])
        new_string = str(args["new_string"])
        old_string = args.get("old_string")
        start_line = args.get("start_line")
        end_line = args.get("end_line")
        replace_all = bool(args.get("replace_all", False))
        rationale = str(args.get("rationale", "agent edit via tool_edit"))

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
            original = (await writer.read(path)).decode("utf-8", errors="replace")

            if has_lines:
                result = await _line_range_replace(
                    writer, path, original, int(start_line), int(end_line),
                    new_string, rationale,
                )
            else:
                result = await _string_replace(
                    writer, path, original, str(old_string), new_string,
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
            parameters=_PARAMETERS,
            fn=_execute,
            metadata={"file_op": "edit", TOOL_RESULT_FORMAT_METADATA_KEY: "diff"},
        )
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
        # Map the position in the stripped file back to the original.
        # Walk the original file's lines to find the matching range.
        orig_lines = file_content.split("\n")
        stripped_lines = stripped_file.split("\n")
        # Find which line in stripped_file the match starts on
        prefix = stripped_file[:idx]
        start_line = prefix.count("\n")
        # Count how many stripped lines the search spans
        search_line_count = stripped_search.count("\n") + 1
        # Extract corresponding lines from the original
        matched_lines = orig_lines[start_line : start_line + search_line_count]
        return "\n".join(matched_lines)

    return None


async def _string_replace(
    writer: Any,
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
    result = await writer.replace(
        path, original.encode("utf-8"), updated.encode("utf-8"), rationale=rationale,
    )
    if result.error is not None:
        return _error(result.error)
    before_lines = original[: original.index(actual)].count("\n")
    new_lines_count = new_string.count("\n") + 1
    snippet = _snippet_around(updated, before_lines + 1, before_lines + new_lines_count)
    return _ok(f"Updated {path!r}:\n{snippet}")


async def _line_range_replace(
    writer: Any,
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
    replaced_len = sum(len(l) for l in lines[start - 1 : end])
    shrinkage = _check_shrinkage(original, updated, replaced_len, len(new_string))
    if shrinkage:
        return _error(shrinkage)
    result = await writer.replace(
        path, original.encode("utf-8"), updated.encode("utf-8"), rationale=rationale,
    )
    if result.error is not None:
        return _error(result.error)
    new_line_count = new_string.count("\n") + 1
    snippet = _snippet_around(updated, start, start + new_line_count - 1)
    return _ok(f"Replaced lines {start}-{end} in {path!r}:\n{snippet}")


def _ok(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)])


def _error(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)], is_error=True)
