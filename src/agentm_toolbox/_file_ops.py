"""Deterministic file-tool rendering and mutation planning."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import PurePath
from typing import Final

from agentm_toolbox._match import find_actual_string
from agentm_toolbox._state import (
    FileReadState,
    LineRange,
    ReadStateStore,
    content_hash_for,
)

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Result:
    """Outcome of a tool invocation.  ``text`` is the LLM-visible output."""

    text: str
    is_error: bool = False
    total_lines: int = 0
    is_partial: bool = False
    content_hash: str = ""
    read_ranges: tuple[LineRange, ...] | None = None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CONTEXT_LINES: Final[int] = 4
_MAX_UNINTENDED_SHRINK_LINES: Final[int] = 5

_BINARY_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".webm",
        ".flv",
        ".wmv",
        ".mp3",
        ".wav",
        ".flac",
        ".aac",
        ".ogg",
        ".wma",
        ".m4a",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".tiff",
        ".webp",
        ".ico",
        ".svg",
        ".zip",
        ".tar",
        ".gz",
        ".bz2",
        ".xz",
        ".7z",
        ".rar",
        ".bin",
        ".exe",
        ".dll",
        ".so",
        ".dylib",
        ".o",
        ".a",
        ".pyc",
        ".class",
        ".pdf",
        ".doc",
        ".docx",
        ".xls",
        ".xlsx",
        ".ppt",
        ".pptx",
        ".sqlite",
        ".db",
    }
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve(cwd: str, path: str) -> str:
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(cwd, path))


def _check_binary(path: str) -> str | None:
    ext = PurePath(path).suffix.lower()
    if ext in _BINARY_EXTENSIONS:
        return (
            f"Cannot read binary file {path!r} ({ext} format). "
            "Use bash to inspect metadata (e.g. `file <path>`, `ls -la <path>`) "
            "or process it with appropriate tools."
        )
    return None


def _snippet_around(content: str, start_line: int, end_line: int) -> str:
    lines = content.splitlines()
    total = len(lines)
    snippet_start = max(0, start_line - 1 - _CONTEXT_LINES)
    snippet_end = min(total, end_line + _CONTEXT_LINES)
    numbered = [
        f"{snippet_start + i + 1}\t{line}"
        for i, line in enumerate(lines[snippet_start:snippet_end])
    ]
    return "\n".join(numbered)


def _snippet_range(content: str, start_line: int, end_line: int) -> LineRange | None:
    total = _line_count(content)
    if total <= 0:
        return None
    snippet_start = max(1, start_line - _CONTEXT_LINES)
    snippet_end = min(total, end_line + _CONTEXT_LINES)
    if snippet_start > snippet_end:
        return None
    return (snippet_start, snippet_end)


def _line_count(content: str) -> int:
    return len(content.splitlines())


def _find_spans(content: str, needle: str) -> tuple[tuple[int, int], ...]:
    spans: list[tuple[int, int]] = []
    start = 0
    while True:
        index = content.find(needle, start)
        if index < 0:
            return tuple(spans)
        end = index + len(needle)
        spans.append((index, end))
        start = end


def _span_line_range(content: str, start: int, end: int) -> LineRange:
    total = _line_count(content)
    start_line = content.count("\n", 0, start) + 1
    newline_count = content[start:end].count("\n")
    end_line = min(total, start_line + newline_count)
    return (start_line, max(start_line, end_line))


def _replacement_line_count(
    old_range: LineRange, old_string: str, new_string: str
) -> int:
    old_start, old_end = old_range
    old_line_count = old_end - old_start + 1
    newline_delta = new_string.count("\n") - old_string.count("\n")
    return max(0, old_line_count + newline_delta)


def _merge_line_ranges(ranges: tuple[LineRange, ...]) -> tuple[LineRange, ...]:
    if not ranges:
        return ()
    ordered = sorted(ranges)
    merged: list[LineRange] = [ordered[0]]
    for start, end in ordered[1:]:
        previous_start, previous_end = merged[-1]
        if start <= previous_end + 1:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return tuple(merged)


def _translate_read_ranges(
    ranges: tuple[LineRange, ...],
    *,
    edit_start: int,
    edit_end: int,
    new_line_count: int,
) -> tuple[LineRange, ...]:
    old_line_count = edit_end - edit_start + 1
    delta = new_line_count - old_line_count
    translated: list[LineRange] = []
    for start, end in ranges:
        if end < edit_start:
            translated.append((start, end))
            continue
        if start > edit_end:
            translated.append((start + delta, end + delta))
            continue
        if start < edit_start:
            translated.append((start, edit_start - 1))
        if new_line_count > 0:
            translated.append((edit_start, edit_start + new_line_count - 1))
        if end > edit_end:
            translated.append((edit_start + new_line_count, end + delta))
    return _merge_line_ranges(
        tuple((start, end) for start, end in translated if start <= end)
    )


def _check_shrinkage(
    original: str, updated: str, old_len: int, new_len: int
) -> str | None:
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


# ---------------------------------------------------------------------------
# FileToolbox
# ---------------------------------------------------------------------------


class FileToolbox:
    """Stateful, I/O-free file-tool planner. One instance per session.

    Callers provide the current bytes and apply planned mutations through
    their authoritative resource backend. This keeps validation and output
    identical across local, sandbox, remote, and transactional environments.
    """

    def __init__(
        self,
        *,
        cwd: str = ".",
        max_size: int = 262_144,
        require_read: bool = True,
        default_limit: int = 250,
        state: ReadStateStore | None = None,
    ) -> None:
        self._cwd = os.path.abspath(cwd)
        self._max_size = max_size
        self._require_read = require_read
        self._default_limit = default_limit
        self.state = state or ReadStateStore()

    # -- read ---------------------------------------------------------------

    def read_bytes(
        self,
        path: str,
        data: bytes,
        *,
        offset: int | None = None,
        limit: int | None = None,
        mtime_ns: int = 0,
    ) -> Result:
        """Render and record a read from an environment-provided byte view."""

        binary_err = _check_binary(path)
        if binary_err is not None:
            return Result(text=binary_err, is_error=True)
        caller_wants_range = offset is not None or limit is not None
        if len(data) > self._max_size and not caller_wants_range:
            return Result(
                text=(
                    f"File content ({len(data)} bytes) exceeds maximum "
                    f"allowed size ({self._max_size} bytes). "
                    "Use offset and limit parameters to read specific "
                    "portions of the file."
                ),
                is_error=True,
            )

        try:
            all_lines = data.decode("utf-8", errors="replace").splitlines()
            total = len(all_lines)

            off = max(0, int(offset) - 1) if offset is not None else 0
            caller_wants_range = offset is not None or limit is not None
            if limit is not None:
                lim = int(limit)
            elif (
                not caller_wants_range
                and self._default_limit
                and total > self._default_limit
            ):
                lim = self._default_limit
            else:
                lim = None

            if lim is not None and lim > 0:
                sliced = all_lines[off : off + lim]
            else:
                sliced = all_lines[off:]

            is_partial = off > 0 or (lim is not None and lim > 0 and off + lim < total)

            chash = content_hash_for(data)
            read_start = off + 1
            read_end = off + len(sliced)
            if read_start <= read_end:
                self.state.record(
                    _resolve(self._cwd, path),
                    total_lines=total,
                    is_partial=is_partial,
                    mtime_ns=mtime_ns,
                    content_hash=chash,
                    start_line=read_start,
                    end_line=read_end,
                )
            else:
                self.state.record(
                    _resolve(self._cwd, path),
                    total_lines=total,
                    is_partial=is_partial,
                    mtime_ns=mtime_ns,
                    content_hash=chash,
                )

            numbered = [f"{off + i + 1}\t{line}" for i, line in enumerate(sliced)]

            if is_partial:
                end_line = off + len(sliced)
                header = f"(showing lines {off + 1}-{end_line} of {total})"
            else:
                header = f"({total} lines total)"

            return Result(
                text=header + "\n" + "\n".join(numbered),
                total_lines=total,
                is_partial=is_partial,
                content_hash=chash,
            )
        except Exception as exc:
            return Result(text=f"Failed to read {path!r}: {exc}", is_error=True)

    # -- write --------------------------------------------------------------

    def plan_write(
        self,
        path: str,
        current: bytes | None,
        content: str,
    ) -> tuple[Result, bytes | None]:
        """Validate a write against a supplied view without mutating it."""

        resolved = _resolve(self._cwd, path)
        file_exists = current is not None
        if file_exists and self._require_read:
            rs = self.state.get(resolved)
            if rs is None:
                return (
                    Result(
                        text=(
                            f"File {path!r} already exists. Read it first before "
                            "overwriting so you can see its current content. "
                            "Use the read tool, then write."
                        ),
                        is_error=True,
                    ),
                    None,
                )
            if rs.is_partial:
                return (
                    Result(
                        text=(
                            f"You read {path!r} with offset/limit (partial view). "
                            "Read the full file before overwriting."
                        ),
                        is_error=True,
                    ),
                    None,
                )
            if rs.content_hash:
                assert current is not None
                current_hash = content_hash_for(current)
                if current_hash != rs.content_hash:
                    return (
                        Result(
                            text=(
                                "File has been modified since you read it. "
                                "Read it again before writing."
                            ),
                            is_error=True,
                        ),
                        None,
                    )

        content_bytes = content.encode("utf-8")
        total_lines = _line_count(content)
        chash = content_hash_for(content_bytes)
        action = "Updated" if file_exists else "Created"
        return (
            Result(
                text=f"{action} {path!r} ({len(content_bytes)} bytes)",
                total_lines=total_lines,
                content_hash=chash,
            ),
            content_bytes,
        )

    def accept_content(
        self,
        path: str,
        content: bytes,
        *,
        is_partial: bool = False,
        mtime_ns: int = 0,
        read_ranges: tuple[LineRange, ...] | None = None,
    ) -> None:
        """Advance read state after a planned mutation was successfully staged."""

        decoded = content.decode("utf-8", errors="replace")
        self.state.record(
            _resolve(self._cwd, path),
            total_lines=_line_count(decoded),
            is_partial=is_partial,
            mtime_ns=mtime_ns,
            content_hash=content_hash_for(content),
            ranges=read_ranges,
        )

    # -- edit ---------------------------------------------------------------

    def plan_edit(
        self,
        path: str,
        current: bytes | None,
        *,
        old_string: str | None = None,
        new_string: str = "",
        start_line: int | None = None,
        end_line: int | None = None,
        replace_all: bool = False,
    ) -> tuple[Result, bytes | None]:
        """Validate and calculate an edit against a supplied byte view."""

        resolved = _resolve(self._cwd, path)
        if current is None:
            return (
                Result(
                    text=f"Failed to read {path!r}: file does not exist", is_error=True
                ),
                None,
            )
        rs = self.state.get(resolved)
        if self._require_read and rs is None:
            return (
                Result(
                    text=(
                        f"You must read {path!r} before editing it. "
                        "Use the read tool first so you can see the exact "
                        "content and line numbers."
                    ),
                    is_error=True,
                ),
                None,
            )
        if (
            rs is not None
            and rs.content_hash
            and content_hash_for(current) != rs.content_hash
        ):
            return (
                Result(
                    text=(
                        f"File has been modified since you last read it. "
                        f"Read {path!r} again before editing."
                    ),
                    is_error=True,
                ),
                None,
            )

        has_old = old_string is not None and old_string != ""
        has_lines = start_line is not None and end_line is not None
        if has_old and has_lines:
            return (
                Result(
                    text="Provide either old_string OR start_line/end_line, not both.",
                    is_error=True,
                ),
                None,
            )
        if not has_old and not has_lines:
            return (
                Result(
                    text="Provide old_string or start_line + end_line.",
                    is_error=True,
                ),
                None,
            )

        original = current.decode("utf-8", errors="replace")
        if has_lines:
            assert start_line is not None
            assert end_line is not None
            return self._plan_line_range_replace(
                path,
                original,
                start_line,
                end_line,
                new_string,
                rs,
            )
        assert old_string is not None
        return self._plan_string_replace(
            path,
            original,
            old_string,
            new_string,
            replace_all,
            rs,
        )

    def _check_edit_read_coverage(
        self,
        path: str,
        read_state: FileReadState | None,
        ranges: tuple[LineRange, ...],
    ) -> Result | None:
        if not self._require_read or read_state is None:
            return None
        for start, end in ranges:
            if read_state.covers(start, end):
                continue
            label = f"line {start}" if start == end else f"lines {start}-{end}"
            return Result(
                text=(
                    f"You have not read {label} of {path!r} in the current "
                    "file version. Read that range before editing it."
                ),
                is_error=True,
            )
        return None

    def _read_ranges_after_edit(
        self,
        read_state: FileReadState | None,
        *,
        edits: tuple[tuple[int, int, int], ...],
        snippet_ranges: tuple[LineRange, ...],
    ) -> tuple[LineRange, ...] | None:
        if not self._require_read or read_state is None or not read_state.is_partial:
            return None
        if len(edits) != 1:
            return _merge_line_ranges(snippet_ranges)
        edit_start, edit_end, new_line_count = edits[0]
        translated = _translate_read_ranges(
            read_state.ranges,
            edit_start=edit_start,
            edit_end=edit_end,
            new_line_count=new_line_count,
        )
        return _merge_line_ranges(translated + snippet_ranges)

    def _plan_string_replace(
        self,
        path: str,
        original: str,
        old_string: str,
        new_string: str,
        replace_all: bool,
        read_state: FileReadState | None,
    ) -> tuple[Result, bytes | None]:
        actual = find_actual_string(original, old_string)
        if actual is None:
            return (
                Result(
                    text=f"String not found in {path!r}: {old_string!r}",
                    is_error=True,
                ),
                None,
            )
        spans = _find_spans(original, actual)
        occurrences = len(spans)
        if not replace_all and occurrences != 1:
            return (
                Result(
                    text=(
                        f"String is not unique in {path!r}: found {occurrences} matches"
                    ),
                    is_error=True,
                ),
                None,
            )
        selected_spans = spans if replace_all else spans[:1]
        edit_ranges = tuple(
            _span_line_range(original, start, end) for start, end in selected_spans
        )
        coverage_error = self._check_edit_read_coverage(
            path, read_state, edit_ranges
        )
        if coverage_error is not None:
            return coverage_error, None
        updated = (
            original.replace(actual, new_string)
            if replace_all
            else original.replace(actual, new_string, 1)
        )
        shrinkage = _check_shrinkage(original, updated, len(actual), len(new_string))
        if shrinkage:
            return Result(text=shrinkage, is_error=True), None

        first_range = edit_ranges[0]
        new_line_count = _replacement_line_count(first_range, actual, new_string)
        snippet_end = first_range[0] + max(new_line_count, 1) - 1
        snippet = _snippet_around(updated, first_range[0], snippet_end)
        snippet_range = _snippet_range(updated, first_range[0], snippet_end)
        read_ranges = self._read_ranges_after_edit(
            read_state,
            edits=(
                (
                    first_range[0],
                    first_range[1],
                    new_line_count,
                ),
            )
            if len(edit_ranges) == 1
            else (),
            snippet_ranges=(snippet_range,) if snippet_range is not None else (),
        )
        return (
            Result(text=f"Updated {path!r}:\n{snippet}", read_ranges=read_ranges),
            updated.encode("utf-8"),
        )

    def _plan_line_range_replace(
        self,
        path: str,
        original: str,
        start: int,
        end: int,
        new_string: str,
        read_state: FileReadState | None,
    ) -> tuple[Result, bytes | None]:
        lines = original.splitlines(keepends=True)
        total = len(lines)
        if start < 1 or end < start or start > total:
            return (
                Result(
                    text=(
                        f"Invalid line range [{start}, {end}] for {path!r} "
                        f"({total} lines). Lines are 1-based."
                    ),
                    is_error=True,
                ),
                None,
            )
        end = min(end, total)
        coverage_error = self._check_edit_read_coverage(path, read_state, ((start, end),))
        if coverage_error is not None:
            return coverage_error, None
        before = lines[: start - 1]
        after = lines[end:]
        if new_string and not new_string.endswith("\n"):
            new_string += "\n"
        updated = "".join(before) + new_string + "".join(after)
        replaced_len = sum(len(ln) for ln in lines[start - 1 : end])
        shrinkage = _check_shrinkage(original, updated, replaced_len, len(new_string))
        if shrinkage:
            return Result(text=shrinkage, is_error=True), None

        new_line_count = _line_count(new_string)
        snippet_end = start + max(new_line_count, 1) - 1
        snippet = _snippet_around(updated, start, snippet_end)
        snippet_range = _snippet_range(updated, start, snippet_end)
        read_ranges = self._read_ranges_after_edit(
            read_state,
            edits=((start, end, new_line_count),),
            snippet_ranges=(snippet_range,) if snippet_range is not None else (),
        )
        return (
            Result(
                text=f"Replaced lines {start}-{end} in {path!r}:\n{snippet}",
                read_ranges=read_ranges,
            ),
            updated.encode("utf-8"),
        )
