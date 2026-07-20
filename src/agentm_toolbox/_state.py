"""Session-local read state for read-before-mutation coordination."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

LineRange = tuple[int, int]


@dataclass(slots=True)
class FileReadState:
    total_lines: int
    is_partial: bool
    mtime_ns: int = 0
    content_hash: str = ""
    ranges: tuple[LineRange, ...] = ()

    def covers(self, start_line: int, end_line: int) -> bool:
        if end_line < start_line:
            return True
        if start_line < 1:
            return False
        if not self.ranges:
            return not self.is_partial and end_line <= self.total_lines
        return _ranges_cover(self.ranges, start_line, end_line)


def content_hash_for(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class ReadStateStore:
    """In-memory read-state store owned by one file-tool session."""

    def __init__(self) -> None:
        self._states: dict[str, FileReadState] = {}

    def record(
        self,
        path: str,
        *,
        total_lines: int,
        is_partial: bool,
        mtime_ns: int = 0,
        content_hash: str = "",
        start_line: int | None = None,
        end_line: int | None = None,
        ranges: tuple[LineRange, ...] | None = None,
    ) -> None:
        normalized = os.path.normpath(path)
        read_ranges = _read_ranges(
            total_lines,
            is_partial=is_partial,
            start_line=start_line,
            end_line=end_line,
            ranges=ranges,
        )
        existing = self._states.get(normalized)
        if (
            existing is not None
            and content_hash
            and existing.content_hash == content_hash
            and existing.total_lines == total_lines
        ):
            read_ranges = _merge_ranges(existing.ranges + read_ranges)
        is_partial = not _ranges_cover_full_file(total_lines, read_ranges)
        self._states[normalized] = FileReadState(
            total_lines=total_lines,
            is_partial=is_partial,
            mtime_ns=mtime_ns,
            content_hash=content_hash,
            ranges=read_ranges,
        )

    def get(self, path: str) -> FileReadState | None:
        return self._states.get(os.path.normpath(path))


def _read_ranges(
    total_lines: int,
    *,
    is_partial: bool,
    start_line: int | None,
    end_line: int | None,
    ranges: tuple[LineRange, ...] | None,
) -> tuple[LineRange, ...]:
    if ranges is not None:
        return _normalize_ranges(total_lines, ranges)
    if start_line is not None and end_line is not None:
        return _normalize_ranges(total_lines, ((start_line, end_line),))
    if is_partial or total_lines <= 0:
        return ()
    return ((1, total_lines),)


def _normalize_ranges(
    total_lines: int, ranges: tuple[LineRange, ...]
) -> tuple[LineRange, ...]:
    if total_lines <= 0:
        return ()
    clipped: list[LineRange] = []
    for start, end in ranges:
        clipped_start = max(1, int(start))
        clipped_end = min(total_lines, int(end))
        if clipped_start <= clipped_end:
            clipped.append((clipped_start, clipped_end))
    return _merge_ranges(tuple(clipped))


def _merge_ranges(ranges: tuple[LineRange, ...]) -> tuple[LineRange, ...]:
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


def _ranges_cover(
    ranges: tuple[LineRange, ...], start_line: int, end_line: int
) -> bool:
    cursor = start_line
    for start, end in ranges:
        if end < cursor:
            continue
        if start > cursor:
            return False
        if end >= end_line:
            return True
        cursor = end + 1
    return False


def _ranges_cover_full_file(total_lines: int, ranges: tuple[LineRange, ...]) -> bool:
    if total_lines <= 0:
        return True
    return _ranges_cover(ranges, 1, total_lines)
