# code-health: ignore-file[AM025] -- source-region metrics normalize untyped tool results
"""Active source-region coverage used by IFG policy queries."""

from __future__ import annotations

import posixpath
import re
from collections import Counter, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .ifg_interventions import IfgInterventionState, InterventionQuery
from .ifg_investigation import IfgInvestigationState, InvestigationQuery
from .types import ToolArgs


_LINE_NUMBER_RE = re.compile(r"^\s*(?P<line>\d+)\t(?P<text>.*)$")
_READ_RANGE_RE = re.compile(
    r"showing lines (?P<start>\d+)-(?P<end>\d+) of (?P<total>\d+)",
    re.IGNORECASE,
)
_TOTAL_LINES_RE = re.compile(r"(?P<total>\d+) lines total", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class ParsedSourceRegion:
    """Source text and file-relative lines recovered from a tool result."""

    content_text: str
    line_range: Mapping[str, object]
    numbered: bool


@dataclass(frozen=True, slots=True)
class RegionReadEntry:
    """Overlap metrics for one successful structured read."""

    sequence: int
    turn: int
    path: str
    start_line: int
    end_line: int
    read_lines: int
    overlap_lines: int
    novel_lines: int
    overlap_ratio: float


def parse_source_region(text: str | None) -> ParsedSourceRegion | None:
    """Parse line-numbered tool output or its range header."""
    if not text:
        return None

    numbered: list[tuple[int, str]] = []
    for line in text.splitlines():
        match = _LINE_NUMBER_RE.match(line)
        if match is not None:
            numbered.append((int(match.group("line")), match.group("text")))
    if numbered:
        line_numbers = [line_no for line_no, _content in numbered]
        return ParsedSourceRegion(
            content_text="\n".join(content for _line_no, content in numbered),
            line_range={
                "start_line": min(line_numbers),
                "end_line": max(line_numbers),
                "matched_lines": line_numbers,
                "partial": True,
            },
            numbered=True,
        )

    first_line = text.splitlines()[0] if text.splitlines() else ""
    showing = _READ_RANGE_RE.search(first_line)
    if showing:
        line_range: Mapping[str, object] = {
            "start_line": int(showing.group("start")),
            "end_line": int(showing.group("end")),
            "total_lines": int(showing.group("total")),
            "partial": True,
        }
    else:
        total = _TOTAL_LINES_RE.search(first_line)
        if total:
            total_lines = int(total.group("total"))
            line_range = {
                "start_line": 1,
                "end_line": total_lines,
                "total_lines": total_lines,
                "partial": False,
            }
        else:
            line_range = {}
    return ParsedSourceRegion(
        content_text=text,
        line_range=line_range,
        numbered=False,
    )


class IfgRegionState:
    """Tracks active read coverage and invalidates it on file mutation."""

    def __init__(self, max_reads: int = 500) -> None:
        self._coverage: dict[str, list[tuple[int, int]]] = {}
        self._content_hashes: dict[str, str] = {}
        self._reads: deque[RegionReadEntry] = deque(maxlen=max_reads)
        self._sequence = 0

    def entries(self) -> deque[RegionReadEntry]:
        return self._reads

    def record(
        self,
        tool_name: str,
        args: ToolArgs,
        result: Mapping[str, object] | None,
        turn: int,
    ) -> None:
        if tool_name not in {"read", "edit", "write"} or not result:
            return
        if result.get("error"):
            return

        raw_path = args.get("path") or args.get("file_path")
        if not isinstance(raw_path, str) or not raw_path:
            return
        path = posixpath.normpath(raw_path)
        content_hash = _string_value(result.get("content_hash"))

        if tool_name == "read":
            self._record_read(path, result, turn, content_hash)
        elif tool_name == "edit":
            self._record_edit(path, args, result, content_hash)
        else:
            self._coverage.pop(path, None)
            self._update_hash(path, content_hash)

    def _record_read(
        self,
        path: str,
        result: Mapping[str, object],
        turn: int,
        content_hash: str | None,
    ) -> None:
        parsed = parse_source_region(_string_value(result.get("text")))
        if parsed is None:
            return
        intervals = _line_intervals(parsed.line_range)
        if not intervals:
            return

        previous_hash = self._content_hashes.get(path)
        if content_hash and previous_hash and content_hash != previous_hash:
            self._coverage.pop(path, None)

        coverage = self._coverage.get(path, [])
        read_lines = _interval_size(intervals)
        overlap_lines = _intersection_size(intervals, coverage)
        self._sequence += 1
        self._reads.append(
            RegionReadEntry(
                sequence=self._sequence,
                turn=turn,
                path=path,
                start_line=intervals[0][0],
                end_line=intervals[-1][1],
                read_lines=read_lines,
                overlap_lines=overlap_lines,
                novel_lines=read_lines - overlap_lines,
                overlap_ratio=overlap_lines / read_lines,
            )
        )
        self._coverage[path] = _merge_intervals([*coverage, *intervals])
        self._update_hash(path, content_hash)

    def _record_edit(
        self,
        path: str,
        args: ToolArgs,
        result: Mapping[str, object],
        content_hash: str | None,
    ) -> None:
        parsed = parse_source_region(_string_value(result.get("text")))
        intervals = _line_intervals(parsed.line_range) if parsed else []
        if not intervals:
            self._coverage.pop(path, None)
            self._update_hash(path, content_hash)
            return

        start = intervals[0][0]
        end = intervals[-1][1]
        old_string = args.get("old_string")
        new_string = args.get("new_string")
        delta = (
            new_string.count("\n") - old_string.count("\n")
            if isinstance(old_string, str) and isinstance(new_string, str)
            else 0
        )
        self._coverage[path] = _invalidate_and_shift(
            self._coverage.get(path, []),
            start=start,
            end=end,
            delta=delta,
        )
        self._update_hash(path, content_hash)

    def _update_hash(self, path: str, content_hash: str | None) -> None:
        if content_hash:
            self._content_hashes[path] = content_hash


class RegionReadQuery:
    """DSL query methods over regional read observations."""

    def __init__(self, state: IfgRegionState) -> None:
        self._state = state

    def count(self, *, last: int | None = None, path: str | None = None) -> int:
        return len(self._window(last=last, path=path))

    def overlap_count(
        self,
        *,
        last: int | None = None,
        min_ratio: float = 0.8,
        min_lines: int = 10,
        path: str | None = None,
    ) -> int:
        return sum(
            1
            for entry in self._window(last=last, path=path)
            if entry.overlap_ratio >= min_ratio and entry.overlap_lines >= min_lines
        )

    def overlap_lines(self, *, last: int | None = None, path: str | None = None) -> int:
        return sum(entry.overlap_lines for entry in self._window(last=last, path=path))

    def read_lines(self, *, last: int | None = None, path: str | None = None) -> int:
        return sum(entry.read_lines for entry in self._window(last=last, path=path))

    def overlap_ratio(
        self, *, last: int | None = None, path: str | None = None
    ) -> float:
        window = self._window(last=last, path=path)
        read_lines = sum(entry.read_lines for entry in window)
        if read_lines == 0:
            return 0.0
        return sum(entry.overlap_lines for entry in window) / read_lines

    def summary(
        self,
        *,
        last: int | None = None,
        min_ratio: float = 0.8,
        min_lines: int = 10,
    ) -> dict[str, object]:
        window = self._window(last=last)
        repeated = [
            entry
            for entry in window
            if entry.overlap_ratio >= min_ratio and entry.overlap_lines >= min_lines
        ]
        hot_paths = Counter(entry.path for entry in repeated).most_common(5)
        return {
            "reads": len(window),
            "repeated_reads": len(repeated),
            "overlap_lines": sum(entry.overlap_lines for entry in window),
            "read_lines": sum(entry.read_lines for entry in window),
            "overlap_ratio": round(self.overlap_ratio(last=last), 3),
            "hot_paths": [f"{path} ({count})" for path, count in hot_paths],
        }

    def _window(
        self, *, last: int | None, path: str | None = None
    ) -> list[RegionReadEntry]:
        entries = list(self._state.entries())
        if last is not None:
            entries = [] if last <= 0 else entries[-last:]
        if path is not None:
            normalized = posixpath.normpath(path)
            entries = [entry for entry in entries if entry.path == normalized]
        return entries


class IfgQuery:
    """Root object exposed as ``ifg`` in policy expressions."""

    __slots__ = ("intervention", "investigation", "region_reads")

    def __init__(
        self,
        regions: IfgRegionState,
        interventions: IfgInterventionState,
        investigation: IfgInvestigationState | None = None,
    ) -> None:
        self.region_reads = RegionReadQuery(regions)
        self.intervention = InterventionQuery(interventions)
        self.investigation = InvestigationQuery(
            investigation or IfgInvestigationState()
        )


def _string_value(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _line_intervals(line_range: Mapping[str, object]) -> list[tuple[int, int]]:
    matched = line_range.get("matched_lines")
    if isinstance(matched, Sequence) and not isinstance(matched, (str, bytes)):
        lines = sorted(
            {line for value in matched if (line := _positive_int(value)) is not None}
        )
        if lines:
            intervals: list[tuple[int, int]] = []
            start = previous = lines[0]
            for line in lines[1:]:
                if line == previous + 1:
                    previous = line
                    continue
                intervals.append((start, previous))
                start = previous = line
            intervals.append((start, previous))
            return intervals

    range_start = _positive_int(line_range.get("start_line"))
    range_end = _positive_int(line_range.get("end_line"))
    if range_start is None or range_end is None or range_end < range_start:
        return []
    return [(range_start, range_end)]


def _positive_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = int(str(value))
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _merge_intervals(intervals: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1] + 1:
            merged.append((start, end))
        else:
            previous_start, previous_end = merged[-1]
            merged[-1] = (previous_start, max(previous_end, end))
    return merged


def _interval_size(intervals: Sequence[tuple[int, int]]) -> int:
    return sum(end - start + 1 for start, end in intervals)


def _intersection_size(
    left: Sequence[tuple[int, int]], right: Sequence[tuple[int, int]]
) -> int:
    total = 0
    left_index = right_index = 0
    while left_index < len(left) and right_index < len(right):
        left_start, left_end = left[left_index]
        right_start, right_end = right[right_index]
        total += max(0, min(left_end, right_end) - max(left_start, right_start) + 1)
        if left_end < right_end:
            left_index += 1
        else:
            right_index += 1
    return total


def _invalidate_and_shift(
    coverage: Sequence[tuple[int, int]],
    *,
    start: int,
    end: int,
    delta: int,
) -> list[tuple[int, int]]:
    updated: list[tuple[int, int]] = []
    for current_start, current_end in coverage:
        if current_end < start:
            updated.append((current_start, current_end))
            continue
        if current_start > end:
            updated.append((max(1, current_start + delta), max(1, current_end + delta)))
            continue
        if current_start < start:
            updated.append((current_start, start - 1))
        if current_end > end:
            shifted_start = max(1, end + 1 + delta)
            shifted_end = max(1, current_end + delta)
            if shifted_end >= shifted_start:
                updated.append((shifted_start, shifted_end))
    return _merge_intervals(updated)
