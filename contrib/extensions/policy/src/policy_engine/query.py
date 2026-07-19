"""Policy engine query primitives — the runtime query API."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from typing import TYPE_CHECKING

from .types import (
    EMPTY,
    EffectRecord,
    FileStateEntry,
    ToolLogEntry,
    WhereClause,
    _EmptyType,
    match_row,
)

if TYPE_CHECKING:
    from collections import deque

    from .state import FileState


# ---------------------------------------------------------------------------
# TableQuery — bound to a deque-based state table (tool_log)
# ---------------------------------------------------------------------------


class TableQuery:
    """Query interface bound to a tool_log deque."""

    def __init__(
        self,
        get_entries: Callable[[], deque[ToolLogEntry]],
        cross_session_fn: Callable[..., int] | None = None,
    ) -> None:
        self._get_entries = get_entries
        self._cross_session_fn = cross_session_fn

    def _window(
        self, last: int | None = None, since: int | None = None
    ) -> list[ToolLogEntry]:
        entries = self._get_entries()
        if last is not None:
            start = max(0, len(entries) - last)
            return list(entries)[start:]
        if since is not None:
            return [e for e in entries if e.turn >= since]
        return list(entries)

    def count(
        self,
        where: WhereClause | None = None,
        last: int | None = None,
        since: int | None = None,
        scope: str | None = None,
        ttl: str | None = None,
    ) -> int:
        if scope == "user" and self._cross_session_fn:
            ttl_days = _parse_ttl(ttl) if ttl else 14
            fp = where.get("error_fingerprint") if where else None
            return self._cross_session_fn(
                error_fingerprint=fp, ttl_days=ttl_days
            )
        window = self._window(last=last, since=since)
        if not where:
            return len(window)
        return sum(1 for row in window if match_row(row, where))

    def distinct(
        self,
        field: str,
        last: int | None = None,
        since: int | None = None,
        where: WhereClause | None = None,
    ) -> int:
        window = self._window(last=last, since=since)
        if where:
            window = [r for r in window if match_row(r, where)]
        values: set[object] = set()
        for row in window:
            v = row.get(field)
            if v is not None:
                values.add(v)
        return len(values)

    def exists(
        self,
        where: WhereClause | None = None,
        last: int | None = None,
        since: int | None = None,
    ) -> bool:
        window = self._window(last=last, since=since)
        if not where:
            return len(window) > 0
        return any(match_row(row, where) for row in window)

    def last(self, where: WhereClause | None = None) -> ToolLogEntry | _EmptyType:
        entries = self._get_entries()
        for row in reversed(entries):
            if where is None or match_row(row, where):
                return row
        return EMPTY


# ---------------------------------------------------------------------------
# FileStateQuery — bound to the file_state entity map
# ---------------------------------------------------------------------------


class FileStateQuery:
    """Query interface for the file_state table."""

    def __init__(self, file_state: FileState) -> None:
        self._fs = file_state

    def get(self, path: str) -> FileStateEntry | _EmptyType:
        result = self._fs.get(path)
        return result if result is not None else EMPTY


# ---------------------------------------------------------------------------
# EffectLogQuery — queries over effect_log records
# ---------------------------------------------------------------------------


class EffectLogQuery:
    """Query interface for the effect_log (rule firing history)."""

    def __init__(self, get_entries: Callable[[], deque[EffectRecord]]) -> None:
        self._get_entries = get_entries

    def _window(
        self, last: int | None = None, since: int | None = None
    ) -> list[EffectRecord]:
        entries = self._get_entries()
        if last is not None:
            start = max(0, len(entries) - last)
            return list(entries)[start:]
        if since is not None:
            return [e for e in entries if e.turn >= since]
        return list(entries)

    def count(
        self,
        where: WhereClause | None = None,
        last: int | None = None,
        since: int | None = None,
    ) -> int:
        window = self._window(last=last, since=since)
        if not where:
            return len(window)
        return sum(1 for rec in window if match_row(rec, where))

    def exists(
        self,
        where: WhereClause | None = None,
        last: int | None = None,
        since: int | None = None,
    ) -> bool:
        window = self._window(last=last, since=since)
        if not where:
            return len(window) > 0
        return any(match_row(rec, where) for rec in window)

    def last(self, where: WhereClause | None = None) -> EffectRecord | _EmptyType:
        entries = self._get_entries()
        for rec in reversed(entries):
            if where is None or match_row(rec, where):
                return rec
        return EMPTY


# ---------------------------------------------------------------------------
# Standalone query primitives
# ---------------------------------------------------------------------------


def streak(
    get_entries: Callable[[], deque[ToolLogEntry]],
    where: WhereClause,
    last: int | None = None,
) -> int:
    """Count consecutive matching events from most recent backward."""
    entries = get_entries()
    if last is not None:
        start = max(0, len(entries) - last)
        window = list(entries)[start:]
    else:
        window = list(entries)

    count = 0
    for row in reversed(window):
        if match_row(row, where):
            count += 1
        else:
            break
    return count


def trend(
    get_entries: Callable[[], deque[ToolLogEntry]],
    field: str,
    where: WhereClause | None = None,
    last: int | None = None,
) -> str:
    """Linear slope classification: 'increasing' | 'decreasing' | 'stable'."""
    entries = get_entries()
    if last is not None:
        start = max(0, len(entries) - last)
        window = list(entries)[start:]
    else:
        window = list(entries)

    if where:
        window = [r for r in window if match_row(r, where)]

    raw_values = [r.get(field, 0) for r in window if isinstance(r.get(field), (int, float))]
    values: list[float] = [float(v) for v in raw_values if v is not None]
    if len(values) < 3:
        return "stable"

    n = len(values)
    x_mean = (n - 1) / 2
    y_mean = sum(values) / n
    num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    den = sum((i - x_mean) ** 2 for i in range(n))
    if den == 0:
        return "stable"
    slope = num / den
    threshold = y_mean * 0.1 if y_mean != 0 else 0.5
    if slope > threshold:
        return "increasing"
    if slope < -threshold:
        return "decreasing"
    return "stable"


def ratio(count_a: int, count_b: int) -> float:
    """Safe division: 0 if denominator is 0."""
    if count_b == 0:
        return 0.0
    return count_a / count_b


def sequence(
    get_entries: Callable[[], deque[ToolLogEntry]],
    steps: list[dict[str, object]],
    last: int | None = None,
) -> bool:
    """Ordered pattern match over event sequence. Linear scan."""
    entries = get_entries()
    if last is not None:
        start = max(0, len(entries) - last)
        window = list(entries)[start:]
    else:
        window = list(entries)

    step_idx = 0
    last_match_pos = -1

    for i, row in enumerate(window):
        if step_idx >= len(steps):
            break

        step_spec = steps[step_idx]
        where: WhereClause = step_spec.get("where", {})  # type: ignore[assignment]
        gap_max = step_spec.get("gap_max")
        absent: WhereClause | None = step_spec.get("absent")  # type: ignore[assignment]

        if match_row(row, where):
            if step_idx > 0 and gap_max is not None:
                if (i - last_match_pos - 1) > int(str(gap_max)):
                    return False
            if step_idx > 0 and absent is not None:
                for j in range(last_match_pos + 1, i):
                    if match_row(window[j], absent):
                        return False
            last_match_pos = i
            step_idx += 1

    return step_idx >= len(steps)


def group(
    get_entries: Callable[[], deque[ToolLogEntry]],
    field: str,
    where: WhereClause | None = None,
    last: int | None = None,
    top: int = 10,
) -> list[tuple[str | int | None, int]]:
    """Group-by with top-K results."""
    entries = get_entries()
    if last is not None:
        start = max(0, len(entries) - last)
        window = list(entries)[start:]
    else:
        window = list(entries)

    if where:
        window = [r for r in window if match_row(r, where)]

    counter: Counter[str | int | None] = Counter()
    for row in window:
        val = row.get(field)
        counter[val] += 1

    return counter.most_common(top)


def diff(set_a: set[str] | list[str], set_b: set[str] | list[str]) -> set[str]:
    """Set difference."""
    return set(set_a) - set(set_b)


def lookup(
    tables: dict[str, object],
    table: str,
    key: str | int,
    field: str,
) -> object | _EmptyType:
    """Cross-table point lookup by key."""
    tbl = tables.get(table)
    if tbl is None:
        return EMPTY
    try:
        row = tbl.get(key)  # type: ignore[attr-defined]
    except (TypeError, AttributeError):
        return EMPTY
    if row is None:
        return EMPTY
    if isinstance(row, dict):
        return row.get(field, EMPTY)
    return row.__dict__.get(field, EMPTY)


def _parse_ttl(ttl: str) -> int:
    """Parse TTL string like '14d' into days."""
    ttl = ttl.strip().lower()
    if ttl.endswith("d"):
        try:
            return int(ttl[:-1])
        except ValueError:
            return 14
    try:
        return int(ttl)
    except ValueError:
        return 14
