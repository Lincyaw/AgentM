"""Policy engine shared types — dataclasses, sentinels, typed records."""

from __future__ import annotations

import types as pytypes
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Literal


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

ToolArgs = dict[str, str | int | float | bool | list | None]
WhereClause = dict[str, str | int | None]
EffectType = Literal["notify", "block", "escalate", "abort"]
RuleMode = Literal["enforce", "observe"]


# ---------------------------------------------------------------------------
# EMPTY sentinel — null-safe attribute access
# ---------------------------------------------------------------------------


class _EmptyType:
    """Sentinel for query results that found nothing.

    Falsy in boolean context. Attribute access returns safe defaults.
    """

    __slots__ = ()

    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return "EMPTY"

    def __getattr__(self, name: str) -> int:
        return -1

    def __getitem__(self, key: str | int) -> int:
        return -1

    def __contains__(self, item: object) -> bool:
        return False

    def __iter__(self) -> Iterator:
        return iter(())

    def __len__(self) -> int:
        return 0


EMPTY = _EmptyType()


# ---------------------------------------------------------------------------
# Evidence model (entity_registry)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Evidence:
    type: str
    turn: int
    detail: str


@dataclass(slots=True)
class EvidenceList:
    """Evidence records for one entity. Supports rule queries."""

    records: list[Evidence] = field(default_factory=list)

    def has(self, *, type: str) -> bool:
        return any(e.type == type for e in self.records)

    def strongest(self) -> Evidence | _EmptyType:
        priority = ("tool_success", "structural", "user_provided",
                    "tool_failure", "lexical_match", "dict_recall")
        for p in priority:
            for e in reversed(self.records):
                if e.type == p:
                    return e
        return EMPTY if not self.records else self.records[-1]

    def __bool__(self) -> bool:
        return len(self.records) > 0

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[Evidence]:
        return iter(self.records)

    def format_for_diagnostic(self) -> str:
        if not self.records:
            return "(no evidence)"
        lines = ["| Type | Turn | Detail |", "|---|---|---|"]
        for e in self.records:
            lines.append(f"| {e.type} | {e.turn} | {e.detail} |")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entity record
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class EntityRecord:
    entity: str
    entity_type: str
    evidence: EvidenceList = field(default_factory=EvidenceList)
    first_seen_turn: int = 0
    last_seen_turn: int = 0
    occurrence_count: int = 0

    _MAX_EVIDENCE_PER_TYPE: int = 2

    def add_evidence(self, ev: Evidence) -> None:
        self.last_seen_turn = ev.turn
        self.occurrence_count += 1
        existing = [e for e in self.evidence.records if e.type == ev.type]
        if len(existing) < self._MAX_EVIDENCE_PER_TYPE:
            self.evidence.records.append(ev)
        else:
            self.evidence.records.remove(existing[-1])
            self.evidence.records.append(ev)


# ---------------------------------------------------------------------------
# Tool log entry (typed record replacing generic Row for tool_log)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ToolLogEntry:
    turn: int = 0
    tool: str = ""
    args_hash: str = ""
    path: str | None = None
    cmd: str | None = None
    exit_code: int | None = None
    error: str | None = None
    error_fingerprint: str | None = None
    error_category: str | None = None
    duration_ms: int = 0
    result_length: int = 0
    is_repeat: bool = False
    repeat_count: int = 0

    def get(self, field: str, default: int | str | None = None) -> str | int | bool | None:
        try:
            return object.__getattribute__(self, field)
        except AttributeError:
            return default


# ---------------------------------------------------------------------------
# File state entry
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FileStateEntry:
    path: str = ""
    content_hash: str | None = None
    first_read_turn: int | None = None
    last_read_turn: int | None = None
    last_write_turn: int | None = None
    read_count: int = 0
    write_count: int = 0
    reverts_to_prior_hash: bool = False
    _content_hashes: set[str] = field(default_factory=set)

    def get(self, field: str, default: int | str | None = None) -> str | int | bool | None:
        try:
            return object.__getattribute__(self, field)
        except AttributeError:
            return default


# ---------------------------------------------------------------------------
# Effect log record
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EffectRecord:
    rule_id: str
    mode: str
    channel: str
    effect: str
    reason: str
    turn: int

    def get(self, field: str, default: str | int | None = None) -> str | int | None:
        return self.__dict__.get(field, default)


# ---------------------------------------------------------------------------
# Guard (fast-reject filter built from `match` clause)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Guard:
    """Compiled match clause — tool name + arg patterns."""

    tool_names: frozenset[str] | None = None
    field_patterns: tuple[tuple[str, str], ...] = ()

    def reject(self, tool_name: str, args: ToolArgs) -> bool:
        if self.tool_names is not None and tool_name not in self.tool_names:
            return True
        for path, pattern in self.field_patterns:
            value = _resolve_dotted(args, path)
            if not _match_pattern(str(value) if value is not None else "", pattern):
                return True
        return False


# ---------------------------------------------------------------------------
# Effect spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EffectSpec:
    effect: EffectType
    reason_template: str = ""


# ---------------------------------------------------------------------------
# Rule instance (compiled form)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RuleInstance:
    rule_id: str
    layer: int  # 0-3
    mode: RuleMode
    channel: str
    guard: Guard
    predicate: pytypes.CodeType | None = None
    effect: EffectSpec = field(default_factory=lambda: EffectSpec(effect="notify"))
    escalate_context_expr: pytypes.CodeType | None = None
    cooldown_turns: int = 0
    last_fired_turn: int = -1


# ---------------------------------------------------------------------------
# Pattern matching helpers (where-clause language)
# ---------------------------------------------------------------------------


def _match_pattern(value: str, pattern: str) -> bool:
    """Match a value against the where-clause pattern language."""
    if pattern == "":
        return True
    if pattern == "null":
        return value == "" or value == "None"
    if pattern.startswith("!="):
        return value != pattern[2:]
    if "|" in pattern and "*" not in pattern:
        return value in pattern.split("|")
    if pattern.startswith("*") and pattern.endswith("*"):
        return pattern[1:-1] in value
    if pattern.endswith("*"):
        return value.startswith(pattern[:-1])
    if pattern.startswith("*"):
        return value.endswith(pattern[1:])
    return value == pattern


def match_row(row: ToolLogEntry | EffectRecord | FileStateEntry, where: WhereClause) -> bool:
    """Check if a typed record matches ALL where-clause conditions."""
    for field_path, pattern in where.items():
        fp = str(field_path)
        value = row.get(fp) if "." not in fp else _resolve_nested_attr(row, fp)
        if isinstance(pattern, int):
            if value != pattern:
                return False
        elif isinstance(pattern, str):
            val_str = str(value) if value is not None else ""
            if not _match_pattern(val_str, pattern):
                return False
        elif pattern is None:
            if value is not None:
                return False
        else:
            if value != pattern:
                return False
    return True


def _resolve_nested_attr(obj: object, path: str) -> object | None:
    """Resolve a dotted path like 'args.cmd' against a typed record."""
    parts = path.split(".")
    current: object = obj
    for part in parts:
        if isinstance(current, dict):
            current = current.get(part)
        else:
            try:
                current = object.__getattribute__(current, part)
            except AttributeError:
                return None
        if current is None:
            return None
    return current


def _resolve_dotted(obj: object, path: str) -> object | None:
    """Resolve a dotted path against a dict (for Guard field patterns)."""
    parts = path.split(".")
    current: object = obj
    for part in parts:
        if isinstance(current, dict):
            current = current.get(part)
        else:
            try:
                current = object.__getattribute__(current, part)
            except AttributeError:
                return None
        if current is None:
            return None
    return current
