"""Context-index op log fold and phase merge.

The extractor maintains an append-only index edit log. Records are semantic
items extracted from trajectory turns; links are witness-bearing references
between records. This module is pure computation and performs no I/O.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from llmharness.schema import (
    CommitmentStatus,
    Edge,
    EdgeKind,
    EdgeRole,
    Event,
    EventKind,
    ExternalRef,
    Phase,
)

# ---------------------------------------------------------------------------
# Index ops
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecordUpsert:
    """Insert or replace an index record by id."""

    id: int
    kind: str
    summary: str
    source_turns: tuple[int, ...]
    external_refs: tuple[ExternalRef, ...] = field(default_factory=tuple)
    status: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "op": "record_upsert",
            "id": self.id,
            "kind": self.kind,
            "summary": self.summary,
            "source_turns": list(self.source_turns),
            "external_refs": [r.to_dict() for r in self.external_refs],
        }
        if self.status is not None:
            d["status"] = self.status
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RecordUpsert:
        return cls(
            id=int(data["id"]),
            kind=str(data["kind"]),
            summary=str(data.get("summary", "")),
            source_turns=tuple(int(t) for t in (data.get("source_turns") or [])),
            external_refs=tuple(
                ExternalRef.from_dict(r) for r in (data.get("external_refs") or [])
            ),
            status=data.get("status"),
        )


@dataclass(frozen=True)
class RecordDelete:
    """Delete an index record by id; cascades to incident links at fold time."""

    id: int

    def to_dict(self) -> dict[str, Any]:
        return {"op": "record_delete", "id": self.id}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RecordDelete:
        return cls(id=int(data["id"]))


@dataclass(frozen=True)
class LinkUpsert:
    """Insert or replace one witness-bearing link keyed by (src, dst, kind)."""

    src: int
    dst: int
    kind: str
    reason: str
    cited_entities: tuple[str, ...]
    cited_quote: str
    src_turns: tuple[int, ...]
    dst_turns: tuple[int, ...]
    role: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "op": "link_upsert",
            "src": self.src,
            "dst": self.dst,
            "kind": self.kind,
            "reason": self.reason,
            "cited_entities": list(self.cited_entities),
            "cited_quote": self.cited_quote,
            "src_turns": list(self.src_turns),
            "dst_turns": list(self.dst_turns),
        }
        if self.role is not None:
            d["role"] = self.role
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LinkUpsert:
        return cls(
            src=int(data["src"]),
            dst=int(data["dst"]),
            kind=str(data["kind"]),
            reason=str(data.get("reason", "")),
            cited_entities=tuple(str(e) for e in (data.get("cited_entities") or [])),
            cited_quote=str(data.get("cited_quote", "")),
            src_turns=tuple(int(t) for t in (data.get("src_turns") or [])),
            dst_turns=tuple(int(t) for t in (data.get("dst_turns") or [])),
            role=data.get("role"),
        )


@dataclass(frozen=True)
class LinkDelete:
    """Delete a link by (src, dst, kind)."""

    src: int
    dst: int
    kind: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": "link_delete",
            "src": self.src,
            "dst": self.dst,
            "kind": self.kind,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LinkDelete:
        return cls(
            src=int(data["src"]),
            dst=int(data["dst"]),
            kind=str(data["kind"]),
        )


IndexOp = RecordUpsert | RecordDelete | LinkUpsert | LinkDelete

_OP_TABLE: Final[dict[str, type[RecordUpsert | RecordDelete | LinkUpsert | LinkDelete]]] = {
    "record_upsert": RecordUpsert,
    "record_delete": RecordDelete,
    "link_upsert": LinkUpsert,
    "link_delete": LinkDelete,
}


def parse_op(data: dict[str, Any]) -> IndexOp:
    """Dispatch on the ``"op"`` discriminator to build the right op."""
    op = data.get("op")
    if not isinstance(op, str):
        raise ValueError(f"index op payload missing 'op' discriminator: {data!r}")
    cls = _OP_TABLE.get(op)
    if cls is None:
        raise ValueError(f"unknown index op discriminator {op!r}")
    return cls.from_dict(data)


# ---------------------------------------------------------------------------
# Index fold
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Index:
    """Folded view of an op log: records by id, links by (src, dst, kind)."""

    records: dict[int, Event] = field(default_factory=dict)
    links: dict[tuple[int, int, str], Edge] = field(default_factory=dict)

    def records_list(self) -> list[Event]:
        """Records in insertion order."""
        return list(self.records.values())

    def links_list(self) -> list[Edge]:
        """Links in insertion order."""
        return list(self.links.values())


def fold_index(ops: Iterable[IndexOp]) -> Index:
    """Fold an op iterable into an Index. Pure, no I/O."""
    records: dict[int, Event] = {}
    links: dict[tuple[int, int, str], Edge] = {}
    for op in ops:
        if isinstance(op, RecordUpsert):
            records[op.id] = Event(
                id=op.id,
                kind=EventKind(op.kind),
                summary=op.summary,
                source_turns=list(op.source_turns),
                external_refs=op.external_refs,
                status=CommitmentStatus(op.status) if op.status else None,
            )
        elif isinstance(op, RecordDelete):
            records.pop(op.id, None)
            for key in [k for k in links if k[0] == op.id or k[1] == op.id]:
                del links[key]
        elif isinstance(op, LinkUpsert):
            key = (op.src, op.dst, op.kind)
            links[key] = Edge(
                src=op.src,
                dst=op.dst,
                kind=EdgeKind(op.kind),
                reason=op.reason,
                src_turns=op.src_turns,
                dst_turns=op.dst_turns,
                cited_entities=op.cited_entities,
                cited_quote=op.cited_quote,
                role=EdgeRole(op.role) if op.role else None,
            )
        elif isinstance(op, LinkDelete):
            links.pop((op.src, op.dst, op.kind), None)
        else:  # pragma: no cover
            raise TypeError(f"unknown index op type: {type(op).__name__}")
    return Index(records=records, links=links)


# ---------------------------------------------------------------------------
# Phase merge
# ---------------------------------------------------------------------------

_BREAK_KINDS: frozenset[EventKind] = frozenset(
    {EventKind.TASK, EventKind.HYP, EventKind.DEC, EventKind.CONCL}
)
_RUN_KINDS: frozenset[EventKind] = frozenset({EventKind.ACT})
_MAX_RUN_SUMMARY_CHARS = 1200
_RUN_SUMMARY_SEPARATOR = " | "
_RUN_KIND_LABEL = "act_run"


def merge_to_phases(events: Sequence[Event]) -> list[Phase]:
    """Collapse consecutive act runs into phases."""
    phases: list[Phase] = []
    next_id = 1
    run_buffer: list[Event] = []

    def _flush_run() -> None:
        nonlocal next_id
        if not run_buffer:
            return
        if len(run_buffer) == 1:
            ev = run_buffer[0]
            phases.append(
                Phase(
                    id=next_id,
                    kind=ev.kind.value,
                    member_event_ids=(ev.id,),
                    source_turns=tuple(sorted(set(ev.source_turns))),
                    summary=ev.summary,
                )
            )
        else:
            ids = tuple(e.id for e in run_buffer)
            turns = tuple(sorted({t for e in run_buffer for t in e.source_turns}))
            joined = _RUN_SUMMARY_SEPARATOR.join(e.summary for e in run_buffer)
            if len(joined) > _MAX_RUN_SUMMARY_CHARS:
                joined = joined[: _MAX_RUN_SUMMARY_CHARS - 3] + "..."
            phases.append(
                Phase(
                    id=next_id,
                    kind=_RUN_KIND_LABEL,
                    member_event_ids=ids,
                    source_turns=turns,
                    summary=joined,
                )
            )
        next_id += 1
        run_buffer.clear()

    for ev in events:
        if ev.kind in _BREAK_KINDS:
            _flush_run()
            phases.append(
                Phase(
                    id=next_id,
                    kind=ev.kind.value,
                    member_event_ids=(ev.id,),
                    source_turns=tuple(sorted(set(ev.source_turns))),
                    summary=ev.summary,
                )
            )
            next_id += 1
        elif ev.kind in _RUN_KINDS:
            run_buffer.append(ev)
        else:  # pragma: no cover
            _flush_run()
            phases.append(
                Phase(
                    id=next_id,
                    kind=ev.kind.value,
                    member_event_ids=(ev.id,),
                    source_turns=tuple(sorted(set(ev.source_turns))),
                    summary=ev.summary,
                )
            )
            next_id += 1
    _flush_run()
    return phases


__all__ = [
    "Index",
    "IndexOp",
    "LinkDelete",
    "LinkUpsert",
    "RecordDelete",
    "RecordUpsert",
    "fold_index",
    "merge_to_phases",
    "parse_op",
]
