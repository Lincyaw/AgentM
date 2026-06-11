"""Graph data model, op log fold, and phase merge — pure computation, no I/O."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from llmharness.schema import Edge, EdgeKind, Event, EventKind, ExternalRef, Phase

# ---------------------------------------------------------------------------
# Graph ops
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NodeUpsert:
    """Insert or replace an event node by id."""

    id: int
    kind: str
    summary: str
    source_turns: tuple[int, ...]
    external_refs: tuple[ExternalRef, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": "node_upsert",
            "id": self.id,
            "kind": self.kind,
            "summary": self.summary,
            "source_turns": list(self.source_turns),
            "external_refs": [r.to_dict() for r in self.external_refs],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NodeUpsert:
        return cls(
            id=int(data["id"]),
            kind=str(data["kind"]),
            summary=str(data.get("summary", "")),
            source_turns=tuple(int(t) for t in (data.get("source_turns") or [])),
            external_refs=tuple(
                ExternalRef.from_dict(r) for r in (data.get("external_refs") or [])
            ),
        )


@dataclass(frozen=True)
class NodeDelete:
    """Delete an event node by id; cascades to incident edges at fold time."""

    id: int

    def to_dict(self) -> dict[str, Any]:
        return {"op": "node_delete", "id": self.id}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NodeDelete:
        return cls(id=int(data["id"]))


@dataclass(frozen=True)
class EdgeUpsert:
    """Insert or replace one witness-bearing edge keyed by (src, dst, kind)."""

    src: int
    dst: int
    kind: str
    reason: str
    cited_entities: tuple[str, ...]
    cited_quote: str
    src_turns: tuple[int, ...]
    dst_turns: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": "edge_upsert",
            "src": self.src,
            "dst": self.dst,
            "kind": self.kind,
            "reason": self.reason,
            "cited_entities": list(self.cited_entities),
            "cited_quote": self.cited_quote,
            "src_turns": list(self.src_turns),
            "dst_turns": list(self.dst_turns),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EdgeUpsert:
        return cls(
            src=int(data["src"]),
            dst=int(data["dst"]),
            kind=str(data["kind"]),
            reason=str(data.get("reason", "")),
            cited_entities=tuple(str(e) for e in (data.get("cited_entities") or [])),
            cited_quote=str(data.get("cited_quote", "")),
            src_turns=tuple(int(t) for t in (data.get("src_turns") or [])),
            dst_turns=tuple(int(t) for t in (data.get("dst_turns") or [])),
        )


@dataclass(frozen=True)
class EdgeDelete:
    """Delete an edge by (src, dst, kind); kind is mandatory for unambiguous selection."""

    src: int
    dst: int
    kind: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": "edge_delete",
            "src": self.src,
            "dst": self.dst,
            "kind": self.kind,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EdgeDelete:
        return cls(
            src=int(data["src"]),
            dst=int(data["dst"]),
            kind=str(data["kind"]),
        )


GraphOp = NodeUpsert | NodeDelete | EdgeUpsert | EdgeDelete

_OP_TABLE: dict[str, type] = {
    "node_upsert": NodeUpsert,
    "node_delete": NodeDelete,
    "edge_upsert": EdgeUpsert,
    "edge_delete": EdgeDelete,
}


def parse_op(data: dict[str, Any]) -> GraphOp:
    """Dispatch on the ``"op"`` discriminator to build the right op."""
    op = data.get("op")
    if not isinstance(op, str):
        raise ValueError(f"graph op payload missing 'op' discriminator: {data!r}")
    cls = _OP_TABLE.get(op)
    if cls is None:
        raise ValueError(f"unknown graph op discriminator {op!r}")
    return cls.from_dict(data)  # type: ignore[attr-defined,no-any-return]


# ---------------------------------------------------------------------------
# Graph fold
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Graph:
    """Folded view of an op log: nodes keyed by id, edges by (src, dst, kind)."""

    nodes: dict[int, Event] = field(default_factory=dict)
    edges: dict[tuple[int, int, str], Edge] = field(default_factory=dict)

    def nodes_list(self) -> list[Event]:
        """Events in insertion order."""
        return list(self.nodes.values())

    def edges_list(self) -> list[Edge]:
        """Edges in insertion order."""
        return list(self.edges.values())


def fold_graph(ops: Iterable[GraphOp]) -> Graph:
    """Fold an op iterable into a Graph. Pure, no I/O."""
    nodes: dict[int, Event] = {}
    edges: dict[tuple[int, int, str], Edge] = {}
    for op in ops:
        if isinstance(op, NodeUpsert):
            nodes[op.id] = Event(
                id=op.id,
                kind=EventKind(op.kind),
                summary=op.summary,
                source_turns=list(op.source_turns),
                external_refs=op.external_refs,
            )
        elif isinstance(op, NodeDelete):
            nodes.pop(op.id, None)
            for key in [k for k in edges if k[0] == op.id or k[1] == op.id]:
                del edges[key]
        elif isinstance(op, EdgeUpsert):
            key = (op.src, op.dst, op.kind)
            edges[key] = Edge(
                src=op.src,
                dst=op.dst,
                kind=EdgeKind(op.kind),
                reason=op.reason,
                src_turns=op.src_turns,
                dst_turns=op.dst_turns,
                cited_entities=op.cited_entities,
                cited_quote=op.cited_quote,
            )
        elif isinstance(op, EdgeDelete):
            edges.pop((op.src, op.dst, op.kind), None)
        else:  # pragma: no cover
            raise TypeError(f"unknown graph op type: {type(op).__name__}")
    return Graph(nodes=nodes, edges=edges)


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

    _flush_run()
    return phases
