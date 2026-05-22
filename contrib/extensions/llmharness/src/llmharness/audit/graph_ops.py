"""Event-sourcing graph ops: the persistent extractor graph as an op log.

The audit graph survives across firings. Each firing's extractor child
emits a sequence of ops which the substrate folds onto the prior graph
to produce the current view. Given an initial graph (empty) plus the
ordered op log, any historical state is reproducible.

Four op kinds, all frozen and dict-serializable:

- :class:`NodeUpsert` — insert or replace an event node by id.
- :class:`NodeDelete` — remove an event node; cascades to incident edges.
- :class:`EdgeUpsert` — insert or replace an edge keyed by ``(src, dst, kind)``.
- :class:`EdgeDelete` — remove an edge by ``(src, dst, kind)``. ``kind`` is
  MANDATORY so the selector is unambiguous; per the original audit-graph
  contract, multiple edges may share the same ``(src, dst)`` pair across
  ``kind=data`` / ``kind=ref``.

Every dict carries an ``"op"`` discriminator field so :func:`parse_op`
can dispatch without ambiguity. The discriminator values are:
``"node_upsert"`` / ``"node_delete"`` / ``"edge_upsert"`` /
``"edge_delete"``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..schema import EdgeKind, EventKind, ExternalRef


@dataclass(frozen=True)
class NodeUpsert:
    """Insert or replace an event node by ``id``.

    ``kind`` is the :class:`~llmharness.schema.EventKind` value (str);
    validated by the caller before constructing the op. ``source_turns``
    is a tuple of absolute trajectory turn indices.

    ``external_refs`` carries cross-firing references from this event
    back into the cumulative graph (see :class:`ExternalRef`). It must
    survive the op-log/fold round-trip so the legacy
    ``AUDIT_EVENT``-to-op translation in ``_scan_branch`` does not
    drop them when reading prior-firing entries; the auditor and the
    next firing's ``recent_graph`` payload both depend on them being
    present on the folded :class:`Event`.
    """

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
    """Delete an event node by id. Cascades to incident edges at fold time."""

    id: int

    def to_dict(self) -> dict[str, Any]:
        return {"op": "node_delete", "id": self.id}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NodeDelete:
        return cls(id=int(data["id"]))


@dataclass(frozen=True)
class EdgeUpsert:
    """Insert or replace one witness-bearing edge keyed by ``(src, dst, kind)``.

    ``src_turns`` / ``dst_turns`` are populated by the op builder at the
    time the op is constructed (lifted from the endpoint nodes' source
    turns in the *folded* view). Keeping them on the op makes replay
    self-contained — the fold does not need to re-read node state to
    materialize an :class:`~llmharness.schema.Edge`.
    """

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
            cited_entities=tuple(
                str(e) for e in (data.get("cited_entities") or [])
            ),
            cited_quote=str(data.get("cited_quote", "")),
            src_turns=tuple(int(t) for t in (data.get("src_turns") or [])),
            dst_turns=tuple(int(t) for t in (data.get("dst_turns") or [])),
        )


@dataclass(frozen=True)
class EdgeDelete:
    """Delete an edge identified by ``(src, dst, kind)``.

    ``kind`` is mandatory because ``(src, dst)`` alone is not unique —
    a single pair may carry both a ``data`` and a ``ref`` edge. Forcing
    the selector to include ``kind`` rules out the ambiguity at the
    op-log level.
    """

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
    """Dispatch on the ``"op"`` discriminator to build the right op.

    Raises ``ValueError`` if the discriminator is missing or unknown —
    the caller is expected to validate upstream when reading durable
    entries, so an exception here marks a programmer error or a corrupt
    log entry, not a runtime branch.
    """
    op = data.get("op")
    if not isinstance(op, str):
        raise ValueError(f"graph op payload missing 'op' discriminator: {data!r}")
    cls = _OP_TABLE.get(op)
    if cls is None:
        raise ValueError(f"unknown graph op discriminator {op!r}")
    return cls.from_dict(data)  # type: ignore[attr-defined,no-any-return]


def validate_node_upsert_kind(kind: str) -> bool:
    """True iff ``kind`` is a valid :class:`~llmharness.schema.EventKind` value."""
    try:
        EventKind(kind)
    except ValueError:
        return False
    return True


def validate_edge_kind(kind: str) -> bool:
    """True iff ``kind`` is a valid :class:`~llmharness.schema.EdgeKind` value."""
    try:
        EdgeKind(kind)
    except ValueError:
        return False
    return True


__all__ = [
    "EdgeDelete",
    "EdgeUpsert",
    "GraphOp",
    "NodeDelete",
    "NodeUpsert",
    "parse_op",
    "validate_edge_kind",
    "validate_node_upsert_kind",
]
