"""Fail-stop tests for the event-sourcing op log + fold.

The persistent audit graph is the **fold** of an ordered op list. If
:func:`fold_graph` semantics drift or :meth:`GraphOp.to_dict` /
``from_dict`` round-trip lose data, every replayed graph downstream
goes wrong. These tests pin the load-bearing positions:

- empty fold = empty graph
- last-write-wins on repeated NodeUpsert
- NodeDelete cascades to all incident edges
- EdgeDelete keyed by (src, dst, kind) — removing data leaves ref alone
- replay equivalence: fold(ops_a + ops_b) == fold(ops_a then ops_b)
- to_dict / from_dict round-trips preserve every op variant
"""

from __future__ import annotations

import pytest

from llmharness.audit.graph_fold import fold_graph
from llmharness.audit.graph_ops import (
    EdgeDelete,
    EdgeUpsert,
    NodeDelete,
    NodeUpsert,
    parse_op,
)
from llmharness.schema import EdgeKind, EventKind, ExternalRef


def _nu(eid: int, *, kind: str = "evid", summary: str = "x", turns: tuple[int, ...] = (1,)) -> NodeUpsert:
    return NodeUpsert(id=eid, kind=kind, summary=summary, source_turns=turns)


def _eu(src: int, dst: int, *, kind: str = "data", reason: str = "r") -> EdgeUpsert:
    return EdgeUpsert(
        src=src,
        dst=dst,
        kind=kind,
        reason=reason,
        cited_entities=("ent",) if kind == "data" else (),
        cited_quote="" if kind == "data" else "quote",
        src_turns=(1,),
        dst_turns=(2,),
    )


# --- fold semantics --------------------------------------------------------


def test_empty_op_list_yields_empty_graph() -> None:
    g = fold_graph([])
    assert g.nodes == {}
    assert g.edges == {}
    assert g.nodes_list() == []
    assert g.edges_list() == []


def test_single_node_upsert_creates_node() -> None:
    g = fold_graph([_nu(1, kind="task", summary="root", turns=(1, 2))])
    assert list(g.nodes) == [1]
    ev = g.nodes[1]
    assert ev.id == 1
    assert ev.kind is EventKind.TASK
    assert ev.summary == "root"
    assert ev.source_turns == [1, 2]


def test_repeated_node_upsert_is_last_write_wins() -> None:
    g = fold_graph(
        [
            _nu(1, summary="first"),
            _nu(1, summary="revised"),
            _nu(1, kind="concl", summary="final"),
        ]
    )
    assert len(g.nodes) == 1
    assert g.nodes[1].summary == "final"
    assert g.nodes[1].kind is EventKind.CONCL


def test_node_delete_cascades_to_incident_edges() -> None:
    """A deleted node drops every edge that touches it on either side."""
    ops = [
        _nu(1),
        _nu(2),
        _nu(3),
        _eu(1, 2, kind="data"),
        _eu(2, 3, kind="ref"),
        _eu(1, 3, kind="ref"),
        NodeDelete(id=2),
    ]
    g = fold_graph(ops)
    assert set(g.nodes.keys()) == {1, 3}
    # (1,2,data) and (2,3,ref) cascade; (1,3,ref) survives.
    assert set(g.edges.keys()) == {(1, 3, "ref")}


def test_edge_delete_removes_only_the_keyed_edge() -> None:
    """Two edges between the same nodes differ by kind; delete is keyed."""
    ops = [
        _nu(1),
        _nu(2),
        _eu(1, 2, kind="data"),
        _eu(1, 2, kind="ref"),
        EdgeDelete(src=1, dst=2, kind="data"),
    ]
    g = fold_graph(ops)
    assert set(g.edges.keys()) == {(1, 2, "ref")}


def test_edge_delete_of_missing_edge_is_noop() -> None:
    g = fold_graph(
        [
            _nu(1),
            _nu(2),
            EdgeDelete(src=1, dst=2, kind="data"),  # never existed
        ]
    )
    assert g.edges == {}


def test_node_delete_of_missing_node_is_noop() -> None:
    g = fold_graph([NodeDelete(id=42)])
    assert g.nodes == {}


def test_fold_is_associative_against_concatenation() -> None:
    """``fold(a + b) == fold(fold(a) then b)`` is the property the
    cross-firing op log relies on: each firing's ops can be applied on
    top of any prior fold and yield the same final state as folding
    the whole concatenated log from scratch.
    """
    ops_a = [_nu(1), _nu(2), _eu(1, 2, kind="data")]
    ops_b = [_nu(3), _eu(2, 3, kind="ref"), NodeDelete(id=1)]

    direct = fold_graph(ops_a + ops_b)

    fold_a = fold_graph(ops_a)
    # Materialise fold_a back as ops to apply ops_b on top.
    replay_prefix = [
        NodeUpsert(
            id=ev.id,
            kind=ev.kind.value,
            summary=ev.summary,
            source_turns=tuple(ev.source_turns),
        )
        for ev in fold_a.nodes_list()
    ] + [
        EdgeUpsert(
            src=ed.src,
            dst=ed.dst,
            kind=ed.kind.value,
            reason=ed.reason,
            cited_entities=ed.cited_entities,
            cited_quote=ed.cited_quote,
            src_turns=ed.src_turns,
            dst_turns=ed.dst_turns,
        )
        for ed in fold_a.edges_list()
    ]
    indirect = fold_graph(replay_prefix + ops_b)

    assert set(direct.nodes.keys()) == set(indirect.nodes.keys())
    assert set(direct.edges.keys()) == set(indirect.edges.keys())


def test_nodes_and_edges_iterate_in_insertion_order() -> None:
    """Downstream consumers (auditor compose, replay sidecar) depend on
    stable ordering. The fold MUST preserve op order in the materialised
    node / edge lists.
    """
    g = fold_graph([_nu(3), _nu(1), _nu(2), _eu(3, 1, kind="data"), _eu(1, 2, kind="ref")])
    assert [ev.id for ev in g.nodes_list()] == [3, 1, 2]
    assert [(e.src, e.dst, e.kind.value) for e in g.edges_list()] == [
        (3, 1, "data"),
        (1, 2, "ref"),
    ]


# --- round-trip ------------------------------------------------------------


@pytest.mark.parametrize(
    "op",
    [
        NodeUpsert(id=7, kind="hyp", summary="bet", source_turns=(3, 4, 5)),
        # Round-trip for the external_refs payload — this is the B2
        # fix: NodeUpsert must carry the cross-firing edge tuple
        # through to_dict / from_dict so legacy AUDIT_EVENT translation
        # and live op-log writes preserve cumulative connectivity.
        NodeUpsert(
            id=8,
            kind="evid",
            summary="evidence",
            source_turns=(10,),
            external_refs=(
                ExternalRef(
                    to_recent_event_id=3,
                    kind=EdgeKind.DATA,
                    reason="follows from prior table",
                    cited_entities=("abnormal_traces",),
                    cited_quote="",
                ),
            ),
        ),
        NodeDelete(id=12),
        EdgeUpsert(
            src=1,
            dst=2,
            kind="data",
            reason="evidence supports",
            cited_entities=("table_a", "service_b"),
            cited_quote="",
            src_turns=(1, 2),
            dst_turns=(3,),
        ),
        EdgeUpsert(
            src=1,
            dst=2,
            kind="ref",
            reason="references earlier turn",
            cited_entities=(),
            cited_quote="latency spike",
            src_turns=(1,),
            dst_turns=(3,),
        ),
        EdgeDelete(src=1, dst=2, kind="ref"),
    ],
)
def test_to_dict_from_dict_round_trip(op: object) -> None:
    """Every op variant must round-trip through ``to_dict`` /
    :func:`parse_op` losslessly — replay reads the dict form back from
    durable session entries."""
    d = op.to_dict()  # type: ignore[attr-defined]
    parsed = parse_op(d)
    assert parsed == op


def test_parse_op_rejects_unknown_discriminator() -> None:
    with pytest.raises(ValueError):
        parse_op({"op": "not_a_real_op", "id": 1})


def test_parse_op_rejects_missing_discriminator() -> None:
    with pytest.raises(ValueError):
        parse_op({"id": 1, "kind": "evid"})


def test_node_upsert_carries_external_refs_through_fold() -> None:
    """The fold must materialise ``Event.external_refs`` from the op,
    not synthesize ``()`` defaults. Regression for B2: prior code
    used ``Event(...)`` without the field, so legacy AUDIT_EVENT
    translation silently dropped cross-firing connectivity that the
    auditor + next firing's ``recent_graph`` payload both depend on.
    """
    ext = ExternalRef(
        to_recent_event_id=5,
        kind=EdgeKind.REF,
        reason="cites earlier turn",
        cited_entities=(),
        cited_quote="latency spike",
    )
    g = fold_graph(
        [
            NodeUpsert(
                id=8,
                kind="concl",
                summary="root cause",
                source_turns=(20,),
                external_refs=(ext,),
            ),
        ]
    )
    ev = g.nodes[8]
    assert ev.external_refs == (ext,)


def test_edge_kind_round_trip_against_schema_enum() -> None:
    """The folded :class:`Edge` carries an :class:`EdgeKind` enum even
    though ops store kind as a string — proves the fold materialises
    the schema-level type correctly.
    """
    g = fold_graph(
        [
            _nu(1),
            _nu(2),
            _eu(1, 2, kind="data"),
            _eu(1, 2, kind="ref"),
        ]
    )
    kinds = {e.kind for e in g.edges_list()}
    assert kinds == {EdgeKind.DATA, EdgeKind.REF}
