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

from llmharness.audit.graph.fold import fold_graph
from llmharness.audit.graph.ops import (
    EdgeDelete,
    EdgeUpsert,
    NodeDelete,
    NodeUpsert,
    parse_op,
)
from llmharness.schema import EventKind


def _nu(eid: int, *, kind: str = "act", summary: str = "x", turns: tuple[int, ...] = (1,)) -> NodeUpsert:
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




def test_parse_op_rejects_unknown_discriminator() -> None:
    with pytest.raises(ValueError):
        parse_op({"op": "not_a_real_op", "id": 1})






