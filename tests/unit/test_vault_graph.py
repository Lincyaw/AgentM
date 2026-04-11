"""Focused regression tests for vault graph operations."""

from __future__ import annotations

import sqlite3

import pytest

from agentm.tools.vault.graph import (
    LintResult,
    TraverseEdge,
    TraverseNode,
    TraverseResult,
    get_backlinks,
    lint,
    traverse,
)
from agentm.tools.vault.schema import create_schema


@pytest.fixture
def conn() -> sqlite3.Connection:
    c = sqlite3.connect(":memory:")
    create_schema(c)
    return c


def _insert_note(conn: sqlite3.Connection, path: str, title: str) -> None:
    conn.execute(
        "INSERT INTO notes (path, type, title, confidence) VALUES (?, 'concept', ?, 'fact')",
        (path, title),
    )


def _insert_link(conn: sqlite3.Connection, source: str, target: str) -> None:
    conn.execute("INSERT INTO links (source, target) VALUES (?, ?)", (source, target))


def _seed_diamond(conn: sqlite3.Connection) -> None:
    for path in ("A", "B", "C", "D"):
        _insert_note(conn, path, f"Note {path}")
    for source, target in (("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")):
        _insert_link(conn, source, target)
    conn.commit()


def test_get_backlinks_returns_all_sources_for_target(conn: sqlite3.Connection) -> None:
    _seed_diamond(conn)

    assert sorted(get_backlinks(conn, "D")) == ["B", "C"]
    assert get_backlinks(conn, "ghost") == []


def test_traverse_respects_direction_depth_and_edge_orientation(conn: sqlite3.Connection) -> None:
    for path in ("A", "B", "C"):
        _insert_note(conn, path, f"Note {path}")
    _insert_link(conn, "A", "B")
    _insert_link(conn, "B", "C")
    conn.commit()

    forward = traverse(conn, "A", depth=1, direction="forward")
    assert forward.start == "A"
    assert {node.path for node in forward.nodes} == {"A", "B"}
    assert TraverseEdge(source="A", target="B") in forward.edges

    backward = traverse(conn, "C", depth=2, direction="backward")
    assert {node.path for node in backward.nodes} == {"A", "B", "C"}


def test_traverse_both_deduplicates_nodes_across_paths(conn: sqlite3.Connection) -> None:
    _seed_diamond(conn)

    result = traverse(conn, "B", depth=2, direction="both")
    paths = [node.path for node in result.nodes]

    assert len(paths) == len(set(paths))
    assert "D" in paths


def test_lint_detects_dead_links_and_orphans(conn: sqlite3.Connection) -> None:
    _insert_note(conn, "A", "Note A")
    _insert_note(conn, "B", "Note B")
    _insert_note(conn, "orphan", "Orphan")
    _insert_link(conn, "A", "B")
    _insert_link(conn, "A", "missing")
    conn.commit()

    result = lint(conn)

    assert ("A", "missing") in result.dead_links
    assert "orphan" in result.orphan_notes


def test_graph_dataclasses_are_frozen() -> None:
    with pytest.raises(AttributeError):
        TraverseResult(start="A", nodes=[], edges=[]).start = "B"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        TraverseNode(path="A", depth=0, title="A").depth = 1  # type: ignore[misc]
    with pytest.raises(AttributeError):
        TraverseEdge(source="A", target="B").source = "C"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        LintResult(dead_links=[], orphan_notes=[]).dead_links = []  # type: ignore[misc]
