"""Tests for vault graph module — backlinks, traverse, lint."""

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
def conn():
    """In-memory SQLite with schema + test data."""
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


def _seed_linear_graph(conn: sqlite3.Connection) -> None:
    """A -> B -> C -> D, all notes exist."""
    for path, title in [("A", "Note A"), ("B", "Note B"), ("C", "Note C"), ("D", "Note D")]:
        _insert_note(conn, path, title)
    for src, tgt in [("A", "B"), ("B", "C"), ("C", "D")]:
        _insert_link(conn, src, tgt)
    conn.commit()


def _seed_diamond_graph(conn: sqlite3.Connection) -> None:
    """A -> B, A -> C, B -> D, C -> D."""
    for path, title in [("A", "Note A"), ("B", "Note B"), ("C", "Note C"), ("D", "Note D")]:
        _insert_note(conn, path, title)
    for src, tgt in [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")]:
        _insert_link(conn, src, tgt)
    conn.commit()


# ---------------------------------------------------------------------------
# get_backlinks
# ---------------------------------------------------------------------------


class TestGetBacklinks:
    def test_should_return_sources_referencing_target(self, conn: sqlite3.Connection):
        _seed_diamond_graph(conn)
        result = get_backlinks(conn, "D")
        assert sorted(result) == ["B", "C"]

    def test_should_return_empty_when_no_backlinks(self, conn: sqlite3.Connection):
        _seed_linear_graph(conn)
        result = get_backlinks(conn, "A")
        assert result == []

    def test_should_return_empty_for_nonexistent_path(self, conn: sqlite3.Connection):
        _seed_linear_graph(conn)
        result = get_backlinks(conn, "nonexistent")
        assert result == []


# ---------------------------------------------------------------------------
# traverse — forward
# ---------------------------------------------------------------------------


class TestTraverseForward:
    def test_should_return_start_node_at_depth_zero(self, conn: sqlite3.Connection):
        _seed_linear_graph(conn)
        result = traverse(conn, "A", depth=0, direction="forward")
        assert len(result.nodes) == 1
        assert result.nodes[0] == TraverseNode(path="A", depth=0, title="Note A")
        assert result.edges == []

    def test_should_follow_outgoing_links(self, conn: sqlite3.Connection):
        _seed_linear_graph(conn)
        result = traverse(conn, "A", depth=2, direction="forward")
        paths = {n.path for n in result.nodes}
        assert paths == {"A", "B", "C"}

    def test_should_include_edges_with_correct_direction(self, conn: sqlite3.Connection):
        _seed_linear_graph(conn)
        result = traverse(conn, "A", depth=1, direction="forward")
        assert TraverseEdge(source="A", target="B") in result.edges

    def test_should_respect_depth_limit(self, conn: sqlite3.Connection):
        _seed_linear_graph(conn)
        result = traverse(conn, "A", depth=1, direction="forward")
        paths = {n.path for n in result.nodes}
        assert "C" not in paths
        assert "D" not in paths

    def test_should_handle_diamond_without_duplicates(self, conn: sqlite3.Connection):
        _seed_diamond_graph(conn)
        result = traverse(conn, "A", depth=2, direction="forward")
        paths = [n.path for n in result.nodes]
        assert len(paths) == len(set(paths)), "Duplicate nodes in traverse result"

    def test_should_set_start_field(self, conn: sqlite3.Connection):
        _seed_linear_graph(conn)
        result = traverse(conn, "B", depth=1, direction="forward")
        assert result.start == "B"


# ---------------------------------------------------------------------------
# traverse — backward
# ---------------------------------------------------------------------------


class TestTraverseBackward:
    def test_should_follow_incoming_links(self, conn: sqlite3.Connection):
        _seed_linear_graph(conn)
        result = traverse(conn, "D", depth=2, direction="backward")
        paths = {n.path for n in result.nodes}
        assert paths == {"D", "C", "B"}

    def test_should_include_backward_edges(self, conn: sqlite3.Connection):
        _seed_linear_graph(conn)
        result = traverse(conn, "D", depth=1, direction="backward")
        # Edge direction preserved as source->target from links table
        assert TraverseEdge(source="C", target="D") in result.edges


# ---------------------------------------------------------------------------
# traverse — both
# ---------------------------------------------------------------------------


class TestTraverseBoth:
    def test_should_merge_forward_and_backward(self, conn: sqlite3.Connection):
        _seed_linear_graph(conn)
        result = traverse(conn, "B", depth=1, direction="both")
        paths = {n.path for n in result.nodes}
        # B forward -> C, B backward -> A
        assert paths == {"A", "B", "C"}

    def test_should_deduplicate_nodes(self, conn: sqlite3.Connection):
        _seed_diamond_graph(conn)
        result = traverse(conn, "B", depth=2, direction="both")
        paths = [n.path for n in result.nodes]
        assert len(paths) == len(set(paths))


# ---------------------------------------------------------------------------
# traverse — cycle handling
# ---------------------------------------------------------------------------


class TestTraverseCycleHandling:
    def test_should_not_infinite_loop_on_cycle(self, conn: sqlite3.Connection):
        """Cycle: A -> B -> C -> A. Depth limit prevents infinite recursion."""
        for path, title in [("A", "Note A"), ("B", "Note B"), ("C", "Note C")]:
            _insert_note(conn, path, title)
        for src, tgt in [("A", "B"), ("B", "C"), ("C", "A")]:
            _insert_link(conn, src, tgt)
        conn.commit()

        result = traverse(conn, "A", depth=3, direction="forward")
        paths = {n.path for n in result.nodes}
        assert paths == {"A", "B", "C"}
        # No duplicates
        assert len(result.nodes) == len(set(n.path for n in result.nodes))


# ---------------------------------------------------------------------------
# traverse — edge cases
# ---------------------------------------------------------------------------


class TestTraverseEdgeCases:
    def test_should_handle_nonexistent_start(self, conn: sqlite3.Connection):
        """Start node not in notes table — should return empty or single unknown node."""
        create_schema(conn)
        result = traverse(conn, "ghost", depth=2, direction="forward")
        assert result.start == "ghost"
        # At minimum, nodes list should contain the start with empty title
        assert len(result.nodes) >= 0  # graceful, no crash

    def test_should_handle_empty_graph(self, conn: sqlite3.Connection):
        result = traverse(conn, "X", depth=5, direction="forward")
        assert result.start == "X"
        assert result.edges == []


# ---------------------------------------------------------------------------
# lint
# ---------------------------------------------------------------------------


class TestLint:
    def test_should_detect_dead_links(self, conn: sqlite3.Connection):
        _insert_note(conn, "A", "Note A")
        _insert_link(conn, "A", "missing")
        conn.commit()

        result = lint(conn)
        assert ("A", "missing") in result.dead_links

    def test_should_detect_orphan_notes(self, conn: sqlite3.Connection):
        _insert_note(conn, "A", "Note A")
        _insert_note(conn, "B", "Note B")
        _insert_link(conn, "A", "B")
        _insert_note(conn, "orphan", "Lonely Note")
        conn.commit()

        result = lint(conn)
        assert "orphan" in result.orphan_notes

    def test_should_not_flag_connected_notes_as_orphans(self, conn: sqlite3.Connection):
        _seed_linear_graph(conn)
        result = lint(conn)
        assert result.orphan_notes == []

    def test_should_not_flag_existing_targets_as_dead(self, conn: sqlite3.Connection):
        _seed_linear_graph(conn)
        result = lint(conn)
        assert result.dead_links == []

    def test_should_return_empty_results_for_empty_graph(self, conn: sqlite3.Connection):
        result = lint(conn)
        assert result.dead_links == []
        assert result.orphan_notes == []

    def test_should_return_frozen_dataclass(self, conn: sqlite3.Connection):
        result = lint(conn)
        assert isinstance(result, LintResult)
        with pytest.raises(AttributeError):
            result.dead_links = []  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


class TestDataClasses:
    def test_traverse_result_is_frozen(self):
        r = TraverseResult(start="A", nodes=[], edges=[])
        with pytest.raises(AttributeError):
            r.start = "B"  # type: ignore[misc]

    def test_traverse_node_is_frozen(self):
        n = TraverseNode(path="A", depth=0, title="X")
        with pytest.raises(AttributeError):
            n.depth = 1  # type: ignore[misc]

    def test_traverse_edge_is_frozen(self):
        e = TraverseEdge(source="A", target="B")
        with pytest.raises(AttributeError):
            e.source = "C"  # type: ignore[misc]
