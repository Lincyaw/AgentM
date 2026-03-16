"""Vault link-graph operations — backlinks, traverse, lint."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TraverseNode:
    path: str
    depth: int
    title: str


@dataclass(frozen=True)
class TraverseEdge:
    source: str
    target: str


@dataclass(frozen=True)
class TraverseResult:
    start: str
    nodes: list[TraverseNode]
    edges: list[TraverseEdge]


@dataclass(frozen=True)
class LintResult:
    dead_links: list[tuple[str, str]]  # (source, target)
    orphan_notes: list[str]


# ---------------------------------------------------------------------------
# Backlinks
# ---------------------------------------------------------------------------


def get_backlinks(conn: sqlite3.Connection, path: str) -> list[str]:
    """Return all source paths that link to *path*."""
    rows = conn.execute(
        "SELECT source FROM links WHERE target = ?", (path,)
    ).fetchall()
    return [r[0] for r in rows]


# ---------------------------------------------------------------------------
# Traverse
# ---------------------------------------------------------------------------

_FORWARD_CTE = """\
WITH RECURSIVE graph(path, depth) AS (
    SELECT ?, 0
    UNION
    SELECT l.target, g.depth + 1
    FROM links l JOIN graph g ON l.source = g.path
    WHERE g.depth < ?
)
SELECT DISTINCT g.path, MIN(g.depth) AS depth, COALESCE(n.title, '') AS title
FROM graph g LEFT JOIN notes n ON g.path = n.path
GROUP BY g.path
"""

_BACKWARD_CTE = """\
WITH RECURSIVE graph(path, depth) AS (
    SELECT ?, 0
    UNION
    SELECT l.source, g.depth + 1
    FROM links l JOIN graph g ON l.target = g.path
    WHERE g.depth < ?
)
SELECT DISTINCT g.path, MIN(g.depth) AS depth, COALESCE(n.title, '') AS title
FROM graph g LEFT JOIN notes n ON g.path = n.path
GROUP BY g.path
"""


def _collect_edges(
    conn: sqlite3.Connection,
    node_paths: set[str],
    direction: str,
) -> list[TraverseEdge]:
    """Return directed edges between discovered nodes."""
    if not node_paths:
        return []
    placeholders = ",".join("?" for _ in node_paths)
    paths = list(node_paths)
    if direction == "forward":
        sql = f"SELECT source, target FROM links WHERE source IN ({placeholders}) AND target IN ({placeholders})"  # noqa: S608
    elif direction == "backward":
        sql = f"SELECT source, target FROM links WHERE target IN ({placeholders}) AND source IN ({placeholders})"  # noqa: S608
    else:  # both
        sql = (
            f"SELECT source, target FROM links "  # noqa: S608
            f"WHERE source IN ({placeholders}) AND target IN ({placeholders})"
        )
    params = paths + paths if direction != "both" else paths + paths
    rows = conn.execute(sql, params).fetchall()
    return [TraverseEdge(source=r[0], target=r[1]) for r in rows]


def traverse(
    conn: sqlite3.Connection,
    start: str,
    depth: int,
    direction: str,
) -> TraverseResult:
    """BFS subgraph traversal with depth limit.

    *direction*: ``"forward"`` | ``"backward"`` | ``"both"``.
    """
    node_map: dict[str, TraverseNode] = {}

    if direction in ("forward", "both"):
        for row in conn.execute(_FORWARD_CTE, (start, depth)).fetchall():
            path, d, title = row
            if path not in node_map or d < node_map[path].depth:
                node_map[path] = TraverseNode(path=path, depth=d, title=title)

    if direction in ("backward", "both"):
        for row in conn.execute(_BACKWARD_CTE, (start, depth)).fetchall():
            path, d, title = row
            if path not in node_map or d < node_map[path].depth:
                node_map[path] = TraverseNode(path=path, depth=d, title=title)

    nodes = sorted(node_map.values(), key=lambda n: (n.depth, n.path))
    node_paths = {n.path for n in nodes}
    edges = _collect_edges(conn, node_paths, direction)

    return TraverseResult(start=start, nodes=nodes, edges=edges)


# ---------------------------------------------------------------------------
# Lint
# ---------------------------------------------------------------------------


def lint(conn: sqlite3.Connection) -> LintResult:
    """Detect dead links and orphan notes."""
    dead_rows = conn.execute(
        "SELECT source, target FROM links "
        "WHERE target NOT IN (SELECT path FROM notes)"
    ).fetchall()
    dead_links = [(r[0], r[1]) for r in dead_rows]

    orphan_rows = conn.execute(
        "SELECT path FROM notes "
        "WHERE path NOT IN (SELECT source FROM links) "
        "AND path NOT IN (SELECT target FROM links)"
    ).fetchall()
    orphan_notes = [r[0] for r in orphan_rows]

    return LintResult(dead_links=dead_links, orphan_notes=orphan_notes)
