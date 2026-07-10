"""Deterministic backward-edge lineage derivation over the workflow journal.

The journal keys every ``agent()`` call by ``sha256(prompt, opts)`` and
downstream calls interpolate upstream results into their prompts, so the
dependency graph is already content-addressed: entry X's result appearing
verbatim inside entry Y's prompt means X's output flowed into Y. This module
derives those edges by pure substring matching — the session index's
deterministic layer; no LLM is involved and no self-report is trusted
(reliability-substrate.md §4.2).

When a script transforms a result before interpolating it (extracts one JSON
field, reformats), verbatim matching misses the edge. Nodes with a stored
prompt but no verbatim parent therefore fall back to conservative
program-order candidates: every entry recorded earlier. Precision degrades;
completeness does not. Note the fallback ordering uses record timestamps, so
after a node has been invalidated and re-recorded its timestamp is its
*latest* run — conservative candidates for downstream nodes remain a
superset, which keeps the fallback sound.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from agentm.extensions.builtin._workflow.sdk import JournalEntry

# Results shorter than this are too generic to treat a substring hit as a
# real dataflow edge ("ok", "yes", a bare number would match everywhere).
_MIN_VERBATIM_LEN = 12

EdgeKind = Literal["verbatim"]


@dataclass(slots=True)
class LineageEdge:
    """``src``'s result flowed (verbatim) into ``dst``'s prompt."""

    src: str
    dst: str
    kind: EdgeKind = "verbatim"


@dataclass(slots=True)
class LineageGraph:
    entries: list[JournalEntry]
    edges: list[LineageEdge]
    # key → earlier-entry keys, for nodes with a prompt but no verbatim
    # parent (transformed-dataflow fallback; conservative, ordered oldest
    # first).
    order_candidates: dict[str, list[str]]


def derive_lineage(entries: list[JournalEntry]) -> LineageGraph:
    """Derive backward edges among journal entries by verbatim matching."""
    edges: list[LineageEdge] = []
    for src in entries:
        needle = src.result.strip()
        if len(needle) < _MIN_VERBATIM_LEN:
            continue
        for dst in entries:
            if dst.key == src.key or dst.prompt is None:
                continue
            if needle in dst.prompt:
                edges.append(LineageEdge(src=src.key, dst=dst.key))

    has_parent = {edge.dst for edge in edges}
    ordered = sorted(entries, key=lambda entry: entry.timestamp)
    order_candidates: dict[str, list[str]] = {}
    for index, entry in enumerate(ordered):
        if entry.prompt is None or entry.key in has_parent:
            continue
        earlier = [prior.key for prior in ordered[:index]]
        if earlier:
            order_candidates[entry.key] = earlier
    return LineageGraph(
        entries=entries, edges=edges, order_candidates=order_candidates
    )


def ancestors(graph: LineageGraph, key: str) -> list[str]:
    """Transitive verbatim ancestors of ``key`` (nearest first, BFS order)."""
    parents_of: dict[str, list[str]] = {}
    for edge in graph.edges:
        parents_of.setdefault(edge.dst, []).append(edge.src)
    seen: set[str] = set()
    frontier = [key]
    out: list[str] = []
    while frontier:
        node = frontier.pop(0)
        for parent in parents_of.get(node, []):
            if parent in seen or parent == key:
                continue
            seen.add(parent)
            out.append(parent)
            frontier.append(parent)
    return out


__all__ = (
    "LineageEdge",
    "LineageGraph",
    "ancestors",
    "derive_lineage",
)
