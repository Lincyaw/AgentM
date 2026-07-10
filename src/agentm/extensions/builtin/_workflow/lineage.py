"""Deterministic backward-edge lineage derivation over the workflow journal.

The journal keys every ``agent()`` call by ``sha256(prompt, opts)`` and
downstream calls interpolate upstream results into their prompts, so the
dependency graph is already content-addressed: entry X's result appearing
verbatim inside entry Y's prompt means X's output flowed into Y. This module
derives those edges by pure substring matching — the session index's
deterministic layer; no LLM is involved and no self-report is trusted
(reliability-substrate.md §4.2).

Precision caveat, stated honestly: when a script transforms a result before
interpolating it (extracts one JSON field, reformats), verbatim matching
misses that edge. A node with NO verbatim parent falls back to conservative
program-order candidates (every entry recorded earlier), so fully-transformed
dataflow degrades to a complete-but-imprecise candidate set — but a node
with one verbatim parent plus one transformed dependency reports only the
verbatim edge and gets no fallback. Consumers scoping an invalidation
closure should treat verbatim edges as the precise subset, not as proof
there are no other dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentm.extensions.builtin._workflow.journal import JournalEntry

# Results shorter than this are too generic to treat a substring hit as a
# real dataflow edge ("ok", "yes", a bare number would match everywhere).
_MIN_VERBATIM_LEN = 12
# Cheap pre-match: results can be ~100KB and ``in`` cost scales with needle
# length, so probe with a bounded prefix first and confirm with the full
# needle only on prefix hits.
_FINGERPRINT_CHARS = 256


@dataclass(slots=True)
class LineageEdge:
    """``src``'s result flowed (verbatim) into ``dst``'s prompt."""

    src: str
    dst: str


@dataclass(slots=True)
class LineageGraph:
    edges: list[LineageEdge]
    # key → earlier-entry keys, for nodes with a prompt but no verbatim
    # parent (transformed-dataflow fallback; conservative, oldest first).
    order_candidates: dict[str, list[str]]


def derive_lineage(entries: list[JournalEntry]) -> LineageGraph:
    """Derive backward edges among journal entries by verbatim matching."""
    ordered = sorted(entries, key=lambda entry: entry.timestamp)
    edges: list[LineageEdge] = []
    for src_index, src in enumerate(ordered):
        needle = src.result.strip()
        if len(needle) < _MIN_VERBATIM_LEN:
            continue
        fingerprint = needle[:_FINGERPRINT_CHARS]
        # A result can only flow into a prompt recorded after it.
        for dst in ordered[src_index + 1 :]:
            if dst.prompt is None:
                continue
            if fingerprint in dst.prompt and needle in dst.prompt:
                edges.append(LineageEdge(src=src.key, dst=dst.key))

    has_parent = {edge.dst for edge in edges}
    order_candidates: dict[str, list[str]] = {}
    prefix: list[str] = []
    for entry in ordered:
        if entry.prompt is not None and entry.key not in has_parent and prefix:
            order_candidates[entry.key] = list(prefix)
        prefix.append(entry.key)
    return LineageGraph(edges=edges, order_candidates=order_candidates)


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
