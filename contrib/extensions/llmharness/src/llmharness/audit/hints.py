"""Advisory hints module — DEPRECATED stub (v3 transitional, commit 1/5).

The v2 hint signals (``repeated_actions``, ``convergence_ratio``,
``reachability_gaps``, ``open_branches``, ``multi_branch_syntheses``,
plus the ``compute`` renderer) all walked ``Event.refs`` to traverse
the graph. The v3 schema break removes ``Event.refs`` and replaces it
with first-class :class:`~llmharness.schema.Edge` records (issue #134).

This whole module is deleted in commit 2 in favour of scenario-
registered :class:`~llmharness.audit.registry.Check` atoms returning
:class:`~llmharness.schema.Finding` records (design §4.c).

For commit 1 we keep the import surface alive (``adapters/agentm.py``
imports ``compute``) but every public function is a no-op stub.
"""

from __future__ import annotations

from ..schema import Event


def repeated_actions(graph: list[Event]) -> list[tuple[str, list[int]]]:
    """No-op stub. Always returns ``[]``."""

    del graph
    return []


def convergence_ratio(graph: list[Event]) -> float:
    """No-op stub. Always returns ``0.0``."""

    del graph
    return 0.0


def reachability_gaps(graph: list[Event]) -> list[int]:
    """No-op stub. Always returns ``[]``."""

    del graph
    return []


def open_branches(graph: list[Event]) -> list[int]:
    """No-op stub. Always returns ``[]``."""

    del graph
    return []


def multi_branch_syntheses(graph: list[Event]) -> list[int]:
    """No-op stub. Always returns ``[]``."""

    del graph
    return []


def compute(graph: list[Event]) -> str:
    """No-op stub. Always returns an empty string until commit 2 deletes this module."""

    del graph
    return ""


__all__ = [
    "compute",
    "convergence_ratio",
    "multi_branch_syntheses",
    "open_branches",
    "reachability_gaps",
    "repeated_actions",
]
