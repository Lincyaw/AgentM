"""Phase 1 graph validator — DEPRECATED stub (v3 transitional, commit 1/5).

The v2 ``validate(...)`` checks (ref resolution, no cycles, kind↔source
rules, task / conclusion reachability) all relied on ``Event.refs``,
which the v3 schema break removes (issue #134). The whole module is
deleted in commit 2 in favour of the witness pipeline at edge-build
time (design §4.c, §6).

For commit 1 we keep the import surface alive (``validate_graph`` is
still imported by ``adapters/agentm.py`` and re-exported from
``audit.extractor.__init__``) but the body is a no-op that always
reports zero violations. Adapter behaviour during commit 1: every graph
passes; commit 2 rewrites the call site.
"""

from __future__ import annotations

from ...schema import Event


def validate(
    *,
    new_events: list[Event],
    existing_events: list[Event],
    turn_index_to_kinds: dict[int, set[str]],
) -> list[str]:
    """No-op stub. Always returns ``[]`` until commit 2 deletes this module."""

    del new_events, existing_events, turn_index_to_kinds  # silence unused-arg lint
    return []


__all__ = ["validate"]
