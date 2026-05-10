"""Smoke tests for the v3 reference check atoms (issue #134, commit 5).

Each atom registers exactly one Check on an :class:`AuditCheckRegistry`
through the published ``llmharness.audit_registry`` service. These
tests exercise the full registration -> ``run_all`` path against
hand-crafted :class:`CheckContext` snapshots:

- A trigger context yields ≥1 finding with the expected category.
- A non-trigger context yields zero findings.
- Mounting all three atoms on the same registry and running against
  a context designed to trigger two of them yields exactly those
  two categories.

Pinned positions: install() fail-fast on missing service, and the
end-to-end registration pipe (atom.install -> registry.run_all ->
Finding). The registry's own contract is covered by
``test_audit_registry.py``; no need to re-test.
"""

from __future__ import annotations

from typing import Any

import pytest

from llmharness.audit.registry import (
    SERVICE_KEY,
    AuditCheckRegistry,
    CheckContext,
)
from llmharness.extensions import (
    check_open_branches,
    check_premature_conclusion,
    check_repeated_actions,
)
from llmharness.schema import Edge, EdgeKind, Event, EventKind


class _StubAPI:
    """Minimal ExtensionAPI duck — only ``get_service`` is exercised."""

    def __init__(self, services: dict[str, Any] | None = None) -> None:
        self._services: dict[str, Any] = dict(services or {})

    def get_service(self, name: str) -> Any | None:
        return self._services.get(name)


def _api_with_registry() -> tuple[_StubAPI, AuditCheckRegistry]:
    registry = AuditCheckRegistry()
    api = _StubAPI({SERVICE_KEY: registry})
    return api, registry


# ---------------------------------------------------------------------------
# install() fail-fast contract


def test_check_atoms_fail_fast_when_registry_service_missing() -> None:
    """Each atom raises RuntimeError when the registry service is absent."""

    api = _StubAPI()  # empty service map
    for atom in (
        check_repeated_actions,
        check_open_branches,
        check_premature_conclusion,
    ):
        with pytest.raises(RuntimeError, match="audit registry service"):
            atom.install(api, {})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# check_repeated_actions


def test_repeated_actions_triggers_on_identical_act_summaries() -> None:
    api, registry = _api_with_registry()
    check_repeated_actions.install(api, {})  # type: ignore[arg-type]

    events = (
        Event(id=1, kind=EventKind.ACT, summary="ran ls in /tmp"),
        Event(id=2, kind=EventKind.ACT, summary="ran ls in /tmp"),
        Event(id=3, kind=EventKind.ACT, summary="ran ls in /tmp"),
        Event(id=4, kind=EventKind.HYP, summary="ran ls in /tmp"),  # not act
    )
    findings, errors = registry.run_all(CheckContext(events=events, edges=()))

    assert errors == {}
    assert len(findings) == 1
    f = findings[0]
    assert f.category == "repeated_actions"
    assert f.related_event_ids == (1, 2, 3)


def test_repeated_actions_silent_when_each_summary_unique() -> None:
    api, registry = _api_with_registry()
    check_repeated_actions.install(api, {})  # type: ignore[arg-type]

    events = (
        Event(id=1, kind=EventKind.ACT, summary="ran ls"),
        Event(id=2, kind=EventKind.ACT, summary="ran cat"),
    )
    findings, errors = registry.run_all(CheckContext(events=events, edges=()))

    assert errors == {}
    assert findings == []


# ---------------------------------------------------------------------------
# check_open_branches


def test_open_branches_triggers_on_dec_or_hyp_without_outgoing_data_edge() -> None:
    api, registry = _api_with_registry()
    check_open_branches.install(api, {})  # type: ignore[arg-type]

    events = (
        Event(id=1, kind=EventKind.HYP, summary="maybe table A is the cause"),
        Event(id=2, kind=EventKind.DEC, summary="picked option B"),
        Event(id=3, kind=EventKind.EVID, summary="A row count is 0"),
    )
    # only event 1 has an outgoing data edge -> events 2 stays open;
    # event 3 isn't dec/hyp so it's irrelevant.
    edges = (
        Edge(
            src=1,
            dst=3,
            kind=EdgeKind.DATA,
            reason="hypothesis about A is supported by evidence",
            src_turns=(1,),
            dst_turns=(2,),
            cited_entities=("A",),
        ),
    )
    findings, errors = registry.run_all(CheckContext(events=events, edges=edges))

    assert errors == {}
    assert len(findings) == 1
    assert findings[0].category == "open_branches"
    assert findings[0].related_event_ids == (2,)


def test_open_branches_silent_when_every_dec_hyp_has_outgoing_data_edge() -> None:
    api, registry = _api_with_registry()
    check_open_branches.install(api, {})  # type: ignore[arg-type]

    events = (
        Event(id=1, kind=EventKind.HYP, summary="hyp"),
        Event(id=2, kind=EventKind.EVID, summary="evid"),
    )
    edges = (
        Edge(
            src=1,
            dst=2,
            kind=EdgeKind.DATA,
            reason="hyp supported by evid",
            src_turns=(1,),
            dst_turns=(2,),
            cited_entities=("x",),
        ),
    )
    findings, errors = registry.run_all(CheckContext(events=events, edges=edges))

    assert errors == {}
    assert findings == []


# ---------------------------------------------------------------------------
# check_premature_conclusion


def test_premature_conclusion_triggers_on_concl_with_few_incoming_edges() -> None:
    api, registry = _api_with_registry()
    check_premature_conclusion.install(api, {})  # type: ignore[arg-type]

    events = (
        Event(id=1, kind=EventKind.EVID, summary="evid"),
        Event(id=2, kind=EventKind.CONCL, summary="root cause is X"),
    )
    # only one incoming edge into concl #2 -> below threshold (2)
    edges = (
        Edge(
            src=1,
            dst=2,
            kind=EdgeKind.DATA,
            reason="evid supports concl",
            src_turns=(1,),
            dst_turns=(2,),
            cited_entities=("X",),
        ),
    )
    findings, errors = registry.run_all(CheckContext(events=events, edges=edges))

    assert errors == {}
    assert len(findings) == 1
    assert findings[0].category == "premature_conclusion"
    assert findings[0].related_event_ids == (2,)


def test_premature_conclusion_silent_when_concl_has_two_incoming_edges() -> None:
    api, registry = _api_with_registry()
    check_premature_conclusion.install(api, {})  # type: ignore[arg-type]

    events = (
        Event(id=1, kind=EventKind.EVID, summary="evid 1"),
        Event(id=2, kind=EventKind.EVID, summary="evid 2"),
        Event(id=3, kind=EventKind.CONCL, summary="root cause is X"),
    )
    edges = (
        Edge(
            src=1,
            dst=3,
            kind=EdgeKind.DATA,
            reason="e1 supports concl",
            src_turns=(1,),
            dst_turns=(3,),
            cited_entities=("X",),
        ),
        Edge(
            src=2,
            dst=3,
            kind=EdgeKind.REF,
            reason="e2 referenced by concl",
            src_turns=(2,),
            dst_turns=(3,),
            cited_entities=("X",),
        ),
    )
    findings, errors = registry.run_all(CheckContext(events=events, edges=edges))

    assert errors == {}
    assert findings == []


# ---------------------------------------------------------------------------
# Multi-atom mount


def test_multi_atom_mount_only_triggered_categories_fire() -> None:
    """Mount all three atoms; design context to trigger exactly two."""

    api, registry = _api_with_registry()
    check_repeated_actions.install(api, {})  # type: ignore[arg-type]
    check_open_branches.install(api, {})  # type: ignore[arg-type]
    check_premature_conclusion.install(api, {})  # type: ignore[arg-type]
    assert len(registry.registered_checks()) == 3

    # repeated_actions: events 1 and 2 share an act summary -> trigger.
    # open_branches: event 3 is a dec with NO outgoing data edge -> trigger.
    # premature_conclusion: there is no concl event -> silent.
    events = (
        Event(id=1, kind=EventKind.ACT, summary="ran ls"),
        Event(id=2, kind=EventKind.ACT, summary="ran ls"),
        Event(id=3, kind=EventKind.DEC, summary="picked plan A"),
    )
    findings, errors = registry.run_all(CheckContext(events=events, edges=()))

    assert errors == {}
    categories = {f.category for f in findings}
    assert categories == {"repeated_actions", "open_branches"}
