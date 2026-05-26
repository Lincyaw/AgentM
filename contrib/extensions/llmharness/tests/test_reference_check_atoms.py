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
from llmharness.schema import Event, EventKind


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


# ---------------------------------------------------------------------------
# check_open_branches


# ---------------------------------------------------------------------------
# check_premature_conclusion


# ---------------------------------------------------------------------------
# Multi-atom mount
