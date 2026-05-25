"""Fail-stop tests for the auditor's findings + check_errors plumbing.

The auditor prompt embeds an advisory FINDINGS block populated from the
:class:`AuditCheckRegistry`. Empty / raising / multi-finding paths must
each surface predictably; otherwise scenario-registered checks become
unreliable signals — the worst kind because they look reliable.
"""

from __future__ import annotations

import json

from llmharness.audit.auditor.prompt import build_auditor_system_prompt
from llmharness.audit.registry import AuditCheckRegistry, CheckContext
from llmharness.schema import Finding


def test_multiple_findings_appear_in_registration_order() -> None:
    f1 = Finding(category="loop", description="repeated act #3", related_event_ids=(3,))
    f2 = Finding(category="branch", description="open dec #5", related_event_ids=(5,))

    class _Check:
        name = "demo"

        def __call__(self, ctx: CheckContext) -> list[Finding]:
            return [f1, f2]

    registry = AuditCheckRegistry()
    registry.register_check(_Check())
    findings, errors = registry.run_all(CheckContext(events=(), edges=()))
    assert findings == [f1, f2]

    prompt = build_auditor_system_prompt(
        events=(),
        edges=(),
        findings=findings,
        check_errors=errors,
        continuation_notes=[],
        summary_threshold=30,
    )
    # Both descriptions present, in order.
    pos1 = prompt.find("repeated act #3")
    pos2 = prompt.find("open dec #5")
    assert pos1 != -1 and pos2 != -1
    assert pos1 < pos2


def test_raising_check_recorded_other_checks_still_run() -> None:
    class _Boom:
        name = "boom_check"

        def __call__(self, ctx: CheckContext) -> list[Finding]:
            raise RuntimeError("boom!")

    survivor_finding = Finding(
        category="ok", description="survivor saw something", related_event_ids=()
    )

    class _Survivor:
        name = "survivor_check"

        def __call__(self, ctx: CheckContext) -> list[Finding]:
            return [survivor_finding]

    registry = AuditCheckRegistry()
    registry.register_check(_Boom())
    registry.register_check(_Survivor())
    findings, errors = registry.run_all(CheckContext(events=(), edges=()))

    assert findings == [survivor_finding]
    assert "boom_check" in errors
    assert "boom!" in errors["boom_check"]

    prompt = build_auditor_system_prompt(
        events=(),
        edges=(),
        findings=findings,
        check_errors=errors,
        continuation_notes=[],
        summary_threshold=30,
    )
    # Failed check name is rendered in a non-blocking line.
    assert "checks_failed" in prompt
    assert "boom_check" in prompt
    assert "non-blocking" in prompt
    # Survivor finding still rendered.
    assert "survivor saw something" in prompt
    # Sanity: the embedded JSON line for checks_failed parses.
    for line in prompt.splitlines():
        if line.startswith("checks_failed: "):
            payload = line[len("checks_failed: ") :].split(" (non-blocking")[0]
            parsed = json.loads(payload)
            assert "boom_check" in parsed
            break
    else:  # pragma: no cover - sentinel: line must exist
        raise AssertionError("checks_failed line not found")
