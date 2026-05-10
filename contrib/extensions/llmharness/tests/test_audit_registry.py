"""Fail-stop tests for the audit-check registry (issue #134, v3 commit 1).

Pinned positions: the registry's contract with scenario atoms
(idempotency, raise-tolerant fan-out, fail-fast on non-callables).
Anything else — re-export plumbing, Protocol typing, dataclass
round-trips — is framework guarantee and not tested here.
"""

from __future__ import annotations

import pytest

from llmharness.audit.registry import AuditCheckRegistry, CheckContext
from llmharness.schema import Finding


def _empty_ctx() -> CheckContext:
    return CheckContext(events=(), edges=())


class _NamedCheck:
    """A minimal callable Check stand-in usable by the registry."""

    def __init__(self, name: str, findings: list[Finding] | None = None) -> None:
        self.name = name
        self._findings = list(findings or [])

    def __call__(self, ctx: CheckContext) -> list[Finding]:
        return list(self._findings)


class _RaisingCheck:
    """A callable check that always raises — for fault-tolerance tests."""

    def __init__(self, name: str, msg: str) -> None:
        self.name = name
        self._msg = msg

    def __call__(self, ctx: CheckContext) -> list[Finding]:
        raise RuntimeError(self._msg)


def test_register_check_idempotent_on_same_name_and_id() -> None:
    """Re-registering the same callable under the same name is a no-op."""

    reg = AuditCheckRegistry()
    c = _NamedCheck(name="alpha")
    reg.register_check(c)
    reg.register_check(c)
    assert len(reg.registered_checks()) == 1


def test_register_check_distinct_callables_same_name_both_appear() -> None:
    """Two distinct callables sharing a name both register (key is (name, id))."""

    reg = AuditCheckRegistry()
    a = _NamedCheck(name="shared")
    b = _NamedCheck(name="shared")
    assert id(a) != id(b)
    reg.register_check(a)
    reg.register_check(b)
    registered = reg.registered_checks()
    assert len(registered) == 2
    assert registered[0] is a
    assert registered[1] is b


def test_run_all_aggregates_findings_in_registration_order() -> None:
    """``run_all`` concatenates findings from each check in registration order."""

    f_a = Finding(category="repeated_actions", description="ev #1 repeats")
    f_b = Finding(category="open_branches", description="ev #2 unclosed")
    f_c = Finding(category="open_branches", description="ev #3 unclosed")

    reg = AuditCheckRegistry()
    reg.register_check(_NamedCheck(name="check_a", findings=[f_a]))
    reg.register_check(_NamedCheck(name="check_b", findings=[f_b, f_c]))

    findings, errors = reg.run_all(_empty_ctx())
    assert findings == [f_a, f_b, f_c]
    assert errors == {}


def test_run_all_captures_raising_check_other_checks_still_run() -> None:
    """A check that raises is captured into errors-by-name; others still run."""

    f_ok = Finding(category="ok", description="fine")

    reg = AuditCheckRegistry()
    reg.register_check(_RaisingCheck(name="boom", msg="kaboom"))
    reg.register_check(_NamedCheck(name="ok", findings=[f_ok]))

    findings, errors = reg.run_all(_empty_ctx())
    assert findings == [f_ok]
    assert errors == {"boom": "kaboom"}


def test_run_all_empty_registry() -> None:
    """Empty registry yields no findings and no errors."""

    reg = AuditCheckRegistry()
    findings, errors = reg.run_all(_empty_ctx())
    assert findings == []
    assert errors == {}


def test_register_check_rejects_non_callable() -> None:
    """Non-callable input is rejected at registration time with TypeError."""

    reg = AuditCheckRegistry()
    with pytest.raises(TypeError, match="callable"):
        reg.register_check("not-a-check")  # type: ignore[arg-type]
