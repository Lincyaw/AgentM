"""Fail-stop tests for the audit-check registry (issue #134, v3 commit 1).

Pinned positions: the registry's contract with scenario atoms
(idempotency, raise-tolerant fan-out, fail-fast on non-callables).
Anything else — re-export plumbing, Protocol typing, dataclass
round-trips — is framework guarantee and not tested here.
"""

from __future__ import annotations

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








def test_run_all_captures_raising_check_other_checks_still_run() -> None:
    """A check that raises is captured into errors-by-name; others still run."""

    f_ok = Finding(category="ok", description="fine")

    reg = AuditCheckRegistry()
    reg.register_check(_RaisingCheck(name="boom", msg="kaboom"))
    reg.register_check(_NamedCheck(name="ok", findings=[f_ok]))

    findings, errors = reg.run_all(_empty_ctx())
    assert findings == [f_ok]
    assert errors == {"boom": "kaboom"}




