"""Audit-check registry — mechanism for scenario-registered graph checks.

Per design §4.c. Concrete checks ship as one-file atoms under
``llmharness.extensions``; this module is mechanism only.

The registry is owned by the llmharness adapter and published to
``ExtensionAPI`` under :data:`SERVICE_KEY`. Atoms call::

    api.get_service(SERVICE_KEY).register_check(check)

from inside ``install(api, config)``. At each auditor firing the
adapter constructs a frozen :class:`CheckContext` and calls
:meth:`AuditCheckRegistry.run_all` to gather the advisory
:class:`~llmharness.schema.Finding` list.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Protocol, runtime_checkable

from ..schema import Edge, Event, Finding


@dataclass(frozen=True)
class CheckContext:
    """Frozen graph snapshot fed to every registered check at audit firing time.

    Both fields are tuples (immutable) so a check cannot mutate the
    snapshot the auditor sees, and so the same context can be safely
    shared across all registered checks in one firing.
    """

    events: tuple[Event, ...]
    edges: tuple[Edge, ...]


@runtime_checkable
class Check(Protocol):
    """Pure function: graph snapshot → list of findings.

    ``name`` is human-readable, used as :attr:`Finding.category` when
    a check does not set its own. ``__call__`` MUST be pure — no side
    effects, no I/O, deterministic on identical
    :class:`CheckContext`. Registered checks that raise are captured
    (see :meth:`AuditCheckRegistry.run_all`); the registry does not
    second-guess purity.
    """

    name: str

    def __call__(self, ctx: CheckContext) -> list[Finding]: ...


SERVICE_KEY: Final[str] = "llmharness.audit_registry"
"""ExtensionAPI service key for the per-session :class:`AuditCheckRegistry`."""


class AuditCheckRegistry:
    """Per-session registry of audit checks.

    Owned by the llmharness adapter; published to ExtensionAPI via
    ``api.set_service(SERVICE_KEY, registry)``. Atoms call
    ``api.get_service(SERVICE_KEY).register_check(check)`` from inside
    ``install(api, config)``.

    Idempotent on ``(check.name, id(check))`` — registering the same
    check twice is a no-op. Two distinct callables sharing a name (or
    one callable registered under two different names — though
    name comes from ``check.name``, so this only matters across
    distinct objects) both appear. ``run_all`` tolerates raising
    checks: the registry never propagates a check exception, but it
    does record it.
    """

    def __init__(self) -> None:
        self._checks: list[Check] = []
        self._seen: set[tuple[str, int]] = set()

    def register_check(self, check: Check) -> None:
        """Register ``check``. Idempotent on ``(check.name, id(check))``.

        Raises :class:`TypeError` immediately if ``check`` is not
        callable — registration is fail-fast so misconfigurations
        surface during ``install(api, config)`` rather than at the
        first audit firing.
        """

        if not callable(check):
            raise TypeError(
                f"register_check: expected a callable Check, got {type(check).__name__}"
            )
        name = getattr(check, "name", "")
        key = (str(name), id(check))
        if key in self._seen:
            return
        self._seen.add(key)
        self._checks.append(check)

    def registered_checks(self) -> tuple[Check, ...]:
        """Return registered checks in insertion order."""

        return tuple(self._checks)

    def run_all(self, ctx: CheckContext) -> tuple[list[Finding], dict[str, str]]:
        """Run every registered check; return (findings, errors-by-check-name).

        Findings are concatenated in registration order. A check that
        raises is captured: its ``name`` is recorded in the errors dict
        with ``str(exc)``; other checks still run. If two raising
        checks share a name, the latter overwrites the former — the
        errors dict is keyed by name for human readability, not by
        identity.
        """

        findings: list[Finding] = []
        errors: dict[str, str] = {}
        for check in self._checks:
            name = str(getattr(check, "name", ""))
            try:
                result = check(ctx)
            except Exception as exc:
                errors[name] = str(exc)
                continue
            findings.extend(result)
        return findings, errors


__all__ = [
    "SERVICE_KEY",
    "AuditCheckRegistry",
    "Check",
    "CheckContext",
]
