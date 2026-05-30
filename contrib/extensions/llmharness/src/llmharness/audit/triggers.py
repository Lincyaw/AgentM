"""Pluggable trigger registry for audit cadence decisions.

Replaces the rigid ``turn_count % audit_interval == 0`` cadence with a
composable, atom-based trigger system. Atoms register :class:`Trigger`
implementations via the :class:`TriggerRegistry` published under
:data:`SERVICE_KEY`; at each turn the runner builds a frozen
:class:`TriggerContext` and calls :meth:`TriggerRegistry.evaluate` to
decide whether the extractor and/or auditor should fire.

OR-semantics: any trigger firing causes the auditor to run. Each trigger
independently declares whether it ``requires_extractor`` (default
``True``); if at least one firing trigger does, the extractor also runs.

Follow the pattern of :mod:`llmharness.audit.registry` (the existing
``AuditCheckRegistry``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Protocol, runtime_checkable

SERVICE_KEY: Final[str] = "llmharness.audit_triggers"
"""ExtensionAPI service key for the per-session :class:`TriggerRegistry`."""


@dataclass(frozen=True)
class TriggerContext:
    """Frozen snapshot fed to every registered trigger at cadence-decision time.

    All fields are immutable (tuple / frozen) so a trigger cannot mutate
    runner state. The same context is shared across all registered triggers
    in one evaluation.
    """

    turn_count: int
    messages: tuple[Any, ...]
    latest_assistant_message: Any | None
    terminal_tool_called: str | None


@dataclass(frozen=True)
class TriggerDecision:
    """Outcome of one :meth:`Trigger.should_fire` call."""

    fire: bool
    reason: str = ""
    requires_extractor: bool = True


@runtime_checkable
class Trigger(Protocol):
    """Pure, synchronous predicate: context -> decision.

    ``name`` is human-readable, used in diagnostic output.
    ``should_fire`` MUST be pure -- no side effects, no I/O,
    deterministic on identical :class:`TriggerContext`.
    """

    name: str

    def should_fire(self, ctx: TriggerContext) -> TriggerDecision: ...


class TriggerRegistry:
    """Per-session registry of audit triggers.

    Owned by the llmharness adapter; published to ExtensionAPI via
    ``api.set_service(SERVICE_KEY, registry)``. Atoms call
    ``api.get_service(SERVICE_KEY).register_trigger(trigger)`` from
    inside ``install(api, config)``.

    Idempotent on ``(trigger.name, id(trigger))`` -- registering the
    same trigger twice is a no-op.
    """

    def __init__(self) -> None:
        self._triggers: list[Trigger] = []
        self._seen: set[tuple[str, int]] = set()

    def register_trigger(self, trigger: Trigger) -> None:
        """Register ``trigger``. Idempotent on ``(trigger.name, id(trigger))``."""
        name = getattr(trigger, "name", "")
        key = (str(name), id(trigger))
        if key in self._seen:
            return
        self._seen.add(key)
        self._triggers.append(trigger)

    def registered_triggers(self) -> tuple[Trigger, ...]:
        """Return registered triggers in insertion order."""
        return tuple(self._triggers)

    def evaluate(self, ctx: TriggerContext) -> tuple[bool, bool, list[str]]:
        """Evaluate all registered triggers against ``ctx``.

        Returns ``(auditor_due, extractor_due, reasons)``.

        OR-semantics: any trigger that fires causes ``auditor_due=True``.
        ``extractor_due`` is ``True`` iff at least one firing trigger has
        ``requires_extractor=True``.
        """
        auditor_due = False
        extractor_due = False
        reasons: list[str] = []
        for trigger in self._triggers:
            decision = trigger.should_fire(ctx)
            if decision.fire:
                auditor_due = True
                if decision.requires_extractor:
                    extractor_due = True
                reasons.append(
                    f"{trigger.name}: {decision.reason}" if decision.reason else trigger.name
                )
        return auditor_due, extractor_due, reasons


__all__ = [
    "SERVICE_KEY",
    "Trigger",
    "TriggerContext",
    "TriggerDecision",
    "TriggerRegistry",
]
