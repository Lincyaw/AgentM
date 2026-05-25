"""Cumulative event-sourced audit state + per-step result dataclass.

Taxonomy-only forwarder: the classes live in :mod:`.runner` next to
:class:`HarnessRunner` for the same reason as
:mod:`.settings` — the runner reads ``CumulativeAuditState``
internals directly and constructs ``StepResult`` instances on every
firing.
"""

from __future__ import annotations

from .runner import CumulativeAuditState, StepResult

__all__ = ["CumulativeAuditState", "StepResult"]
