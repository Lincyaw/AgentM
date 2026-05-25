"""Cumulative event-sourced audit state + per-step result dataclass.

Taxonomy-only forwarder: the classes live in :mod:`.runner` next to
:class:`HarnessRunner`. The split was deferred — out of scope for the
pure-rename refactor; the dependency graph is a clean DAG and can be
done in a follow-up.
"""

from __future__ import annotations

from .runner import CumulativeAuditState, StepResult

__all__ = ["CumulativeAuditState", "StepResult"]
