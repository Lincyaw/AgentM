"""LLM-as-harness: cognitive-audit AgentM extension."""

from __future__ import annotations

from .offline import (
    OfflineRunResult,
    SurfaceFiring,
    audit_pipeline_over_trajectory,
    offline_audit,
)
from .schema import (
    Reminder,
    Verdict,
)

# Backward-compat aliases from the deleted root offline.py
SurfacePoint = SurfaceFiring
"""Alias for :class:`SurfaceFiring` (old name from root ``offline.py``)."""

OfflineAuditResult = OfflineRunResult
"""Alias for :class:`OfflineRunResult` (old name from root ``offline.py``)."""


__all__ = [
    "OfflineAuditResult",
    "OfflineRunResult",
    "Reminder",
    "SurfaceFiring",
    "SurfacePoint",
    "Verdict",
    "audit_pipeline_over_trajectory",
    "offline_audit",
]
