"""Cognitive-audit subpackages.

Public re-exports:
- Schema: :class:`~llmharness.schema.Edge`, :class:`~llmharness.schema.EdgeKind`,
  :class:`~llmharness.schema.Finding`.
- Registry: :class:`CheckContext`, :class:`Check`,
  :class:`AuditCheckRegistry`, :data:`SERVICE_KEY`.

Phase-specific entry points still live under :mod:`.extractor` /
:mod:`.auditor`.
"""

from __future__ import annotations

from ..schema import Edge, EdgeKind, Finding
from .registry import SERVICE_KEY, AuditCheckRegistry, Check, CheckContext

__all__ = [
    "SERVICE_KEY",
    "AuditCheckRegistry",
    "Check",
    "CheckContext",
    "Edge",
    "EdgeKind",
    "Finding",
]
