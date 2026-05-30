"""HarnessRunner package.

A single module — :mod:`.runner` — holds :class:`HarnessRunner` together
with the settings dataclasses, cumulative state, Protocols, and
serialization helpers it depends on (a one-way DAG: the runner depends on
them, never the reverse). This init re-exports the public surface so
importers can write ``from llmharness.audit.runner import HarnessRunner``.

``_flatten_assistant_blocks`` / ``_serialize_full_trajectory`` are
re-exported because the live/offline seams and the adapter import them by
this path; the remaining serialization helpers stay private to
:mod:`.runner`.
"""

from __future__ import annotations

from .runner import (
    AUDIT_REGISTRY_SERVICE_KEY,
    FINALIZE_EXTRACTION_TOOL_NAME,
    SUBMIT_VERDICT_TOOL_NAME,
    AuditorChildResult,
    AuditorOutputError,
    AuditorSettings,
    ChildRunner,
    CumulativeAuditState,
    ExtractorSettings,
    ExtractorSpawnError,
    HarnessRunner,
    OpSink,
    RawExtractorOutput,
    RawVerdictOutput,
    SidecarWriter,
    StepResult,
    TriggerRegistry,
    _flatten_assistant_blocks,
    _serialize_full_trajectory,
)

__all__ = [
    "AUDIT_REGISTRY_SERVICE_KEY",
    "FINALIZE_EXTRACTION_TOOL_NAME",
    "SUBMIT_VERDICT_TOOL_NAME",
    "AuditorChildResult",
    "AuditorOutputError",
    "AuditorSettings",
    "ChildRunner",
    "CumulativeAuditState",
    "ExtractorSettings",
    "ExtractorSpawnError",
    "HarnessRunner",
    "OpSink",
    "RawExtractorOutput",
    "RawVerdictOutput",
    "SidecarWriter",
    "StepResult",
    "TriggerRegistry",
    "_flatten_assistant_blocks",
    "_serialize_full_trajectory",
]
