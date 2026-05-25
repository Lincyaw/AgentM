"""HarnessRunner package: settings, cumulative state, sidecar, runner.

The internal layout retains a single module — :mod:`.runner` — that
holds the :class:`HarnessRunner` class together with the dataclasses,
Protocols, and serialization helpers it relies on. The reorg spec
called for splitting these into ``settings.py``, ``cumulative_state.py``,
``sidecar.py`` and ``runner.py``; that split was deferred because every
candidate file kept a hard import dependency on a sibling (notably
``HarnessRunner`` instantiates ``ExtractorSettings.default`` /
``AuditorSettings.default`` and reads ``CumulativeAuditState``
internals while keeping ``SidecarWriter`` as a fixed collaborator), so
the file-level split would have produced four files with tight
cyclical imports rather than four coherent modules. Re-exporting here
preserves the spec's external surface: importers can write
``from llmharness.audit.runner import HarnessRunner`` (new) or
continue importing the same names via this package init.
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
    _flatten_assistant_blocks,
    _render_message_text,
    _serialize_block,
    _serialize_full_trajectory,
    _serialize_message_for_extractor,
    _window_is_non_trivial,
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
    "_flatten_assistant_blocks",
    "_render_message_text",
    "_serialize_block",
    "_serialize_full_trajectory",
    "_serialize_message_for_extractor",
    "_window_is_non_trivial",
]
