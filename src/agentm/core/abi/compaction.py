"""Compaction value types — public ABI for compaction-related atoms.

Pure dataclasses an atom (e.g. ``llm_compaction``) reads/constructs.
The compaction *engine* (``compact``, ``prepare_compaction``,
``estimate_context_tokens``, ``should_compact``) stays in
``_internal/compaction`` and reaches atoms through
``ExtensionAPI.compaction``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CompactionSettings:
    enabled: bool = True
    reserve_tokens: int = 16_384
    keep_recent_tokens: int = 20_000


@dataclass(frozen=True, slots=True)
class CompactionDetails:
    read_files: list[str]
    modified_files: list[str]


@dataclass(frozen=True, slots=True)
class CompactionResult:
    summary: str
    first_kept_entry_id: str
    tokens_before: int
    details: CompactionDetails


@dataclass(frozen=True, slots=True)
class ContextUsageEstimate:
    tokens: int
    usage_tokens: int
    trailing_tokens: int
    last_usage_index: int | None


__all__ = [
    "CompactionDetails",
    "CompactionResult",
    "CompactionSettings",
    "ContextUsageEstimate",
]
