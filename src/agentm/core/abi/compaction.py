"""Compaction subsystem types — shared contract between compaction atoms.

Two atoms collaborate on context-window management:

- ``compaction_prompts`` — provides prompt bodies and entry materializers
- ``llm_compaction`` — owns the compaction engine and scheduling

Both import these types from ``agentm.core.abi``; the §11 contract
forbids atom-to-atom imports, so shared types live here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# -- Entry types -------------------------------------------------------------

ENTRY_TYPE_MESSAGE = "message"
ENTRY_TYPE_COMPACTION = "compaction"
ENTRY_TYPE_BRANCH_SUMMARY = "branch_summary"

ENTRY_MATERIALIZERS: dict[str, Any] = {}


@dataclass(slots=True)
class SessionEntry:
    """A single entry in the session history branch.

    The session tree is append-only: compaction adds entries but never
    removes originals.
    """

    type: str = ""
    payload: Any = None
    timestamp: str = ""
    id: str = ""
    parent_id: str = ""


# -- Prompt key constants ----------------------------------------------------

PROMPT_SUMMARIZATION = "summarization"
PROMPT_SUMMARIZATION_SYSTEM = "summarization_system"
PROMPT_UPDATE_SUMMARIZATION = "update_summarization"
PROMPT_BRANCH_SUMMARY = "branch_summary"
PROMPT_BRANCH_SUMMARY_PREAMBLE = "branch_summary_preamble"


# -- Compaction prompts and settings -----------------------------------------

@dataclass(frozen=True, slots=True)
class CompactionPrompts:
    """Prompt bodies threaded into the compaction engine."""

    summarization_system: str = ""
    update_summarization: str = ""


@dataclass(frozen=True, slots=True)
class CompactionSettings:
    """Per-session compaction configuration."""

    enabled: bool = True
    reserve_tokens: int = 0
    tool_result_max_tokens: int = 0


@dataclass(frozen=True, slots=True)
class CompactionDetails:
    """File-operation detail attached to a compaction result."""

    read_files: list[str] = field(default_factory=list)
    modified_files: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class CompactionResult:
    """Outcome of a full-compress pass."""

    summary: str = ""
    covered_through_turn: int = 0
    tokens_before: int = 0
    measured_tokens_before: int = 0
    estimated_trailing_tokens_before: int = 0
    details: CompactionDetails = field(default_factory=CompactionDetails)


@dataclass(frozen=True, slots=True)
class ContextUsageSnapshot:
    """Point-in-time snapshot of context-window token usage."""

    tokens: int = 0
    measured_tokens: int = 0
    estimated_trailing_tokens: int = 0
    last_usage_index: int | None = None


__all__ = [
    "ENTRY_MATERIALIZERS",
    "ENTRY_TYPE_BRANCH_SUMMARY",
    "ENTRY_TYPE_COMPACTION",
    "ENTRY_TYPE_MESSAGE",
    "CompactionDetails",
    "CompactionPrompts",
    "CompactionResult",
    "CompactionSettings",
    "ContextUsageSnapshot",
    "PROMPT_BRANCH_SUMMARY",
    "PROMPT_BRANCH_SUMMARY_PREAMBLE",
    "PROMPT_SUMMARIZATION",
    "PROMPT_SUMMARIZATION_SYSTEM",
    "PROMPT_UPDATE_SUMMARIZATION",
    "SessionEntry",
]
