"""Compaction value types — public ABI for compaction-related atoms.

Pure dataclasses an atom (e.g. ``llm_compaction``) reads/constructs.
The compaction engine itself lives inside the ``llm_compaction`` atom;
these types are the only stable surface other code should depend on.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CompactionSettings:
    enabled: bool = True
    reserve_tokens: int = 16_384
    tool_result_max_chars: int = 2_000
    """Per-tool-result truncation cap used when rendering tool outputs into
    the summary prompt. Larger values preserve more verbatim tool detail at
    the cost of summary tokens."""


@dataclass(frozen=True, slots=True)
class CompactionDetails:
    read_files: list[str]
    modified_files: list[str]


@dataclass(frozen=True, slots=True)
class CompactionResult:
    summary: str
    covered_through_turn: int
    """The highest turn index folded into this summary. The next compaction
    starts from ``covered_through_turn + 1`` so already-summarized turns are
    not re-summarized (incremental chaining)."""
    tokens_before: int
    details: CompactionDetails


@dataclass(frozen=True, slots=True)
class ContextUsageEstimate:
    tokens: int
    usage_tokens: int
    trailing_tokens: int
    last_usage_index: int | None


@dataclass(frozen=True, slots=True)
class CompactionPrompts:
    """Prompt bodies threaded into the compaction engine by callers.

    Atoms resolve these via ``api.get_service("prompt_templates").get_prompt(...)`` and
    pass an instance into the compaction engine. Empty strings are legal —
    they represent the graceful-degradation path used when the
    ``compaction_prompts`` atom is not installed.
    """

    summarization_system: str
    update_summarization: str


__all__ = [
    "CompactionDetails",
    "CompactionPrompts",
    "CompactionResult",
    "CompactionSettings",
    "ContextUsageEstimate",
]
