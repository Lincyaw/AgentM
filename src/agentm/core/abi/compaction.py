"""Compaction value types — public ABI for compaction-related atoms.

Pure dataclasses an atom (e.g. ``llm_compaction``) reads/constructs.
The compaction engine itself lives inside the ``llm_compaction`` atom;
these types are the only stable surface other code should depend on.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CompactionSettings:
    tool_result_max_tokens: int
    """Per-tool-result truncation cap used when rendering tool outputs into
    the summary prompt. Larger values preserve more verbatim tool detail at
    the cost of longer summary prompts."""

    enabled: bool = True
    reserve_tokens: int = 16_384


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
    """Provider usage plus tiktoken-estimated trailing tokens before compaction."""

    measured_tokens_before: int
    """Provider-reported tokens from the latest measured assistant turn."""

    estimated_trailing_tokens_before: int
    """Tiktoken-estimated tokens in messages after the latest provider usage."""

    details: CompactionDetails


@dataclass(frozen=True, slots=True)
class ContextUsageSnapshot:
    tokens: int
    measured_tokens: int
    estimated_trailing_tokens: int
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
    "ContextUsageSnapshot",
]
