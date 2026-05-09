"""Builtin atom: defaults for the compaction subsystem (issue #76).

Owns two pieces the kernel deliberately does not:

1. The English prompt bodies the LLM-driven compaction engine uses
   (system persona, fresh summarization, incremental update, split-turn
   prefix summarization, branch-summary instructions and preamble).
   Registered into the in-memory prompt registry exposed by
   ``api.prompt_templates.register_prompt``. Engine callers
   (``llm_compaction``, branch summarization) retrieve them via
   ``api.prompt_templates.get_prompt(<name>)``.

2. The default :class:`EntryMaterializer` implementations for the three
   kernel-defined session-entry types: ``message``, ``branch_summary``,
   ``compaction``. Registered into
   ``agentm.core.abi.session.ENTRY_MATERIALIZERS`` so both the compaction
   engine's ``get_message_from_entry`` and the harness's
   ``build_session_context`` resolve materialization through the registry
   rather than branching on ``entry.type`` string literals.

The §11 single-file contract holds: this atom does not import any other
atom and reaches services exclusively via :class:`ExtensionAPI`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agentm.core.abi import (
    AgentMessage,
    AssistantMessage,
    AssistantContent,
    TextContent,
    ToolResultMessage,
    UserMessage,
)
from agentm.core.abi.session import (
    ENTRY_MATERIALIZERS,
    ENTRY_TYPE_BRANCH_SUMMARY,
    ENTRY_TYPE_COMPACTION,
    ENTRY_TYPE_MESSAGE,
    SessionEntry,
)
from agentm.extensions import ExtensionManifest
from agentm.harness.extension import ExtensionAPI


# --- Prompt names -----------------------------------------------------------
#
# These are the canonical keys the compaction engine and ``llm_compaction``
# atom look up via ``api.prompt_templates.get_prompt``. Centralised here so
# atom-side and engine-side stay in agreement.

PROMPT_SUMMARIZATION_SYSTEM = "compaction.summarization_system"
PROMPT_SUMMARIZATION = "compaction.summarization"
PROMPT_UPDATE_SUMMARIZATION = "compaction.update_summarization"
PROMPT_TURN_PREFIX_SUMMARIZATION = "compaction.turn_prefix_summarization"
PROMPT_BRANCH_SUMMARY = "compaction.branch_summary"
PROMPT_BRANCH_SUMMARY_PREAMBLE = "compaction.branch_summary_preamble"


# --- Default prompt bodies --------------------------------------------------

_SUMMARIZATION_SYSTEM = (
    "You are a context summarization assistant. Your task is to read a "
    "conversation and produce a structured summary following the exact "
    "format specified.\n\n"
    "Do NOT continue the conversation. Do NOT respond to any questions in "
    "the conversation. ONLY output the structured summary."
)

_SUMMARIZATION = """The messages above are a conversation to summarize. Create a structured context checkpoint summary that another LLM will use to continue the work.

Use this EXACT format:

## Goal
[What is the user trying to accomplish? Can be multiple items if the session covers different tasks.]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned by user]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Current work]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [Ordered list of what should happen next]

## Critical Context
- [Any data, examples, or references needed to continue]
- [Or "(none)" if not applicable]

Keep each section concise. Preserve exact file paths, function names, and error messages."""

_UPDATE_SUMMARIZATION = """The messages above are NEW conversation messages to incorporate into the existing summary provided in <previous-summary> tags.

Update the existing structured summary with new information. RULES:
- PRESERVE all existing information from the previous summary
- ADD new progress, decisions, and context from the new messages
- UPDATE the Progress section: move items from \"In Progress\" to \"Done\" when completed
- UPDATE \"Next Steps\" based on what was accomplished
- PRESERVE exact file paths, function names, and error messages
- If something is no longer relevant, you may remove it

Use this EXACT format:

## Goal
[Preserve existing goals, add new ones if the task expanded]

## Constraints & Preferences
- [Preserve existing, add new ones discovered]

## Progress
### Done
- [x] [Include previously done items AND newly completed items]

### In Progress
- [ ] [Current work - update based on progress]

### Blocked
- [Current blockers - remove if resolved]

## Key Decisions
- **[Decision]**: [Brief rationale] (preserve all previous, add new)

## Next Steps
1. [Update based on current state]

## Critical Context
- [Preserve important context, add new if needed]

Keep each section concise. Preserve exact file paths, function names, and error messages."""

_TURN_PREFIX_SUMMARIZATION = """This is the PREFIX of a turn that was too large to keep. The SUFFIX (recent work) is retained.

Summarize the prefix to provide context for the retained suffix:

## Original Request
[What did the user ask for in this turn?]

## Early Progress
- [Key decisions and work done in the prefix]

## Context for Suffix
- [Information needed to understand the retained recent work]

Be concise. Focus on what's needed to understand the kept suffix."""

_BRANCH_SUMMARY = """Create a structured summary of this conversation branch for context when returning later.

Use this EXACT format:

## Goal
[What was the user trying to accomplish in this branch?]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Work that was started but not finished]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [What should happen next to continue this work]

Keep each section concise. Preserve exact file paths, function names, and error messages."""

_BRANCH_SUMMARY_PREAMBLE = (
    "The user explored a different conversation branch before returning here.\n"
    "Summary of that exploration:\n\n"
)


# --- EntryMaterializer implementations --------------------------------------


@dataclass(frozen=True, slots=True)
class _MessageEntryMaterializer:
    def to_message(self, entry: SessionEntry) -> AgentMessage | None:
        if isinstance(entry.payload, (UserMessage, AssistantMessage, ToolResultMessage)):
            return entry.payload
        return None


@dataclass(frozen=True, slots=True)
class _BranchSummaryEntryMaterializer:
    def to_message(self, entry: SessionEntry) -> AgentMessage | None:
        payload = entry.payload if isinstance(entry.payload, dict) else {}
        summary = payload.get("summary")
        if not isinstance(summary, str) or not summary:
            return None
        content: list[AssistantContent] = [
            TextContent(type="text", text=f"Branch summary: {summary}")
        ]
        return AssistantMessage(
            role="assistant",
            content=content,
            timestamp=entry.timestamp,
            stop_reason="end_turn",
        )


@dataclass(frozen=True, slots=True)
class _CompactionEntryMaterializer:
    def to_message(self, entry: SessionEntry) -> AgentMessage | None:
        payload = entry.payload if isinstance(entry.payload, dict) else {}
        summary = payload.get("summary")
        if not isinstance(summary, str) or not summary:
            return None
        content: list[AssistantContent] = [TextContent(type="text", text=summary)]
        return AssistantMessage(
            role="assistant",
            content=content,
            timestamp=entry.timestamp,
            stop_reason="end_turn",
        )


# --- Manifest + install -----------------------------------------------------


MANIFEST = ExtensionManifest(
    name="compaction_prompts",
    description=(
        "Default English prompt bodies and entry materializers used by the "
        "LLM-driven compaction engine."
    ),
    # Registers nothing on the kernel registries (no tools, events, commands,
    # providers, or renderers). All registration goes through the
    # PromptTemplatesService and the ENTRY_MATERIALIZERS module dict.
    registers=(),
    config_schema={
        "type": "object",
        "properties": {
            "summarization_system": {"type": "string"},
            "summarization": {"type": "string"},
            "update_summarization": {"type": "string"},
            "turn_prefix_summarization": {"type": "string"},
            "branch_summary": {"type": "string"},
            "branch_summary_preamble": {"type": "string"},
        },
        "additionalProperties": True,
    },
    requires=(),  # Leaf atom: publishes prompt templates through api.prompt_templates.
)


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    bodies: dict[str, str] = {
        PROMPT_SUMMARIZATION_SYSTEM: _override(config, "summarization_system", _SUMMARIZATION_SYSTEM),
        PROMPT_SUMMARIZATION: _override(config, "summarization", _SUMMARIZATION),
        PROMPT_UPDATE_SUMMARIZATION: _override(
            config, "update_summarization", _UPDATE_SUMMARIZATION
        ),
        PROMPT_TURN_PREFIX_SUMMARIZATION: _override(
            config, "turn_prefix_summarization", _TURN_PREFIX_SUMMARIZATION
        ),
        PROMPT_BRANCH_SUMMARY: _override(config, "branch_summary", _BRANCH_SUMMARY),
        PROMPT_BRANCH_SUMMARY_PREAMBLE: _override(
            config, "branch_summary_preamble", _BRANCH_SUMMARY_PREAMBLE
        ),
    }
    for name, body in bodies.items():
        api.prompt_templates.register_prompt(name, body)

    # Module-level mutation: the registry on ``core.abi.session`` is shared
    # across the process. Multiple sessions installing the atom is harmless
    # — the materializer instances are stateless and idempotent.
    ENTRY_MATERIALIZERS[ENTRY_TYPE_MESSAGE] = _MessageEntryMaterializer()
    ENTRY_MATERIALIZERS[ENTRY_TYPE_BRANCH_SUMMARY] = _BranchSummaryEntryMaterializer()
    ENTRY_MATERIALIZERS[ENTRY_TYPE_COMPACTION] = _CompactionEntryMaterializer()


def _override(config: dict[str, Any], key: str, default: str) -> str:
    candidate = config.get(key)
    if isinstance(candidate, str) and candidate:
        return candidate
    return default
