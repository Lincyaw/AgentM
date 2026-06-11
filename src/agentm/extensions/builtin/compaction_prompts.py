"""Builtin atom: defaults for the compaction subsystem (issue #76).

Owns two pieces the kernel deliberately does not:

1. The English prompt bodies the LLM-driven compaction engine uses
   (system persona, fresh summarization, incremental update, branch-summary
   instructions and preamble).
   Registered into the in-memory prompt registry resolved via
   ``api.get_service("prompt_templates").register_prompt``. Engine callers
   (``llm_compaction``, branch summarization) retrieve them via
   ``api.get_service("prompt_templates").get_prompt(<name>)``.

2. The default :class:`EntryMaterializer` implementations for the three
   kernel-defined session-entry types: ``message``, ``branch_summary``,
   ``compaction``. Registered into
   ``agentm.core.abi.session.ENTRY_MATERIALIZERS`` so the harness's
   ``build_session_context`` resolves materialization through the registry
   rather than branching on ``entry.type`` string literals.

The prompt registry is published by the ``prompt_templates`` atom under the
``"prompt_templates"`` service key; this atom resolves it via
``api.get_service("prompt_templates")``.

The §11 single-file contract holds: this atom does not import any other
atom and reaches services exclusively via :class:`ExtensionAPI`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from agentm.core.abi import (
    AgentMessage,
    AssistantContent,
    AssistantMessage,
    COMPACTION_PROMPTS,
    ENTRY_MATERIALIZERS,
    ENTRY_TYPE_BRANCH_SUMMARY,
    ENTRY_TYPE_COMPACTION,
    ENTRY_TYPE_MESSAGE,
    ExtensionAPI,
    SessionEntry,
    TextContent,
    ToolResultMessage,
    UserMessage,
)
from agentm.extensions import ExtensionManifest

# --- Prompt names -----------------------------------------------------------
#
# These are the canonical keys the compaction engine and ``llm_compaction``
# atom look up via ``api.get_service("prompt_templates").get_prompt``.
# Centralised here so atom-side and engine-side stay in agreement.

PROMPT_SUMMARIZATION_SYSTEM = "compaction.summarization_system"
PROMPT_SUMMARIZATION = "compaction.summarization"
PROMPT_UPDATE_SUMMARIZATION = "compaction.update_summarization"
PROMPT_BRANCH_SUMMARY = "compaction.branch_summary"
PROMPT_BRANCH_SUMMARY_PREAMBLE = "compaction.branch_summary_preamble"

# --- Default prompt bodies --------------------------------------------------

_SUMMARIZATION_SYSTEM = (
    "You are a context compaction assistant for an AI agent session. "
    "Your task is to read a conversation transcript and produce a structured "
    "checkpoint summary that will replace the original messages in context. "
    "The summary must contain everything a successor agent needs to continue "
    "the work without re-reading the original.\n\n"
    "Do NOT continue the conversation. Do NOT respond to any questions in "
    "the conversation. ONLY output the structured summary."
)

_SUMMARIZATION = """\
The messages above are a conversation to summarize. Create a structured \
context checkpoint that will replace the original messages. A successor \
agent will use ONLY this summary to continue the work.

Before writing, silently perform:
- **Contradiction check**: if user statements, tool results, or system \
constraints conflict, prefer the most recent reliable source and note the \
conflict.
- **Temporal ordering**: order events chronologically; latest state takes \
priority.
- **Hallucination control**: do not invent facts. Mark unverified \
information as UNVERIFIED.

Use this format:

## Session Identity
Record any visible session metadata (session_id, trace_id, scenario). The \
full original transcript is preserved in the observability log and can be \
searched via `agentm trace messages --session <session_id>` or \
`agentm trace tools --session <session_id>`.

## Goal
What is the user trying to accomplish? Include acceptance criteria if \
stated. Can be multiple items if the session covers different tasks.

## Constraints & Preferences
- Any constraints, preferences, or requirements the user stated
- Safety, privacy, or operational boundaries — preserve verbatim
- Rejected approaches and why they were rejected
- "(none)" if none were mentioned

## Progress
### Done
- [x] Completed tasks/changes — be specific about what was done and where

### In Progress
- [ ] Current work

### Blocked
- Issues preventing progress, if any

## Key Decisions
- **Decision**: Brief rationale, alternative considered, why rejected

## Files & Artifacts
### Read
- `path` — why it was read, key content or takeaway

### Modified
- `path` — what changed, key code snippets or config values

### Created
- `path` — purpose and contents

## Tool Trace Summary
For significant tool calls, preserve:
- Tool name, purpose, and key input parameters
- Result: conclusion, key data, paths, IDs, error codes
- Whether it succeeded or failed
- Impact on next steps

Discard redundant logs, verbose terminal output, and repeated search \
results. Preserve exact error messages, stack traces, exit codes, and \
command lines.

## Errors & Debugging
- Exact error messages and stack traces (verbatim)
- Failed approaches and why they failed — so the successor does not repeat \
them
- Successful fixes and what they addressed

## Next Steps
1. Ordered list of what should happen next

## Recovery Pointers
The original conversation is never deleted. List items where this summary \
is intentionally brief and full detail should be recovered on demand:
- Turn references: "Full implementation details at [Turn N]" — use \
`read_history` tool
- Files: "Re-read `path/to/file` for exact content"
- Previous sessions: "Search via `agentm trace messages --session <id>` \
or `agentm trace tools --session <id> --tool <name>`"

---

Rules:
- The conversation is grouped into turns marked [Turn N]. Cite originating \
turns for specific work (e.g. "implemented parser [Turn 14]"). Full detail \
of any turn can be recovered via the read_history tool, so prefer concise \
turn-tagged pointers over copying long verbatim content.
- Preserve exact file paths, function names, error messages, command \
outputs, variable names, and identifiers — specificity over brevity for \
these.
- For tool results: keep conclusions, key data, paths, IDs, error codes; \
drop verbose logs and terminal noise.
- Do not pad sections with generic filler. If a section has nothing, write \
"(none)" and move on."""

_UPDATE_SUMMARIZATION = """\
The messages above are NEW conversation messages to incorporate into the \
existing summary in <previous-summary> tags.

Before updating, silently check:
- **Contradictions**: if new messages contradict the previous summary, \
prefer the newer source and note the change.
- **Superseded info**: if a decision, status, or file state changed, \
update it — do not stack conflicting versions.

Rules for updating:
- PRESERVE all information from the previous summary that is still relevant
- ADD new progress, decisions, context, files, errors, tool results from \
the new messages
- UPDATE Progress: move items from "In Progress" to "Done" when completed; \
remove resolved blockers
- UPDATE "Next Steps" based on what was accomplished
- PRESERVE the Session Identity section from the previous summary
- PRESERVE exact file paths, function names, error messages, and identifiers
- PRESERVE any existing [Turn N] citations; tag newly summarized work with \
its turn marker the same way
- Do not repeat the early-conversation detail already captured in the \
previous summary — focus on what's new

Use the same section structure as the existing summary. Add new sections \
only when they become relevant.

The read_history tool can recover full detail of any cited turn. For \
previous sessions, `agentm trace` can search the full transcript."""

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
        # Materialized as a ``user`` message: under full-compress the summary
        # can be the trailing message in the rebuilt context, and a provider
        # completion request must not end on an assistant turn.
        payload = entry.payload if isinstance(entry.payload, dict) else {}
        summary = payload.get("summary")
        if not isinstance(summary, str) or not summary:
            return None
        return UserMessage(
            role="user",
            content=[TextContent(type="text", text=summary)],
            timestamp=entry.timestamp,
        )

# --- Manifest + install -----------------------------------------------------

class CompactionPromptsConfig(BaseModel):
    model_config = {"extra": "allow"}

    summarization_system: str | None = None
    summarization: str | None = None
    update_summarization: str | None = None
    branch_summary: str | None = None
    branch_summary_preamble: str | None = None

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
    config_schema=CompactionPromptsConfig,
    requires=("prompt_templates",),
    provides_role=(COMPACTION_PROMPTS,),
)

def install(api: ExtensionAPI, config: CompactionPromptsConfig) -> None:
    bodies: dict[str, str] = {
        PROMPT_SUMMARIZATION_SYSTEM: config.summarization_system or _SUMMARIZATION_SYSTEM,
        PROMPT_SUMMARIZATION: config.summarization or _SUMMARIZATION,
        PROMPT_UPDATE_SUMMARIZATION: config.update_summarization or _UPDATE_SUMMARIZATION,
        PROMPT_BRANCH_SUMMARY: config.branch_summary or _BRANCH_SUMMARY,
        PROMPT_BRANCH_SUMMARY_PREAMBLE: config.branch_summary_preamble or _BRANCH_SUMMARY_PREAMBLE,
    }
    registry = api.get_service("prompt_templates")
    if registry is None:
        raise RuntimeError(
            "compaction_prompts atom requires the prompt_templates service "
            "(install the prompt_templates atom first)"
        )
    for name, body in bodies.items():
        registry.register_prompt(name, body)

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
