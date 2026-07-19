"""Builtin atom: default prompt bodies for the compaction subsystem.

Owns the English prompt bodies the LLM-driven compaction engine can use
(system persona, fresh summarization, incremental update, branch-summary
instructions and preamble). They are registered into the in-memory prompt
registry published by the ``prompt_templates`` atom under the
``"prompt_templates"`` service key, retrieved via
``api.services.get("prompt_templates").get_prompt(<name>)``.

TODO(migration): on ``main`` this atom also registered default
``EntryMaterializer`` implementations into
``agentm.core.abi.session.ENTRY_MATERIALIZERS`` for the ``message`` /
``branch_summary`` / ``compaction`` session-entry types. That registry and the
``SessionEntry`` / ``ENTRY_TYPE_*`` / ``COMPACTION_PROMPTS`` symbols do not
exist on this branch — the current compaction engine (``llm_compaction``) is
config-driven and reads its summary prompt from ``LlmCompactionConfig``, not
from this registry. The materializer registration is therefore dropped, and
these prompt bodies are published but not yet consumed by the engine. Kept so
the content survives for whoever re-wires a registry-backed compaction path.

The single-file contract holds: this atom does not import any other atom and
reaches services exclusively via :class:`AtomAPI`.
"""

from __future__ import annotations

from pydantic import BaseModel

from agentm.core.abi import AtomAPI, AtomInstallPriority
from agentm.extensions import ExtensionManifest

# Locally owned service key + prompt names (single-file contract: no
# atom-to-atom imports). ``prompt_templates`` publishes the same service key.
PROMPT_TEMPLATES_SERVICE = "prompt_templates"
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
stated. Can be multiple items if the session covers different tasks. \
If the conversation contains a structured goal condition (from a \
"[Goal not met ...]" message or a goal-checker injection), preserve \
the goal statement and verification method verbatim — the successor \
agent needs these to know what to work toward and how to verify it.

## Task Specification
All user messages from the compacted region are preserved verbatim \
alongside this summary — the successor agent will see them as separate \
messages. Do NOT reproduce or paraphrase user message content here. \
Instead, note only which user messages contained task specs and any \
key constraints the assistant derived from them that are not obvious \
from the user text alone.

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
- Test execution results: which tests passed, which failed, with exact \
output (exit codes, pass/fail markers). The successor needs this to know \
what still needs fixing vs. what is already working.

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
- All user messages are preserved verbatim alongside this summary — do \
NOT reproduce or paraphrase user message content
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
        "Default English prompt bodies for the LLM-driven compaction engine, "
        "published into the prompt_templates registry."
    ),
    registers=(),
    config_schema=CompactionPromptsConfig,
    requires=("prompt_templates",),
    priority=AtomInstallPriority.CONTEXT,
)


class _CompactionPromptsRuntime:
    def __init__(self, api: AtomAPI, config: CompactionPromptsConfig) -> None:
        self._api = api
        self._prompt_bodies = {
            PROMPT_SUMMARIZATION_SYSTEM: config.summarization_system
            or _SUMMARIZATION_SYSTEM,
            PROMPT_SUMMARIZATION: config.summarization or _SUMMARIZATION,
            PROMPT_UPDATE_SUMMARIZATION: config.update_summarization
            or _UPDATE_SUMMARIZATION,
            PROMPT_BRANCH_SUMMARY: config.branch_summary or _BRANCH_SUMMARY,
            PROMPT_BRANCH_SUMMARY_PREAMBLE: config.branch_summary_preamble
            or _BRANCH_SUMMARY_PREAMBLE,
        }

    def install(self) -> None:
        registry = self._api.services.get(PROMPT_TEMPLATES_SERVICE)
        if registry is None:
            raise RuntimeError(
                "compaction_prompts atom requires the prompt_templates service "
                "(install the prompt_templates atom first)"
            )
        for name, body in self._prompt_bodies.items():
            registry.register_prompt(name, body)
        # TODO(migration): main also registered EntryMaterializer instances for
        # the message / branch_summary / compaction entry types. That registry
        # (ENTRY_MATERIALIZERS) and the SessionEntry type do not exist on this
        # branch, so no materializers are registered.


def install(api: AtomAPI, config: CompactionPromptsConfig) -> None:
    _CompactionPromptsRuntime(api, config).install()
