"""Agent Memory Scope — persistent per-agent memory across runs.

Provides scoped memory (agent / scenario / project) that is injected into
the system prompt via MemoryMiddleware. Memory entries are stored as
plain markdown MEMORY.md index files, one per scope.

Ref: designs/agent-memory.md
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from agentm.harness.middleware import MiddlewareBase, inject_into_system_message
from agentm.harness.types import LoopContext, Message

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Memory scope
# ---------------------------------------------------------------------------

_MEMORY_BEHAVIOR_HEADER = (
    "You have persistent memory that accumulates across runs. The entries below\n"
    "represent learnings, corrections, and preferences from previous sessions.\n"
    "Treat them as strong priors — follow them unless the current task explicitly\n"
    "contradicts them."
)

_MEMORY_BEHAVIOR_FOOTER = (
    "To save a new learning, use `memory_write`. To review or update existing\n"
    "entries, use `memory_read` and `memory_edit`."
)


class MemoryScope(StrEnum):
    """Scope levels for agent memory, ordered by priority (lowest first)."""

    AGENT = "agent"
    SCENARIO = "scenario"
    PROJECT = "project"


_SCOPE_PRIORITY_ORDER = (MemoryScope.PROJECT, MemoryScope.SCENARIO, MemoryScope.AGENT)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentMemoryConfig:
    """Configuration for the agent memory system."""

    enabled: bool = True
    memory_root: str = "./.agent-memory"
    scopes: list[MemoryScope] = field(
        default_factory=lambda: [MemoryScope.AGENT, MemoryScope.SCENARIO, MemoryScope.PROJECT]
    )
    max_prompt_entries: int = 50
    agent_identity: str = ""
    scenario_name: str = ""


# ---------------------------------------------------------------------------
# Directory resolution
# ---------------------------------------------------------------------------


def get_memory_dir(config: AgentMemoryConfig, scope: MemoryScope) -> Path:
    """Return the directory path for a given memory scope.

    - AGENT:    ``{memory_root}/agent/{agent_identity}/``
    - SCENARIO: ``{memory_root}/scenario/{scenario_name}/``
    - PROJECT:  ``{memory_root}/project/``
    """
    root = Path(config.memory_root)
    if scope == MemoryScope.AGENT:
        return root / "agent" / config.agent_identity
    if scope == MemoryScope.SCENARIO:
        return root / "scenario" / config.scenario_name
    # PROJECT
    return root / "project"


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _read_memory_entries(memory_md: Path, max_entries: int) -> list[str]:
    """Read entry lines from a MEMORY.md file, up to *max_entries*.

    An entry line starts with ``-`` (markdown list item).
    """
    if not memory_md.is_file():
        return []
    try:
        text = memory_md.read_text(encoding="utf-8").strip()
    except OSError:
        logger.warning("Failed to read memory file: %s", memory_md)
        return []
    if not text:
        return []
    entries = [line for line in text.splitlines() if line.strip().startswith("-")]
    return entries[:max_entries]


def load_agent_memory_prompt(config: AgentMemoryConfig) -> str:
    """Build the full ``<agent_memory>`` prompt fragment.

    Reads MEMORY.md from each configured scope (PROJECT first, then SCENARIO,
    then AGENT — so agent-scope entries appear last / highest priority).
    Returns an empty string when no memory content exists.
    """
    # Resolution order: PROJECT → SCENARIO → AGENT (lowest → highest priority)
    ordered_scopes = [s for s in _SCOPE_PRIORITY_ORDER if s in config.scopes]

    remaining = config.max_prompt_entries
    scope_blocks: list[str] = []

    for scope in ordered_scopes:
        memory_dir = get_memory_dir(config, scope)
        entries = _read_memory_entries(memory_dir / "MEMORY.md", max_entries=remaining)
        if not entries:
            continue

        remaining -= len(entries)
        entries_text = "\n".join(entries)

        if scope == MemoryScope.PROJECT:
            scope_blocks.append(f'<memory scope="project">\n{entries_text}\n</memory>')
        elif scope == MemoryScope.SCENARIO:
            scope_blocks.append(
                f'<memory scope="scenario" name="{config.scenario_name}">\n{entries_text}\n</memory>'
            )
        else:  # AGENT
            scope_blocks.append(
                f'<memory scope="agent" name="{config.agent_identity}">\n{entries_text}\n</memory>'
            )

        if remaining <= 0:
            break

    if not scope_blocks:
        return ""

    inner = "\n\n".join(scope_blocks)
    return (
        f"<agent_memory>\n"
        f"{_MEMORY_BEHAVIOR_HEADER}\n\n"
        f"{inner}\n\n"
        f"{_MEMORY_BEHAVIOR_FOOTER}\n"
        f"</agent_memory>"
    )


# ---------------------------------------------------------------------------
# MemoryMiddleware
# ---------------------------------------------------------------------------


class MemoryMiddleware(MiddlewareBase):
    """Injects agent memory prompt into the system message on each LLM call.

    The prompt is cached and only re-read from disk when the dirty flag is set.
    The dirty flag is initially True so the first call always loads from disk.
    """

    def __init__(self, config: AgentMemoryConfig) -> None:
        self._config = config
        self._cached_prompt: str = ""
        self._dirty: bool = True

    @property
    def dirty(self) -> bool:
        """Whether the cached prompt needs to be refreshed."""
        return self._dirty

    def mark_dirty(self) -> None:
        """Mark the cache as stale so the next LLM call re-reads MEMORY.md."""
        self._dirty = True

    def _ensure_prompt(self) -> str:
        """Return the cached prompt, refreshing from disk if dirty."""
        if self._dirty:
            self._cached_prompt = load_agent_memory_prompt(self._config)
            self._dirty = False
        return self._cached_prompt

    async def on_llm_start(
        self, messages: list[Message], ctx: LoopContext
    ) -> list[Message]:
        prompt = self._ensure_prompt()
        if not prompt:
            return messages
        return inject_into_system_message(messages, prompt)
