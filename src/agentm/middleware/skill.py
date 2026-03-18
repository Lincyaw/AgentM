"""Skill injection middleware — config-driven skill descriptions in system prompt.

Reads skill descriptions from a MarkdownVault at init time and injects them
into the SystemMessage before each LLM call.

Self-contained: no dependency on ``scenarios/`` modules.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import SystemMessage

from agentm.middleware import AgentMMiddleware
from agentm.tools.vault.store import MarkdownVault

logger = logging.getLogger(__name__)


class SkillMiddleware(AgentMMiddleware):
    """Pre-model middleware that appends skill descriptions to the system prompt.

    At init time, reads each skill's frontmatter from the vault and caches
    ``{path, name, description}``.  At each ``before_model`` call, appends
    an ``## Available Skills`` section to the first ``SystemMessage``.

    Each middleware instance is independent — different agents can configure
    different skill sets.
    """

    def __init__(self, vault: MarkdownVault, skill_paths: list[str]) -> None:
        self._skill_descriptions: list[dict[str, str]] = []
        self._injected_once = False
        for path in skill_paths:
            note = vault.read(path)
            if note is None:
                logger.warning("Skill not found in vault, skipping: %s", path)
                continue
            fm = note["frontmatter"]
            self._skill_descriptions.append({
                "path": path,
                "name": fm.get("name", path),
                "description": fm.get("description", ""),
            })
        logger.info(
            "SkillMiddleware initialized: %d/%d skills loaded from %s",
            len(self._skill_descriptions),
            len(skill_paths),
            vault._vault_dir,
        )

    @property
    def skill_count(self) -> int:
        """Number of skills successfully loaded."""
        return len(self._skill_descriptions)

    def _build_skills_section(self) -> str:
        """Build the ``## Available Skills`` markdown section."""
        if not self._skill_descriptions:
            return ""
        lines = [
            "## Available Skills",
            "",
            "Skills are domain-specific instructions stored in the knowledge "
            "vault as notes with `type: skill`.",
            "",
            "### How to use skills",
            "",
            "1. **Browse**: Review the skill list below. Each entry shows a "
            "short description to help you decide relevance.",
            "2. **Load**: Call `vault_read` with the skill path "
            "(e.g. `vault_read(path=\"skill/diagnose-sql\")`) to load its "
            "full instructions.",
            "3. **Follow sub-skills**: A loaded skill may reference "
            "**sub-skills** via `[[wikilinks]]` — more specific instruction "
            "sets nested under it (e.g. `[[skill/diagnose-sql/traces]]`). "
            "Load them on demand with `vault_read` when you need deeper "
            "guidance for a particular area.",
            "4. **Search**: If you need knowledge beyond the listed skills, "
            "use `vault_search` to find relevant notes by keyword or "
            "semantic similarity. Filter with `{\"type\": \"skill\"}` to "
            "restrict results to skills.",
            "5. **Explore**: Use `vault_list` to browse the vault directory "
            "structure and discover related notes.",
            "6. **Apply**: Follow the loaded instructions for your current task.",
            "",
            "### Skill list",
            "",
        ]
        for s in self._skill_descriptions:
            lines.append(
                f"- **{s['name']}** (`{s['path']}`): {s['description']}"
            )
        return "\n".join(lines)

    def before_model(self, state: dict[str, Any]) -> dict[str, Any] | None:
        if not self._skill_descriptions:
            return None

        skills_section = self._build_skills_section()

        messages = state.get("llm_input_messages") or state.get("messages", [])
        new_messages: list[Any] = []
        injected = False

        for msg in messages:
            if not injected and isinstance(msg, SystemMessage):
                new_content = str(msg.content) + "\n\n" + skills_section
                new_messages.append(SystemMessage(content=new_content))
                injected = True
                if not self._injected_once:
                    self._injected_once = True
                    logger.debug(
                        "Skill descriptions injected into SystemMessage (%d skills, +%d chars)",
                        len(self._skill_descriptions),
                        len(skills_section),
                    )
                injected = True
            else:
                new_messages.append(msg)

        if not injected:
            # No SystemMessage found — prepend one with just the skills section
            new_messages.insert(0, SystemMessage(content=skills_section))

        key = "llm_input_messages" if "llm_input_messages" in state else "messages"
        return {key: new_messages}
