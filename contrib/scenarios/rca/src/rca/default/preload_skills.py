"""Auto-inject skill content into the system prompt at session start.

Unlike ``load_skill`` (which requires the LLM to call a tool), this atom
reads the skill files at install time and appends their content to the
system prompt unconditionally. Use for skills that MUST be present
regardless of model compliance with prompt instructions.
"""

from __future__ import annotations

from pydantic import BaseModel

from agentm.core.abi import BeforeAgentStartEvent, ExtensionAPI
from agentm.extensions import ExtensionManifest

from rca import SCENARIO_ROOT

_SKILLS_DIR = SCENARIO_ROOT / "skills"


class PreloadSkillsConfig(BaseModel):
    model_config = {"extra": "forbid"}
    skills: list[str] = []


MANIFEST = ExtensionManifest(
    name="preload_skills",
    description="Inject skill file contents into the system prompt at session start.",
    registers=("event:before_agent_start",),
    config_schema=PreloadSkillsConfig,
)


def install(api: ExtensionAPI, config: PreloadSkillsConfig) -> None:
    sections: list[str] = []
    for skill_name in config.skills:
        # Try curated/diagnose-sql/<name>.md first, then flat <name>.md
        candidates = [
            _SKILLS_DIR / "curated" / "diagnose-sql" / f"{skill_name}.md",
            _SKILLS_DIR / f"{skill_name}.md",
        ]
        for path in candidates:
            if path.is_file():
                sections.append(path.read_text(encoding="utf-8").strip())
                break

    if not sections:
        return

    injected = "\n\n---\n\n".join(sections)

    def _inject(event: BeforeAgentStartEvent) -> None:
        current = event.system or ""
        updated = f"{current}\n\n{injected}"
        event.system = updated

    api.on(BeforeAgentStartEvent.CHANNEL, _inject)
