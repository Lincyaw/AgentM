"""Builtin skill loader atom."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agentm.core.abi.skill import SkillRecord
from agentm.extensions import ExtensionManifest
from agentm.harness.events import (
    BeforeAgentStartEvent,
    ResourcesDiscoverEvent,
    SessionReadyEvent,
)
from agentm.harness.extension import ExtensionAPI


MANIFEST = ExtensionManifest(
    name="skill_loader",
    description="Discover SKILL.md files and inject an <available_skills> index.",
    registers=(
        "event:before_agent_start",
        "event:resources_discover",
        "event:session_ready",
    ),
    config_schema={
        "type": "object",
        "properties": {
            "skill_paths": {"type": "array", "items": {"type": "string"}},
            "include_defaults": {"type": "boolean"},
            "inherit_claude": {"type": "boolean"},
        },
        "additionalProperties": True,
    },
)


async def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    include_defaults = bool(config.get("include_defaults", True))
    # ``inherit_claude`` defaults to ``include_defaults`` — when callers turn
    # off the standard agentm defaults (e.g. test isolation) they should not
    # silently pick up the real user's ``~/.claude/skills`` either.
    inherit_claude = bool(config.get("inherit_claude", include_defaults))
    configured_skill_paths = [str(path) for path in config.get("skill_paths", [])]
    cached_prompt_block = ""

    async def _populate(_: SessionReadyEvent) -> None:
        nonlocal cached_prompt_block
        discovered_paths = list(configured_skill_paths)
        if inherit_claude:
            # Auto-pick up Claude Code skill directories so users can reuse
            # the same `.claude/skills/<name>/SKILL.md` layout. Non-existent
            # paths are silently ignored by ``load_skills``.
            discovered_paths.append(str(Path.home() / ".claude" / "skills"))
            discovered_paths.append(str(Path(api.cwd) / ".claude" / "skills"))
        responses = await api.events.emit(
            "resources_discover",
            ResourcesDiscoverEvent(cwd=api.cwd, reason="startup"),
        )
        contributed_skills: list[SkillRecord] = []
        for response in responses:
            if not isinstance(response, dict):
                continue
            extra_paths = response.get("skill_paths")
            if isinstance(extra_paths, list):
                discovered_paths.extend(str(path) for path in extra_paths)
            extra_skills = response.get("extra_skills")
            if isinstance(extra_skills, list):
                for entry in extra_skills:
                    if isinstance(entry, SkillRecord):
                        contributed_skills.append(entry)
        skills, _diagnostics = api.skills.load_skills(
            cwd=api.cwd,
            agent_dir=str(Path.home() / ".agentm"),
            skill_paths=tuple(discovered_paths),
            include_defaults=include_defaults,
        )
        # Append peer-contributed records last so they don't shadow disk-based
        # skills with the same name.
        seen_names = {skill.name for skill in skills}
        for record in contributed_skills:
            if record.name in seen_names:
                continue
            seen_names.add(record.name)
            skills.append(record)
        cached_prompt_block = api.skills.format_skills_for_prompt(skills)

    def _inject(event: BeforeAgentStartEvent) -> dict[str, str] | None:
        if not cached_prompt_block:
            return None
        updated = f"{event.system or ''}{cached_prompt_block}"
        event.system = updated
        return {"system": updated}

    api.on("session_ready", _populate)
    api.on("before_agent_start", _inject)
