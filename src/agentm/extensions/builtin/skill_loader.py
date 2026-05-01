"""Builtin skill loader atom."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agentm.core.skills import format_skills_for_prompt, load_skills
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
        },
        "additionalProperties": True,
    },
)


async def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    include_defaults = bool(config.get("include_defaults", True))
    configured_skill_paths = [str(path) for path in config.get("skill_paths", [])]
    cached_prompt_block = ""

    async def _populate(_: SessionReadyEvent) -> None:
        nonlocal cached_prompt_block
        discovered_paths = list(configured_skill_paths)
        responses = await api.events.emit(
            "resources_discover",
            ResourcesDiscoverEvent(cwd=api.cwd, reason="startup"),
        )
        for response in responses:
            if not isinstance(response, dict):
                continue
            extra_paths = response.get("skill_paths")
            if not isinstance(extra_paths, list):
                continue
            discovered_paths.extend(str(path) for path in extra_paths)
        skills, _diagnostics = load_skills(
            cwd=api.cwd,
            agent_dir=str(Path.home() / ".agentm"),
            skill_paths=tuple(discovered_paths),
            include_defaults=include_defaults,
        )
        cached_prompt_block = format_skills_for_prompt(skills)

    def _inject(event: BeforeAgentStartEvent) -> dict[str, str] | None:
        if not cached_prompt_block:
            return None
        updated = f"{event.system or ''}{cached_prompt_block}"
        event.system = updated
        return {"system": updated}

    api.on("session_ready", _populate)
    api.on("before_agent_start", _inject)
