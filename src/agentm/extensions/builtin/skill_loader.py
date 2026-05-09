"""Builtin skill loader atom."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agentm.core.abi.events import DiagnosticEvent
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
    requires=(),  # Leaf atom: consumes resource-discovery responses from any peer.
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
        response_owners: dict[int, str] = {}

        class _ResourceResponseObserver:
            def on_emit_start(self, channel: str, event: Any) -> None:
                del channel, event

            def on_handler_done(
                self,
                channel: str,
                handler: Any,
                value: Any,
                err: BaseException | None,
                duration_ns: int,
            ) -> None:
                del err, duration_ns
                if channel == ResourcesDiscoverEvent.CHANNEL and isinstance(value, dict):
                    response_owners[id(value)] = str(
                        getattr(handler, "_agentm_obs_owner", "<unknown>")
                    )

            def on_emit_end(
                self, channel: str, event: Any, results: list[Any]
            ) -> None:
                del channel, event, results

        unsubscribe = api.add_observer(_ResourceResponseObserver())
        try:
            responses = await api.events.emit(
                ResourcesDiscoverEvent.CHANNEL,
                ResourcesDiscoverEvent(cwd=api.cwd, reason="startup"),
            )
        finally:
            unsubscribe()
        contributed_skills: list[SkillRecord] = []
        allowed_response_keys = {"skill_paths", "extra_skills"}
        for response in responses:
            if not isinstance(response, dict):
                continue
            origin = response_owners.get(id(response), "<unknown>")
            for key in sorted(set(response) - allowed_response_keys):
                await api.events.emit(
                    DiagnosticEvent.CHANNEL,
                    DiagnosticEvent(
                        level="warning",
                        source="skill_loader",
                        message=(
                            f"ignored unknown ResourcesDiscoverEvent response key {key!r} "
                            f"from {origin}"
                        ),
                    ),
                )
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

    api.on(SessionReadyEvent.CHANNEL, _populate)
    api.on(BeforeAgentStartEvent.CHANNEL, _inject)
