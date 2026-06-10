"""Claude Code command markdown as slash commands and model skills."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from agentm.core.abi.skill import SkillRecord
from agentm.core.abi import BusPriority
from agentm.extensions import ExtensionManifest
from agentm.core.abi.events import ResourcesDiscoverEvent, SessionReadyEvent
from agentm.core.abi.extension import CommandSpec, ExtensionAPI

from ._md_skills import parse_md_command_records, parse_md_skill_records


class CommandsConfig(BaseModel):
    model_config = {"extra": "allow"}

    command_paths: list[str] = []
    inherit_claude: bool = True


MANIFEST = ExtensionManifest(
    name="commands",
    description=(
        "Register Claude Code command markdown files as slash commands and "
        "surface them to the model via the skills block."
    ),
    registers=("event:session_ready", "event:resources_discover"),
    config_schema=CommandsConfig,
    tier=2,
)


def _default_roots(cwd: str) -> list[Path]:
    return [
        Path.home() / ".claude" / "commands",
        Path(cwd) / ".claude" / "commands",
    ]


def _make_handler(body: str):  # type: ignore[no-untyped-def]
    def _handler(rest: str, api: ExtensionAPI) -> None:
        rest = rest.strip()
        message = f"{body}\n\n{rest}" if rest else body
        api.send_user_message(message)

    return _handler


def _dedupe_records(records: list[SkillRecord]) -> list[SkillRecord]:
    out: list[SkillRecord] = []
    seen: set[str] = set()
    for record in records:
        if record.name in seen:
            continue
        seen.add(record.name)
        out.append(record)
    return out


async def install(api: ExtensionAPI, config: CommandsConfig) -> None:
    inherit_claude = config.inherit_claude
    configured_paths = [Path(p) for p in config.command_paths]
    discover_in_progress = False
    registered_commands: set[str] = set()

    def _base_roots() -> list[Path]:
        roots = list(configured_paths)
        if inherit_claude:
            roots.extend(_default_roots(api.cwd))
        return roots

    async def _plugin_command_roots(reason: Literal["startup", "reload"]) -> list[Path]:
        nonlocal discover_in_progress
        if discover_in_progress:
            return []
        discover_in_progress = True
        try:
            responses = await api.events.emit(
                ResourcesDiscoverEvent.CHANNEL,
                ResourcesDiscoverEvent(cwd=api.cwd, reason=reason),
            )
        finally:
            discover_in_progress = False
        roots: list[Path] = []
        for response in responses:
            if not isinstance(response, dict):
                continue
            command_paths = response.get("command_paths")
            if isinstance(command_paths, list):
                roots.extend(Path(str(p)) for p in command_paths)
        return roots

    async def _all_roots(reason: Literal["startup", "reload"]) -> list[Path]:
        return [*_base_roots(), *await _plugin_command_roots(reason)]

    async def _populate(_: SessionReadyEvent) -> None:
        for root in await _all_roots("startup"):
            for record in parse_md_command_records(root):
                name = record.skill.name
                if name in registered_commands:
                    continue
                api.register_command(
                    name,
                    CommandSpec(
                        description=record.skill.description,
                        handler=_make_handler(record.body),
                    ),
                )
                registered_commands.add(name)

    async def _contribute(event: ResourcesDiscoverEvent) -> dict[str, Any] | None:
        if discover_in_progress:
            return None
        records: list[SkillRecord] = []
        for root in await _all_roots(event.reason):
            records.extend(parse_md_skill_records(root))
        deduped = _dedupe_records(records)
        if not deduped:
            return None
        return {"extra_skills": deduped}

    api.on(SessionReadyEvent.CHANNEL, _populate)
    api.on(ResourcesDiscoverEvent.CHANNEL, _contribute, priority=BusPriority.POST)


__all__ = ("MANIFEST", "install")
