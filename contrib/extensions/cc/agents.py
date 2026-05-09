"""Claude Code persona discovery for named sub-agent dispatch."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agentm.core.lib.available_agents import available_agents_block
from agentm.extensions import ExtensionManifest
from agentm.harness.events import (
    BeforeAgentStartEvent,
    ResourcesDiscoverEvent,
    SessionReadyEvent,
)
from agentm.harness.extension import ExtensionAPI

from ._md_skills import AgentRecord, parse_md_agent_records


MANIFEST = ExtensionManifest(
    name="agents",
    description=(
        "Discover Claude Code persona markdown files and inject a sub-agent "
        "availability system-prompt block."
    ),
    registers=(
        "event:session_ready",
        "event:before_agent_start",
    ),
    config_schema={
        "type": "object",
        "properties": {
            "agent_paths": {"type": "array", "items": {"type": "string"}},
            "inherit_claude": {"type": "boolean"},
        },
        "additionalProperties": True,
    },
    tier=2,
)


def _default_roots(cwd: str) -> list[Path]:
    return [
        Path.home() / ".claude" / "agents",
        Path(cwd) / ".claude" / "agents",
    ]


def _format_block(agents: list[AgentRecord]) -> str:
    block = available_agents_block(agents)
    if not block:
        return ""
    return "\n\n".join(
        (
            "The following agents (personas) can be dispatched via the "
            "`dispatch_agent` tool. To launch one, first `read` its persona "
            "file, then prepend that text to the `prompt` argument of "
            "`dispatch_agent`.",
            block,
        )
    )


async def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    inherit_claude = bool(config.get("inherit_claude", True))
    configured_paths = [Path(str(p)) for p in config.get("agent_paths", [])]
    cached_block = ""

    async def _populate(_: SessionReadyEvent) -> None:
        nonlocal cached_block
        roots: list[Path] = list(configured_paths)
        if inherit_claude:
            roots.extend(_default_roots(api.cwd))

        responses = await api.events.emit(
            ResourcesDiscoverEvent.CHANNEL,
            ResourcesDiscoverEvent(cwd=api.cwd, reason="startup"),
        )
        for response in responses:
            if not isinstance(response, dict):
                continue
            extra = response.get("agent_paths")
            if isinstance(extra, list):
                roots.extend(Path(str(p)) for p in extra)

        seen: set[str] = set()
        agents: list[AgentRecord] = []
        for root in roots:
            for record in parse_md_agent_records(root):
                if record.name in seen:
                    continue
                seen.add(record.name)
                agents.append(record)
        cached_block = _format_block(agents)

    def _inject(event: BeforeAgentStartEvent) -> dict[str, str] | None:
        if not cached_block:
            return None
        updated = f"{event.system or ''}{cached_block}"
        event.system = updated
        return {"system": updated}

    api.on(SessionReadyEvent.CHANNEL, _populate)
    api.on(BeforeAgentStartEvent.CHANNEL, _inject)


__all__ = ("MANIFEST", "install")
