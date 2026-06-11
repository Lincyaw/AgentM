"""Claude Code persona discovery for named sub-agent dispatch."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from pydantic import BaseModel

from agentm.extensions import ExtensionManifest
from agentm.core.abi import (
    BeforeAgentStartEvent,
    ExtensionAPI,
    ResourcesDiscoverEvent,
    SessionReadyEvent,
)

from ._md_skills import AgentRecord, parse_md_agent_records

# ---------------------------------------------------------------------------
# available_agents XML rendering — inlined from the former
# agentm.core.lib.available_agents (this extension is one of two consumers).
# ---------------------------------------------------------------------------

def _field(persona: Any, name: str, default: Any = "") -> Any:
    if isinstance(persona, Mapping):
        return persona.get(name, default)
    return getattr(persona, name, default)

def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()

def _tools(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
        return ", ".join(str(item).strip() for item in value if str(item).strip())
    return ""

def _agent_items(personas: Any) -> list[tuple[str, Any]]:
    if isinstance(personas, Mapping):
        return [(str(name), persona) for name, persona in sorted(personas.items())]
    items: list[tuple[str, Any]] = []
    for persona in personas:
        name = _text(_field(persona, "name"))
        if name:
            items.append((name, persona))
    return sorted(items, key=lambda item: item[0])

def available_agents_block(
    personas: Any,
    *,
    include_input_schema: bool = False,
) -> str:
    """Render persona metadata as an ``available_agents`` XML block."""

    items = _agent_items(personas)
    if not items:
        return ""
    lines = ["<available_agents>"]
    for name, persona in items:
        lines.append("  <agent>")
        lines.append(f"    <name>{escape(name)}</name>")
        description = _text(_field(persona, "description"))
        if description:
            lines.append(f"    <description>{escape(description)}</description>")
        file_path = _text(_field(persona, "file_path"))
        if file_path:
            lines.append(f"    <persona_file>{escape(file_path)}</persona_file>")
        tools = _tools(_field(persona, "tools"))
        if tools:
            lines.append(f"    <tools>{escape(tools)}</tools>")
        model = _text(_field(persona, "model"))
        if model:
            lines.append(f"    <model>{escape(model)}</model>")
        input_schema = _field(persona, "input_schema", None)
        if include_input_schema and isinstance(input_schema, Mapping):
            lines.append('    <input_schema advisory="true">')
            required = _tools(input_schema.get("required"))
            optional = _tools(input_schema.get("optional"))
            if required:
                lines.append(f"      <required>{escape(required)}</required>")
            if optional:
                lines.append(f"      <optional>{escape(optional)}</optional>")
            lines.append("    </input_schema>")
        lines.append("  </agent>")
    lines.append("</available_agents>")
    return "\n".join(lines)

class AgentsConfig(BaseModel):
    model_config = {"extra": "allow"}

    agent_paths: list[str] = []
    inherit_claude: bool = True

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
    config_schema=AgentsConfig,
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

async def install(api: ExtensionAPI, config: AgentsConfig) -> None:
    inherit_claude = config.inherit_claude
    configured_paths = [Path(p) for p in config.agent_paths]
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
