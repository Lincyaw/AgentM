"""Builtin atom that surfaces Claude Code's ``.claude/agents/*.md`` files
to the orchestrator as named personas.

Each markdown file under the discovered agent roots is parsed for its
frontmatter (``name``, ``description``, optional ``tools`` / ``model``).
A compact ``<available_agents>`` block is appended to the system prompt so
the model knows which agents exist and how to spawn them via the existing
``dispatch_agent`` tool from :mod:`agentm.extensions.builtin.sub_agent`.

This atom intentionally does *not* reimplement child-session spawning —
it only advertises personas. To launch one, the model reads the agent's
.md body via the ``read`` tool and prepends it to the ``dispatch_agent``
prompt as the child's persona.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentm.core.frontmatter import parse_frontmatter
from agentm.extensions import ExtensionManifest
from agentm.harness.events import BeforeAgentStartEvent, SessionReadyEvent
from agentm.harness.extension import ExtensionAPI


MANIFEST = ExtensionManifest(
    name="claude_agents",
    description=(
        "Discover .claude/agents/*.md personas and inject an "
        "<available_agents> system-prompt block."
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


@dataclass(frozen=True, slots=True)
class _AgentRecord:
    name: str
    description: str
    file_path: str
    tools: str
    model: str


def _default_roots(cwd: str) -> list[Path]:
    return [
        Path.home() / ".claude" / "agents",
        Path(cwd) / ".claude" / "agents",
    ]


def _parse_agent(md_path: Path) -> _AgentRecord | None:
    try:
        text = md_path.read_text(encoding="utf-8")
    except OSError:
        return None
    metadata, _body = parse_frontmatter(text)
    name = metadata.get("name")
    description = metadata.get("description")
    if not isinstance(name, str) or not name.strip():
        name = md_path.stem
    if not isinstance(description, str) or not description.strip():
        return None
    tools_raw = metadata.get("tools", "")
    tools = tools_raw if isinstance(tools_raw, str) else ""
    model_raw = metadata.get("model", "")
    model = model_raw if isinstance(model_raw, str) else ""
    return _AgentRecord(
        name=name.strip(),
        description=description.strip(),
        file_path=str(md_path.resolve()),
        tools=tools.strip(),
        model=model.strip(),
    )


def _format_block(agents: list[_AgentRecord]) -> str:
    if not agents:
        return ""
    lines = [
        "\n\nThe following agents (personas) can be dispatched via the "
        "`dispatch_agent` tool. To launch one, first `read` its persona file, "
        "then prepend that text to the `prompt` argument of `dispatch_agent`.",
        "",
        "<available_agents>",
    ]
    for agent in agents:
        lines.append("  <agent>")
        lines.append(f"    <name>{html.escape(agent.name)}</name>")
        lines.append(
            f"    <description>{html.escape(agent.description)}</description>"
        )
        lines.append(f"    <persona_file>{html.escape(agent.file_path)}</persona_file>")
        if agent.tools:
            lines.append(f"    <tools>{html.escape(agent.tools)}</tools>")
        if agent.model:
            lines.append(f"    <model>{html.escape(agent.model)}</model>")
        lines.append("  </agent>")
    lines.append("</available_agents>")
    return "\n".join(lines)


async def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    inherit_claude = bool(config.get("inherit_claude", True))
    configured_paths = [Path(str(p)) for p in config.get("agent_paths", [])]
    cached_block = ""

    async def _populate(_: SessionReadyEvent) -> None:
        nonlocal cached_block
        roots: list[Path] = list(configured_paths)
        if inherit_claude:
            roots.extend(_default_roots(api.cwd))

        seen: set[str] = set()
        agents: list[_AgentRecord] = []
        for root in roots:
            if not root.is_dir():
                continue
            for md_path in sorted(root.glob("*.md")):
                record = _parse_agent(md_path)
                if record is None or record.name in seen:
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

    api.on("session_ready", _populate)
    api.on("before_agent_start", _inject)
