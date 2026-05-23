"""Verifier-scenario prompt loader (clone of agentm_rca.prompt_loader).

Mirrors the rca loader but resolves prompt / persona files against
``contrib/scenarios/verifier/``. We deliberately clone rather than
import from ``agentm_rca`` to keep scenario packages isolated. If a
third scenario needs the same pattern, dedupe into a shared atom.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from agentm.core.abi.events import (
    BeforeAgentStartEvent,
    ResolveSubagentEvent,
    SessionReadyEvent,
)
from agentm.core.abi.extension import ExtensionAPI
from agentm.core.lib.frontmatter import parse_frontmatter
from agentm.extensions import ExtensionManifest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_VERIFIER_ROOT = _REPO_ROOT / "contrib" / "scenarios" / "verifier"
_PROMPTS_DIR = _VERIFIER_ROOT / "prompts"
_AGENTS_DIR = _VERIFIER_ROOT / "agents"


MANIFEST = ExtensionManifest(
    name="verifier_prompt_loader",
    description=(
        "Inject a verifier-scenario prompt; optionally also load worker "
        "personas and answer resolve_subagent for them."
    ),
    registers=(
        f"event:{SessionReadyEvent.CHANNEL}",
        f"event:{BeforeAgentStartEvent.CHANNEL}",
        f"event:{ResolveSubagentEvent.CHANNEL}",
    ),
    config_schema={
        "type": "object",
        "properties": {
            "prompt": {"type": "string"},
            "personas": {"type": "boolean"},
        },
        "required": ["prompt"],
        "additionalProperties": False,
    },
)


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


def _available_agents_block(personas: Any) -> str:
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
        lines.append("  </agent>")
    lines.append("</available_agents>")
    return "\n".join(lines)


def _parse_tools(raw: Any) -> list[str] | None:
    if isinstance(raw, list):
        tools = [str(item).strip() for item in raw if str(item).strip()]
        return tools or None
    if isinstance(raw, str) and raw.strip():
        tools = [t.strip() for t in raw.split(",") if t.strip()]
        return tools or None
    return None


def _load_personas() -> dict[str, dict[str, Any]]:
    personas: dict[str, dict[str, Any]] = {}
    if not _AGENTS_DIR.is_dir():
        return personas
    for md_path in sorted(_AGENTS_DIR.glob("*.md")):
        try:
            text = md_path.read_text(encoding="utf-8")
        except OSError:
            continue
        metadata, body = parse_frontmatter(text)
        name = metadata.get("name") if isinstance(metadata, dict) else None
        if not isinstance(name, str) or not name.strip():
            name = md_path.stem
        personas[name.strip()] = {
            "body": body.strip(),
            "tools": _parse_tools(
                metadata.get("tools") if isinstance(metadata, dict) else None
            ),
            "description": (
                str(metadata.get("description", "")).strip()
                if isinstance(metadata, dict)
                else ""
            ),
            "file_path": str(md_path),
        }
    return personas


async def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    prompt_name = str(config["prompt"]).strip()
    if not prompt_name:
        raise ValueError(
            "verifier_prompt_loader: 'prompt' must be a non-empty filename"
        )
    prompt_path = _PROMPTS_DIR / prompt_name
    enable_personas = bool(config.get("personas", False))

    cached_system = ""
    personas: dict[str, dict[str, Any]] = {}

    async def _load(_event: SessionReadyEvent) -> None:
        nonlocal cached_system, personas
        prompt = ""
        if prompt_path.is_file():
            prompt = prompt_path.read_text(encoding="utf-8").strip()
            if "{date}" in prompt:
                prompt = prompt.replace("{date}", _dt.date.today().isoformat())
        sections = [prompt] if prompt else []
        if enable_personas:
            personas = _load_personas()
            block = _available_agents_block(personas)
            if block:
                sections.append(block)
        cached_system = "\n\n".join(sections)

    def _inject_prompt(event: BeforeAgentStartEvent) -> dict[str, str] | None:
        if not cached_system:
            return None
        existing = event.system or ""
        merged = f"{cached_system}\n\n{existing}" if existing else cached_system
        event.system = merged
        return {"system": merged}

    def _resolve(payload: Any) -> dict[str, Any] | None:
        if not enable_personas:
            return None
        if isinstance(payload, ResolveSubagentEvent):
            name = payload.name
        elif isinstance(payload, dict):
            raw_name = payload.get("name")
            if not isinstance(raw_name, str):
                return None
            name = raw_name
        else:
            return None
        persona = personas.get(name.strip())
        if persona is None:
            return None
        return {
            "body": persona["body"],
            "tools": persona["tools"],
            "input_schema": None,
            "budget_defaults": None,
            "artifact_kinds": [],
        }

    api.on(SessionReadyEvent.CHANNEL, _load)
    api.on(BeforeAgentStartEvent.CHANNEL, _inject_prompt)
    if enable_personas:
        api.on(ResolveSubagentEvent.CHANNEL, _resolve)


__all__ = ["MANIFEST", "install"]
