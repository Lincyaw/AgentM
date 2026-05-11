"""Generalized prompt-loading atom shared by ``rca`` and ``rca:baseline``.

Subsumes the old per-manifest setup atoms (``orchestrator_setup`` and
``setup``). One config-driven implementation:

* ``prompt`` (required): filename under ``contrib/scenarios/rca/prompts/``.
* ``personas`` (optional, default ``false``): when truthy, also load
  ``contrib/scenarios/rca/agents/*.md``, append the available-agents block
  to the system prompt, and answer ``ResolveSubagentEvent`` so
  :mod:`agentm.extensions.builtin.sub_agent` can dispatch them. Single-agent
  manifests leave this off.

Both prompts and personas live under the canonical ``rca`` scenario
directory so adding a new variant is a manifest-only change.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from agentm.core.lib.frontmatter import parse_frontmatter
from agentm.extensions import ExtensionManifest
from agentm.core.abi.events import (
    BeforeAgentStartEvent,
    ResolveSubagentEvent,
    SessionReadyEvent,
)
from agentm.core.abi.extension import ExtensionAPI


_RCA_ROOT = Path(__file__).resolve().parent.parent.parent
_PROMPTS_DIR = _RCA_ROOT / "prompts"
_AGENTS_DIR = _RCA_ROOT / "agents"


# ---------------------------------------------------------------------------
# available_agents XML rendering — inlined from the former
# agentm.core.lib.available_agents (this scenario is one of two consumers).
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


MANIFEST = ExtensionManifest(
    name="prompt_loader",
    description=(
        "Inject an rca scenario prompt; optionally also load worker personas "
        "and answer resolve_subagent for them."
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
    tier=2,
)


def _parse_tools(raw: Any) -> list[str] | None:
    if isinstance(raw, list):
        tools = [str(item).strip() for item in raw if str(item).strip()]
        return tools or None
    if isinstance(raw, str) and raw.strip():
        tools = [t.strip() for t in raw.split(",") if t.strip()]
        return tools or None
    return None


def _parse_string_list(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    return []


def _parse_input_schema(raw: Any) -> dict[str, list[str]] | None:
    if not isinstance(raw, dict):
        return None
    required = _parse_string_list(raw.get("required"))
    optional = _parse_string_list(raw.get("optional"))
    if not required and not optional:
        return None
    return {"required": required, "optional": optional}


def _parse_budget_defaults(raw: Any) -> dict[str, int] | None:
    if not isinstance(raw, dict):
        return None
    budget: dict[str, int] = {}
    for key in ("max_tool_calls", "max_turns"):
        value = raw.get(key)
        if isinstance(value, int) and value > 0:
            budget[key] = value
    return budget or None


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
            "input_schema": _parse_input_schema(
                metadata.get("input_schema") if isinstance(metadata, dict) else None
            ),
            "budget_defaults": _parse_budget_defaults(
                metadata.get("budget_defaults") if isinstance(metadata, dict) else None
            ),
            "artifact_kinds": _parse_string_list(
                metadata.get("artifact_kinds") if isinstance(metadata, dict) else None
            ),
        }
    return personas


async def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    prompt_name = str(config["prompt"]).strip()
    if not prompt_name:
        raise ValueError("prompt_loader: 'prompt' config must be a non-empty filename")
    prompt_path = _PROMPTS_DIR / prompt_name
    enable_personas = bool(config.get("personas", False))

    cached_system = ""
    personas: dict[str, dict[str, Any]] = {}

    async def _load(_event: SessionReadyEvent) -> None:
        nonlocal cached_system, personas
        prompt = ""
        if prompt_path.is_file():
            prompt = prompt_path.read_text(encoding="utf-8").strip()
            # Substitute the single supported placeholder. Kept narrow on
            # purpose — wider templating belongs in a dedicated atom.
            if "{date}" in prompt:
                prompt = prompt.replace(
                    "{date}", _dt.date.today().isoformat()
                )
        sections = [prompt] if prompt else []
        if enable_personas:
            personas = _load_personas()
            block = available_agents_block(personas, include_input_schema=True)
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
            "input_schema": persona["input_schema"],
            "budget_defaults": persona["budget_defaults"],
            "artifact_kinds": persona["artifact_kinds"],
        }

    api.on(SessionReadyEvent.CHANNEL, _load)
    api.on(BeforeAgentStartEvent.CHANNEL, _inject_prompt)
    if enable_personas:
        api.on(ResolveSubagentEvent.CHANNEL, _resolve)
