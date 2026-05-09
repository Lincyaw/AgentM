"""Shared XML rendering for sub-agent persona advertisements."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from xml.sax.saxutils import escape


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


__all__ = ["available_agents_block"]
