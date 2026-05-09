"""Scenario-local extension wiring orchestrator prompt + worker personas.

Two responsibilities:

1. Load ``prompts/orchestrator.md`` from this scenario directory and prepend
   it to the parent agent's system prompt at ``before_agent_start``.
2. Discover ``agents/*.md`` persona files, append an ``<available_agents>``
   advisory block to the orchestrator prompt, and answer ``resolve_subagent``
   so :mod:`agentm.extensions.builtin.sub_agent` can inject persona metadata
   into child sessions when ``dispatch_agent`` is called.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from agentm.core.lib.frontmatter import parse_frontmatter
from agentm.extensions import ExtensionManifest
from agentm.harness.events import BeforeAgentStartEvent, SessionReadyEvent
from agentm.harness.extension import ExtensionAPI


_SCENARIO_ROOT = Path(__file__).resolve().parent
_PROMPT_PATH = _SCENARIO_ROOT / "prompts" / "orchestrator.md"
_AGENTS_DIR = _SCENARIO_ROOT / "agents"


MANIFEST = ExtensionManifest(
    name="orchestrator_setup",
    description=(
        "Inject the rca orchestrator prompt, advertise worker personas, and "
        "resolve scout / verify / deep_analyze metadata for sub_agent."
    ),
    registers=(
        "event:session_ready",
        "event:before_agent_start",
        "event:resolve_subagent",
    ),
    config_schema={"type": "object", "additionalProperties": False},
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
            "tools": _parse_tools(metadata.get("tools") if isinstance(metadata, dict) else None),
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


def _load_agent_contract_block() -> str:
    """Splice the rcabench-platform agent contract into the system prompt.

    This is the same prompt used by ThinkDepthAI's RCA synthesizer — it
    pins the ``service`` vocabulary to "strings present in the data",
    enumerates the canonical ``fault_kind`` values, and lists the
    synthetic generators that must NOT be reported as services. Without
    this block the orchestrator invents service names like
    ``mysql-database`` when ground truth is ``mysql``.
    """
    try:
        from rcabench_platform.v3.sdk.evaluation.v2 import (
            get_agent_contract_prompt,
        )
    except ImportError:
        return ""
    body = str(get_agent_contract_prompt()).strip()
    if not body:
        return ""
    return (
        "<agent_contract>\n"
        "The shape and vocabulary below are enforced by `submit_final_report`.\n"
        "Match `service` and `fault_kind` exactly; evidence SQL must be "
        "runnable on the case dir.\n\n"
        f"{body}\n"
        "</agent_contract>"
    )


def _format_available_agents_block(personas: dict[str, dict[str, Any]]) -> str:
    if not personas:
        return ""
    lines = ["<available_agents>"]
    for name in sorted(personas):
        persona = personas[name]
        lines.append("  <agent>")
        lines.append(f"    <name>{html.escape(name)}</name>")
        description = str(persona.get("description", "")).strip()
        if description:
            lines.append(
                f"    <description>{html.escape(description)}</description>"
            )
        tools = persona.get("tools")
        if isinstance(tools, list) and tools:
            lines.append(f"    <tools>{html.escape(', '.join(tools))}</tools>")
        input_schema = persona.get("input_schema")
        if isinstance(input_schema, dict):
            lines.append("    <input_schema advisory=\"true\">")
            required = input_schema.get("required")
            optional = input_schema.get("optional")
            if isinstance(required, list) and required:
                lines.append(
                    f"      <required>{html.escape(', '.join(str(x) for x in required))}</required>"
                )
            if isinstance(optional, list) and optional:
                lines.append(
                    f"      <optional>{html.escape(', '.join(str(x) for x in optional))}</optional>"
                )
            lines.append("    </input_schema>")
        lines.append("  </agent>")
    lines.append("</available_agents>")
    return "\n".join(lines)


async def install(api: ExtensionAPI, _config: dict[str, Any]) -> None:
    cached_system = ""
    personas: dict[str, dict[str, Any]] = {}

    async def _load(_event: SessionReadyEvent) -> None:
        nonlocal cached_system, personas
        prompt = ""
        if _PROMPT_PATH.is_file():
            prompt = _PROMPT_PATH.read_text(encoding="utf-8").strip()
        personas = _load_personas()
        available_agents = _format_available_agents_block(personas)
        contract_block = _load_agent_contract_block()
        sections = [s for s in (prompt, available_agents, contract_block) if s]
        cached_system = "\n\n".join(sections)

    def _inject_prompt(event: BeforeAgentStartEvent) -> dict[str, str] | None:
        if not cached_system:
            return None
        existing = event.system or ""
        merged = f"{cached_system}\n\n{existing}" if existing else cached_system
        event.system = merged
        return {"system": merged}

    def _resolve(payload: Any) -> dict[str, Any] | None:
        if not isinstance(payload, dict):
            return None
        name = payload.get("name")
        if not isinstance(name, str):
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

    api.on("session_ready", _load)
    api.on("before_agent_start", _inject_prompt)
    api.on("resolve_subagent", _resolve)
