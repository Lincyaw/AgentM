"""Scenario-local extension wiring orchestrator prompt + worker personas.

Two responsibilities:

1. Load ``prompts/orchestrator.md`` from this scenario directory and prepend
   it to the parent agent's system prompt at ``before_agent_start``.
2. Discover ``agents/*.md`` persona files and answer the ``resolve_subagent``
   event so :mod:`agentm.extensions.builtin.sub_agent` can inject the persona
   body + tool allowlist into a child session when ``dispatch_agent`` is
   called with ``subagent_type=<name>``.
"""

from __future__ import annotations

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
        "Inject the rca orchestrator prompt and resolve scout / verify / "
        "deep_analyze worker personas for the sub_agent extension."
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
        return [str(item).strip() for item in raw if str(item).strip()]
    if isinstance(raw, str) and raw.strip():
        return [t.strip() for t in raw.split(",") if t.strip()]
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
            "tools": _parse_tools(metadata.get("tools") if isinstance(metadata, dict) else None),
            "description": (
                metadata.get("description", "") if isinstance(metadata, dict) else ""
            ),
            "file_path": str(md_path),
        }
    return personas


async def install(api: ExtensionAPI, _config: dict[str, Any]) -> None:
    cached_prompt = ""
    personas: dict[str, dict[str, Any]] = {}

    async def _load(_event: SessionReadyEvent) -> None:
        nonlocal cached_prompt, personas
        if _PROMPT_PATH.is_file():
            cached_prompt = _PROMPT_PATH.read_text(encoding="utf-8").strip()
        personas = _load_personas()

    def _inject_prompt(event: BeforeAgentStartEvent) -> dict[str, str] | None:
        if not cached_prompt:
            return None
        existing = event.system or ""
        merged = f"{cached_prompt}\n\n{existing}" if existing else cached_prompt
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
        return {"body": persona["body"], "tools": persona["tools"]}

    api.on("session_ready", _load)
    api.on("before_agent_start", _inject_prompt)
    api.on("resolve_subagent", _resolve)
