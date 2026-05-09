"""Scenario-local extension: load the single-investigator prompt.

Mirrors :mod:`agentm_rca.orchestrator_setup` but without sub-agent
machinery — there is no sub-agent availability block, no
``resolve_subagent`` handler, no persona discovery. The shared
``rcabench_contract`` atom owns vendor-contract prompt injection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agentm.extensions import ExtensionManifest
from agentm.harness.events import BeforeAgentStartEvent, SessionReadyEvent
from agentm.harness.extension import ExtensionAPI


_SCENARIO_ROOT = Path(__file__).resolve().parent
_PROMPT_PATH = _SCENARIO_ROOT / "prompts" / "investigator.md"


MANIFEST = ExtensionManifest(
    name="setup",
    description=(
        "Inject the single-agent rca investigator prompt and append the "
        "rcabench-platform agent contract block."
    ),
    registers=(
        f"event:{SessionReadyEvent.CHANNEL}",
        f"event:{BeforeAgentStartEvent.CHANNEL}",
    ),
    config_schema={"type": "object", "additionalProperties": False},
    tier=2,
)


async def install(api: ExtensionAPI, _config: dict[str, Any]) -> None:
    cached_system = ""

    async def _load(_event: SessionReadyEvent) -> None:
        nonlocal cached_system
        prompt = ""
        if _PROMPT_PATH.is_file():
            prompt = _PROMPT_PATH.read_text(encoding="utf-8").strip()
        sections = [s for s in (prompt,) if s]
        cached_system = "\n\n".join(sections)

    def _inject_prompt(event: BeforeAgentStartEvent) -> dict[str, str] | None:
        if not cached_system:
            return None
        existing = event.system or ""
        merged = f"{cached_system}\n\n{existing}" if existing else cached_system
        event.system = merged
        return {"system": merged}

    api.on(SessionReadyEvent.CHANNEL, _load)
    api.on(BeforeAgentStartEvent.CHANNEL, _inject_prompt)
