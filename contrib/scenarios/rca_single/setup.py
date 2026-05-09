"""Scenario-local extension: load the single-investigator prompt.

Mirrors :mod:`agentm_rca.orchestrator_setup` but without sub-agent
machinery — there is no ``<available_agents>`` block, no
``resolve_subagent`` handler, no persona discovery. Just the prompt
plus the rcabench-platform agent contract block.
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
        "event:session_ready",
        "event:before_agent_start",
    ),
    config_schema={"type": "object", "additionalProperties": False},
    tier=2,
)


def _load_agent_contract_block() -> str:
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


async def install(api: ExtensionAPI, _config: dict[str, Any]) -> None:
    cached_system = ""

    async def _load(_event: SessionReadyEvent) -> None:
        nonlocal cached_system
        prompt = ""
        if _PROMPT_PATH.is_file():
            prompt = _PROMPT_PATH.read_text(encoding="utf-8").strip()
        contract_block = _load_agent_contract_block()
        sections = [s for s in (prompt, contract_block) if s]
        cached_system = "\n\n".join(sections)

    def _inject_prompt(event: BeforeAgentStartEvent) -> dict[str, str] | None:
        if not cached_system:
            return None
        existing = event.system or ""
        merged = f"{cached_system}\n\n{existing}" if existing else cached_system
        event.system = merged
        return {"system": merged}

    api.on("session_ready", _load)
    api.on("before_agent_start", _inject_prompt)
