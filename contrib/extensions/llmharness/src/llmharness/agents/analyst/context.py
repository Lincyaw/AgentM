"""Analyst context atom — loads the system prompt."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from agentm.core.abi import BeforeAgentStartEvent, ExtensionAPI
from agentm.extensions import ExtensionManifest
from pydantic import BaseModel

_PROMPTS_DIR: Final = Path(__file__).parent / "prompts"


class AnalystContextConfig(BaseModel):
    prompt_name: str = "failure_analysis"


MANIFEST = ExtensionManifest(
    name="analyst_context",
    description="Build the analyst system prompt.",
    registers=("event:before_agent_start",),
    config_schema=AnalystContextConfig,
)


def install(api: ExtensionAPI, config: AnalystContextConfig) -> None:
    prompt_name = config.prompt_name
    md = _PROMPTS_DIR / f"{prompt_name}.md"
    if not md.is_file():
        raise ValueError(f"analyst prompt {prompt_name!r} not found at {md}")
    prompt_text = md.read_text(encoding="utf-8")

    def _before_start(event: BeforeAgentStartEvent) -> None:
        event.system = prompt_text

    api.on(BeforeAgentStartEvent.CHANNEL, _before_start)
