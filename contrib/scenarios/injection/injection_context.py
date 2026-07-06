"""Inject workflow-provided campaign context into injection worker sessions."""
from __future__ import annotations

import json
from typing import Any, Final

from pydantic import BaseModel, ConfigDict

from agentm.core.abi import BeforeAgentStartEvent, ExtensionAPI
from agentm.extensions import ExtensionManifest

class InjectionContextConfig(BaseModel):
    model_config = ConfigDict(extra="allow")


MANIFEST = ExtensionManifest(
    name="injection_context",
    description="Adds workflow-provided Aegis injection context to the system prompt.",
    registers=("event:before_agent_start",),
    config_schema=InjectionContextConfig,
)


def _format(value: Any) -> str:
    if isinstance(value, str):
        return value
    return "```json\n" + json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n```"


def install(api: ExtensionAPI, config: InjectionContextConfig) -> None:
    data = config.model_dump(mode="json")
    if not data:
        return

    sections = [
        "# Workflow-provided injection context",
        "Use this context as authoritative runtime input. Do not infer these values from old rounds when they are present here.",
    ]
    for key, value in data.items():
        sections.append(f"## {key}\n\n{_format(value)}")
    context = "\n\n".join(sections)

    def before_agent_start(event: BeforeAgentStartEvent) -> None:
        current = str(event.system or "")
        event.system = f"{current}\n\n{context}" if current else context

    api.on(BeforeAgentStartEvent.CHANNEL, before_agent_start)


__all__: Final = ["MANIFEST", "install"]
