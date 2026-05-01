"""Builtin ``system_prompt`` atom per extension-as-scenario §7."""

from __future__ import annotations

from typing import Any

from agentm.extensions import ExtensionManifest
from agentm.harness.events import BeforeAgentStartEvent
from agentm.harness.extension import ExtensionAPI


MANIFEST = ExtensionManifest(
    name="system_prompt",
    description="Prepends configured prompt text to the system prompt.",
    registers=("event:before_agent_start",),
    config_schema={
        "type": "object",
        "properties": {
            "prompt": {"type": "string"},
        },
        "required": ["prompt"],
        "additionalProperties": False,
    },
)


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    # ``prompt`` is required by MANIFEST.config_schema; the discovery
    # filter skips this atom when configured with ``{}``.
    prompt = str(config["prompt"])

    def before_agent_start(event: BeforeAgentStartEvent) -> dict[str, str]:
        current = str(event.system or "")
        updated = f"{prompt}\n\n{current}" if current else prompt
        event.system = updated
        return {"system": updated}

    api.on("before_agent_start", before_agent_start)
