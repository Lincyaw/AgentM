"""Builtin ``system_prompt`` atom per extension-as-scenario §7."""

from __future__ import annotations

from typing import Any

from agentm.core.abi.roles import SYSTEM_PROMPT_PROVIDER
from agentm.extensions import ExtensionManifest
from agentm.core.abi.events import BeforeAgentStartEvent
from agentm.core.abi.extension import ExtensionAPI


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
    requires=(),  # Leaf atom: prepends configured prompt text only.
    provides_role=(SYSTEM_PROMPT_PROVIDER,),
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

    api.on(BeforeAgentStartEvent.CHANNEL, before_agent_start)
