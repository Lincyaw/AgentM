"""Generic context injection for devloop agents.

Reads all atom_config keys and injects them as labeled sections into
the system prompt. The workflow orchestrator passes task-specific data
(spec, test feedback, file lists) via atom_config; this atom formats
it so the agent starts with full context.
"""
from __future__ import annotations

import json
from typing import TypeAlias

from agentm.core.abi.events import BeforeAgentStartEvent
from agentm.core.abi.extension import ExtensionAPI
from agentm.extensions import ExtensionManifest

MANIFEST = ExtensionManifest(
    name="devloop_context",
    description="Inject structured context data into the agent's system prompt.",
    registers=("event:before_agent_start",),
    config_schema={"type": "object", "additionalProperties": True},
)


ContextValue: TypeAlias = (
    str | int | float | bool | None | list["ContextValue"] | dict[str, "ContextValue"]
)
DevloopContextConfig: TypeAlias = dict[str, ContextValue]


def _format_value(value: ContextValue) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return "\n".join(f"- {v}" for v in value)
    return f"```json\n{json.dumps(value, indent=2, ensure_ascii=False)}\n```"


def install(api: ExtensionAPI, config: DevloopContextConfig) -> None:
    if not config:
        return

    parts = [
        f"## {key}\n\n{_format_value(value)}"
        for key, value in config.items()
    ]
    context = "\n\n".join(parts)

    def before_agent_start(event: BeforeAgentStartEvent) -> None:
        current = str(event.system or "")
        event.system = f"{current}\n\n{context}" if current else context

    api.on(BeforeAgentStartEvent.CHANNEL, before_agent_start)


__all__ = ["MANIFEST", "install"]
