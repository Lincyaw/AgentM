"""Context injection for WorkGraph conflict resolver workers."""
from __future__ import annotations

import json
from typing import TypeAlias

from pydantic import BaseModel, ConfigDict

from agentm.core.abi import BeforeAgentStartEvent, ExtensionAPI
from agentm.extensions import ExtensionManifest

ContextValue: TypeAlias = (
    str | int | float | bool | None | list["ContextValue"] | dict[str, "ContextValue"]
)


class WorkGraphContextConfig(BaseModel):
    model_config = ConfigDict(extra="allow")


MANIFEST = ExtensionManifest(
    name="workgraph_context",
    description="Inject WorkGraph conflict context into a worker system prompt.",
    registers=("event:before_agent_start",),
    config_schema=WorkGraphContextConfig,
)


def _format_value(value: ContextValue) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return "\n".join(f"- {v}" for v in value)
    return f"```json\n{json.dumps(value, indent=2, ensure_ascii=False)}\n```"


def install(api: ExtensionAPI, config: WorkGraphContextConfig) -> None:
    context_values = dict(config.model_extra or {})
    if not context_values:
        return

    parts = [
        f"## {key}\n\n{_format_value(value)}"
        for key, value in context_values.items()
    ]
    context = "\n\n".join(parts)

    def before_agent_start(event: BeforeAgentStartEvent) -> dict[str, str]:
        current = str(event.system or "")
        updated = f"{current}\n\n{context}" if current else context
        event.system = updated
        return {"system": updated}

    api.on(BeforeAgentStartEvent.CHANNEL, before_agent_start)


__all__ = ["MANIFEST", "install"]
