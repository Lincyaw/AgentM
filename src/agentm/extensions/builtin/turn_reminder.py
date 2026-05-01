"""Builtin ``turn_reminder`` atom per extension-as-scenario §7."""

from __future__ import annotations

from typing import Any

from agentm.extensions import ExtensionManifest
from agentm.harness.events import BeforeAgentStartEvent
from agentm.harness.extension import ExtensionAPI


MANIFEST = ExtensionManifest(
    name="turn_reminder",
    description="Re-injects a reminder into the system prompt every N turns.",
    registers=("event:before_agent_start",),
    config_schema={
        "type": "object",
        "properties": {
            "reminder": {"type": "string"},
            "every_n_turns": {"type": "integer", "minimum": 1},
        },
        "required": ["reminder", "every_n_turns"],
        "additionalProperties": False,
    },
)


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    # ``reminder`` and ``every_n_turns`` are required by
    # MANIFEST.config_schema; the discovery filter skips this atom when
    # configured with ``{}``.
    reminder = str(config["reminder"])
    every_n_turns = int(config["every_n_turns"])

    def before_agent_start(event: BeforeAgentStartEvent) -> dict[str, str] | None:
        turn_count = len(api.session.get_messages())
        if turn_count % every_n_turns != 0:
            return None
        current = str(event.system or "")
        updated = f"{current}\n\n{reminder}" if current else reminder
        event.system = updated
        return {"system": updated}

    api.on("before_agent_start", before_agent_start)
