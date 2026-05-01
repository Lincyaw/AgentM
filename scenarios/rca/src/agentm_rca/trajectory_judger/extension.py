"""Structured-output extension for the trajectory judger recipe."""

from __future__ import annotations

import json
from typing import Any

from agentm.core.kernel import AgentEndEvent, AssistantMessage, TextContent
from agentm.extensions import ExtensionManifest
from agentm.harness.events import BeforeAgentStartEvent
from agentm.harness.extension import ExtensionAPI

from .data import TrajectoryLabel


MANIFEST = ExtensionManifest(
    name="trajectory_judger_output",
    description="Guide the model toward TrajectoryLabel output and persist the parsed result.",
    registers=("event:before_agent_start", "event:agent_end"),
    config_schema=None,
)


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    del config
    schema_json = json.dumps(TrajectoryLabel.model_json_schema(), indent=2, sort_keys=True)

    def before_agent_start(event: BeforeAgentStartEvent) -> dict[str, str]:
        prompt = (
            "You are the trajectory_judger classifier. Read the provided trajectory data "
            "and return exactly one JSON object that validates against the "
            "TrajectoryLabel schema below. Do not wrap the JSON in Markdown.\n\n"
            f"TrajectoryLabel JSON schema:\n{schema_json}"
        )
        current = str(event.system or "")
        updated = f"{prompt}\n\n{current}" if current else prompt
        event.system = updated
        return {"system": updated}

    def on_agent_end(event: AgentEndEvent) -> None:
        message = _last_assistant_message(event.messages)
        if message is None:
            return
        text = _assistant_text(message)
        if not text:
            return
        try:
            label = TrajectoryLabel.model_validate_json(text)
        except Exception:  # noqa: BLE001
            return
        api.session.append_entry("trajectory_label", label)

    api.on("before_agent_start", before_agent_start)
    api.on("agent_end", on_agent_end)


def _last_assistant_message(messages: list[Any]) -> AssistantMessage | None:
    for message in reversed(messages):
        if isinstance(message, AssistantMessage):
            return message
    return None


def _assistant_text(message: AssistantMessage) -> str:
    blocks: list[str] = []
    for block in message.content:
        if isinstance(block, TextContent):
            blocks.append(block.text)
    return "".join(blocks).strip()
