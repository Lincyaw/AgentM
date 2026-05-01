"""Inject RCA answer-schema instructions into the system prompt."""

from __future__ import annotations

import json
from typing import Any

from agentm.extensions import ExtensionManifest
from agentm.harness.events import BeforeAgentStartEvent
from agentm.harness.extension import ExtensionAPI

from agentm_rca.answer_schemas import schema_for_task

MANIFEST = ExtensionManifest(
    name="answer_schema",
    description="Inject the RCA task output schema into the system prompt.",
    registers=("event:before_agent_start",),
)


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    task_type = str(config.get("task_type", "scout"))
    model = schema_for_task(task_type)
    schema_json = json.dumps(model.model_json_schema(), indent=2, sort_keys=True)
    schema_block = (
        f"<output_schema task_type=\"{task_type}\" model=\"{model.__name__}\">\n"
        f"{schema_json}\n"
        "</output_schema>\n"
        "Return a JSON object that matches this schema exactly."
    )

    def _inject(event: BeforeAgentStartEvent) -> dict[str, str]:
        current = str(event.system or "")
        updated = f"{current}\n\n{schema_block}" if current else schema_block
        event.system = updated
        return {"system": updated}

    api.on("before_agent_start", _inject)
