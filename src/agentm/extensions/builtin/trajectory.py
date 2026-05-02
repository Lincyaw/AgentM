"""Builtin ``trajectory`` atom per extension-as-scenario §7."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from agentm.core.abi import AgentEndEvent
from agentm.extensions import ExtensionManifest
from agentm.harness.extension import ExtensionAPI


MANIFEST = ExtensionManifest(
    name="trajectory",
    description="Records major event traffic to memory and persists JSONL on agent end.",
    registers=(
        "event:agent_start",
        "event:agent_end",
        "event:decide_turn_action",
        "event:turn_start",
        "event:turn_end",
        "event:context",
        "event:before_send_to_llm",
        "event:tool_call",
        "event:tool_result",
        "event:before_compact",
        "event:after_compact",
        "event:child_session_start",
        "event:child_session_end",
        "event:cost_budget_exceeded",
        "event:plan_submitted",
    ),
    config_schema={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "channels": {
                "type": ["array", "null"],
                "items": {"type": "string"},
            },
        },
        "additionalProperties": False,
    },
)

_DEFAULT_CHANNELS = (
    "agent_start",
    "agent_end",
    "decide_turn_action",
    "turn_start",
    "turn_end",
    "context",
    "before_send_to_llm",
    "tool_call",
    "tool_result",
    "before_compact",
    "after_compact",
    "child_session_start",
    "child_session_end",
    "cost_budget_exceeded",
    "plan_submitted",
)


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    configured_path = Path(str(config.get("path", "./trajectory.jsonl")))
    output_path = (
        configured_path
        if configured_path.is_absolute()
        else Path(api.cwd) / configured_path
    )
    configured_channels = config.get("channels")
    channels = (
        tuple(str(ch) for ch in configured_channels)
        if configured_channels is not None
        else _DEFAULT_CHANNELS
    )
    records: list[dict[str, Any]] = []

    def _record(channel: str, event: Any) -> None:
        # ``event`` is intentionally ``Any``: the trajectory atom records every
        # channel polymorphically — kernel events, harness events, and
        # extension-defined events all flow through ``_serialize``.
        records.append(
            {
                "timestamp": time.time(),
                "channel": channel,
                "event": _serialize(event),
            }
        )

    for channel in channels:
        if channel == "agent_end":

            def on_agent_end(
                event: AgentEndEvent, *, _channel: str = channel
            ) -> None:
                _record(_channel, event)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with output_path.open("w", encoding="utf-8") as handle:
                    for record in records:
                        handle.write(json.dumps(record, default=str))
                        handle.write("\n")

            api.on(channel, on_agent_end)
            continue

        api.on(channel, lambda event, _channel=channel: _record(_channel, event))


def _serialize(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {key: _serialize(val) for key, val in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _serialize(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    if isinstance(value, tuple):
        return [_serialize(item) for item in value]
    return value
