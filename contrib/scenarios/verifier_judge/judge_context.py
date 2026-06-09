"""Inject whole-graph review context into the judge agent session.

Reads structured config describing the propagation results (injections,
confirmed services, rejected verdicts, throughput) and builds the full
domain context via ``prompt.build_judge_prompt``. The context is appended
to the system prompt so the agent starts with complete case-specific
knowledge.
"""
from __future__ import annotations

from typing import Any

from agentm.core.abi.events import BeforeAgentStartEvent
from agentm.core.abi.extension import ExtensionAPI
from agentm.extensions import ExtensionManifest

from prompt import build_judge_prompt

MANIFEST = ExtensionManifest(
    name="judge_context",
    description="Inject whole-graph review context into the judge agent.",
    registers=("event:before_agent_start",),
    config_schema={
        "type": "object",
        "properties": {
            "injections": {
                "type": "array",
                "items": {"type": "object"},
            },
            "confirmed": {
                "type": "array",
                "items": {"type": "string"},
            },
            "rejected_verdicts": {
                "type": "array",
                "items": {"type": "object"},
            },
            "throughput": {
                "type": "object",
                "additionalProperties": True,
            },
            "seeds": {
                "type": "array",
                "items": {"type": "string"},
            },
            "verdict_by_target": {
                "type": ["object", "null"],
                "additionalProperties": True,
            },
        },
        "required": ["injections", "confirmed"],
        "additionalProperties": False,
    },
)


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    injections = config.get("injections", [])
    confirmed = config.get("confirmed", [])
    if not injections:
        return

    seeds_list: list[str] = config.get("seeds") or [
        i["target"] for i in injections if i.get("target")
    ]
    seeds = set(seeds_list)

    context = build_judge_prompt(
        injections=injections,
        confirmed=confirmed,
        rejected_verdicts=config.get("rejected_verdicts", []),
        throughput=config.get("throughput", {}),
        seeds=seeds,
        verdict_by_target=config.get("verdict_by_target"),
    )

    def before_agent_start(event: BeforeAgentStartEvent) -> None:
        current = str(event.system or "")
        event.system = f"{current}\n\n{context}" if current else context

    api.on(BeforeAgentStartEvent.CHANNEL, before_agent_start)


__all__ = ["MANIFEST", "install"]
