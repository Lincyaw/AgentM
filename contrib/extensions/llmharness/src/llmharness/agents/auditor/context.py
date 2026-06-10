"""Auditor context atom — builds the system prompt from raw data passed by the parent."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from agentm.core.abi.events import BeforeAgentStartEvent
from agentm.core.abi.extension import ExtensionAPI
from agentm.extensions import ExtensionManifest

from llmharness.schema import Edge, Event, Finding, Phase

from .prompt import (
    build_auditor_system_prompt,
    build_auditor_trajectory_prompt,
    load_auditor_prompt,
)


class AuditorContextConfig(TypedDict, total=False):
    events: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    phases: list[dict[str, Any]]
    findings: list[dict[str, Any]]
    check_errors: dict[str, str]
    continuation_notes: list[str]
    summary_threshold: int
    prompt_name: str
    trajectory_snapshot: list[dict[str, Any]]
    mode: Literal["graph", "trajectory"]

MANIFEST = ExtensionManifest(
    name="auditor_context",
    description="Build the auditor system prompt from raw graph data.",
    registers=("event:before_agent_start",),
    config_schema={"type": "object", "additionalProperties": True},
)


def install(api: ExtensionAPI, config: AuditorContextConfig) -> None:  # type: ignore[override]
    if not config:
        return

    events = tuple(
        Event.from_dict(e) for e in (config.get("events") or []) if isinstance(e, dict)
    )
    edges = tuple(
        Edge.from_dict(e) for e in (config.get("edges") or []) if isinstance(e, dict)
    )
    phases = tuple(
        Phase.from_dict(p) for p in (config.get("phases") or []) if isinstance(p, dict)
    )
    findings = [
        Finding.from_dict(f) for f in (config.get("findings") or []) if isinstance(f, dict)
    ]
    check_errors = config.get("check_errors") or {}
    if not isinstance(check_errors, dict):
        check_errors = {}
    continuation_notes = [
        str(n) for n in (config.get("continuation_notes") or []) if isinstance(n, str)
    ]
    summary_threshold = int(config.get("summary_threshold", 30))
    prompt_name = str(config.get("prompt_name", "minimal"))
    trajectory_snapshot = config.get("trajectory_snapshot")
    mode = str(config.get("mode", "graph"))

    base_prompt = load_auditor_prompt(prompt_name)

    if mode == "trajectory" and trajectory_snapshot is not None:
        prompt_text = build_auditor_trajectory_prompt(
            trajectory=trajectory_snapshot,
            continuation_notes=continuation_notes,
            base_prompt=base_prompt,
        )
    else:
        prompt_text = build_auditor_system_prompt(
            events=events,
            edges=edges,
            phases=phases,
            findings=findings,
            check_errors=dict(check_errors),
            continuation_notes=continuation_notes,
            summary_threshold=summary_threshold,
            base_prompt=base_prompt,
        )

    def _before_start(event: BeforeAgentStartEvent) -> None:
        event.system = prompt_text

    api.on(BeforeAgentStartEvent.CHANNEL, _before_start)


__all__ = ["MANIFEST", "install"]
