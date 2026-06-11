"""Auditor context atom — builds the system prompt from raw data passed by the parent."""

from __future__ import annotations

from typing import Any, Literal

from agentm.core.abi import BeforeAgentStartEvent, ExtensionAPI
from agentm.extensions import ExtensionManifest
from pydantic import BaseModel

from llmharness.schema import Edge, Event, Finding, Phase

from .prompt import (
    build_auditor_system_prompt,
    build_auditor_trajectory_prompt,
    load_auditor_prompt,
)


class AuditorContextConfig(BaseModel):
    events: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    phases: list[dict[str, Any]] = []
    findings: list[dict[str, Any]] = []
    check_errors: dict[str, str] = {}
    continuation_notes: list[str] = []
    summary_threshold: int = 30
    prompt_name: str = "minimal"
    trajectory_snapshot: list[dict[str, Any]] | None = None
    mode: Literal["graph", "trajectory"] = "graph"

MANIFEST = ExtensionManifest(
    name="auditor_context",
    description="Build the auditor system prompt from raw graph data.",
    registers=("event:before_agent_start",),
    config_schema=AuditorContextConfig,
)

def install(api: ExtensionAPI, config: AuditorContextConfig) -> None:
    events = tuple(Event.from_dict(e) for e in config.events)
    edges = tuple(Edge.from_dict(e) for e in config.edges)
    phases = tuple(Phase.from_dict(p) for p in config.phases)
    findings = [Finding.from_dict(f) for f in config.findings]

    base_prompt = load_auditor_prompt(config.prompt_name)

    if config.mode == "trajectory" and config.trajectory_snapshot is not None:
        prompt_text = build_auditor_trajectory_prompt(
            trajectory=config.trajectory_snapshot,
            continuation_notes=config.continuation_notes,
            base_prompt=base_prompt,
        )
    else:
        prompt_text = build_auditor_system_prompt(
            events=events,
            edges=edges,
            phases=phases,
            findings=findings,
            check_errors=config.check_errors,
            continuation_notes=config.continuation_notes,
            summary_threshold=config.summary_threshold,
            base_prompt=base_prompt,
        )

    def _before_start(event: BeforeAgentStartEvent) -> None:
        event.system = prompt_text

    api.on(BeforeAgentStartEvent.CHANNEL, _before_start)

__all__ = ["MANIFEST", "install"]
