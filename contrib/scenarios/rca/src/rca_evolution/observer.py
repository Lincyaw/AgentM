"""Observe step: spawn an observer agent that investigates a failed RCA case.

The observer gets tools to inspect the trajectory (get_turn,
get_trajectory_summary, get_gt_info) and submits structured findings
via submit_divergence_report.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

@dataclass(slots=True)
class DivergencePoint:
    turn_index: int
    description: str
    should_have_done: str
    category: str

@dataclass(slots=True)
class DivergenceReport:
    case_id: str
    correct: bool
    root_causes_gt: list[str]
    root_causes_agent: list[str]
    divergence_points: list[DivergencePoint] = field(default_factory=list)
    key_lesson: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "correct": self.correct,
            "root_causes_gt": self.root_causes_gt,
            "root_causes_agent": self.root_causes_agent,
            "divergence_points": [
                {"turn_index": dp.turn_index, "description": dp.description,
                 "should_have_done": dp.should_have_done, "category": dp.category}
                for dp in self.divergence_points
            ],
            "key_lesson": self.key_lesson,
        }

def _extract_gt_services(data_dir: str) -> tuple[list[str], list[str]]:
    injection_path = Path(data_dir) / "injection.json"
    causal_graph_path = Path(data_dir) / "causal_graph.json"

    injected_apps: list[str] = []
    fault_types: list[str] = []
    if injection_path.exists():
        with open(injection_path, encoding="utf-8") as f:
            injection = json.load(f)
        injected_apps = [e["app"] for e in injection.get("engine_config_summary", [])]
        fault_types = [e.get("chaos_type", "unknown") for e in injection.get("engine_config_summary", [])]

    if not injected_apps and causal_graph_path.exists():
        with open(causal_graph_path, encoding="utf-8") as f:
            cg = json.load(f)
        for node in cg.get("root_causes", cg.get("nodes", [])):
            component = node.get("component", "")
            if "|" in component:
                injected_apps.append(component.split("|", 1)[1])

    return injected_apps, fault_types

def _extract_agent_services(response: str) -> list[str]:
    try:
        data = json.loads(response)
    except (json.JSONDecodeError, TypeError):
        return []
    return [
        rc.get("service", "")
        for rc in data.get("root_causes", [])
        if isinstance(rc, dict) and rc.get("service")
    ]

def _build_trajectory_snapshot(trajectory_path: str) -> list[dict[str, Any]]:
    """Parse OTLP JSONL into a list of turn dicts for get_turn."""
    path = Path(trajectory_path)
    if not path.exists():
        return []

    turns: list[dict[str, Any]] = []
    try:
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if isinstance(rec, dict):
                    turns.append(rec)
            except json.JSONDecodeError:
                continue
    except OSError:
        pass
    return turns

def _build_trajectory_summary(snapshot: list[dict[str, Any]]) -> str:
    """Condensed overview of tool calls for get_trajectory_summary."""
    parts: list[str] = []
    turn = 0
    for rec in snapshot:
        rtype = rec.get("type", rec.get("name", ""))
        if "tool_call" in rtype.lower() or rtype == "tool_call":
            name = rec.get("name", rec.get("body", {}).get("name", "?"))
            parts.append(f"[T{turn}] CALL {name}")
        elif "tool_result" in rtype.lower() or rtype == "tool_result":
            turn += 1
        elif "assistant" in rtype.lower():
            parts.append(f"[T{turn}] ASSISTANT message")
    return "\n".join(parts) if parts else "(could not parse trajectory)"

_OBSERVER_PROMPT = """\
You are an expert RCA evaluator. An RCA agent investigated a microservice \
incident and got the WRONG answer.

Your job: use the tools to understand what went wrong.

## Workflow
1. Call ``get_gt_info`` to see the correct root causes and the agent's conclusion.
2. Call ``get_trajectory_summary`` to get an overview of all tool calls.
3. Call ``get_turn`` on specific turns to drill into what the agent did.
4. Once you understand where and why the agent diverged, call \
``submit_divergence_report`` with your findings.

Focus on identifying the EARLIEST point where the investigation went off \
track, and categorize each divergence point.
"""

async def observe_case(
    *,
    case_id: str,
    data_dir: str,
    agent_response: str,
    trajectory_path: str,
    provider_tuple: tuple[str, dict[str, Any]],
) -> DivergenceReport:
    """Spawn an observer agent session to analyze one failed case."""
    from agentm.core.abi import (
        AgentSessionConfig,
        AssistantMessage,
        LoopConfig,
        ToolCallBlock,
    )
    from agentm.core.runtime import AgentSession
    from agentm.core.runtime import create_agent_session

    gt_services, fault_types = _extract_gt_services(data_dir)
    agent_services = _extract_agent_services(agent_response)

    gt_set = {s.lower() for s in gt_services}
    agent_set = {s.lower() for s in agent_services}
    correct = bool(gt_set) and gt_set.issubset(agent_set)

    if correct:
        return DivergenceReport(
            case_id=case_id, correct=True,
            root_causes_gt=gt_services, root_causes_agent=agent_services,
            key_lesson="Correct.",
        )

    snapshot = _build_trajectory_snapshot(trajectory_path)
    summary = _build_trajectory_summary(snapshot)

    gt_info = {
        "correct_root_causes": gt_services,
        "fault_types": fault_types,
        "agent_concluded": agent_services or "(agent submitted nothing)",
    }

    config = AgentSessionConfig(
        cwd=data_dir,
        provider=provider_tuple,
        loop_config=LoopConfig(max_turns=15),
        extensions=[
            ("agentm.extensions.builtin.operations", {"backend": "local"}),
            ("rca_evolution.observer_atom", {
                "trajectory_snapshot": snapshot,
                "gt_info": gt_info,
                "trajectory_summary": summary,
            }),
        ],
    )

    session = await create_agent_session(AgentSession, config)
    try:
        messages = await session.prompt(_OBSERVER_PROMPT)
    finally:
        await session.shutdown()

    # Extract submit_divergence_report args from messages
    report_args: dict[str, Any] | None = None
    for msg in reversed(messages):
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, ToolCallBlock) and block.name == "submit_divergence_report":
                    report_args = block.arguments
                    break
            if report_args:
                break

    if report_args is None:
        logger.warning(f"Observer did not submit report for {case_id}")
        return DivergenceReport(
            case_id=case_id, correct=False,
            root_causes_gt=gt_services, root_causes_agent=agent_services,
            divergence_points=[DivergencePoint(0, "observer did not submit", "unknown", "analysis_failed")],
            key_lesson="Observer failed to submit report.",
        )

    return DivergenceReport(
        case_id=case_id, correct=False,
        root_causes_gt=gt_services, root_causes_agent=agent_services,
        divergence_points=[
            DivergencePoint(
                turn_index=dp.get("turn_index", 0),
                description=dp.get("description", ""),
                should_have_done=dp.get("should_have_done", ""),
                category=dp.get("category", "unknown"),
            )
            for dp in report_args.get("divergence_points", [])
        ],
        key_lesson=report_args.get("key_lesson", ""),
    )
