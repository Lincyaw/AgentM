"""Observe step: GT-aware divergence analysis on failed RCA cases.

Reads a failed case's trajectory + GT, uses an AgentM session to identify
where the agent diverged from the correct investigation path.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_logger = logging.getLogger(__name__)


@dataclass
class DivergencePoint:
    turn_index: int
    description: str
    should_have_done: str
    category: str


@dataclass
class DivergenceReport:
    case_id: str
    correct: bool
    root_causes_gt: list[str]
    root_causes_agent: list[str]
    divergence_points: list[DivergencePoint] = field(default_factory=list)
    key_lesson: str = ""


def _extract_gt_services(data_dir: str) -> tuple[list[str], list[str]]:
    """Extract GT root cause services from injection.json + causal_graph.json."""
    data_path = Path(data_dir)
    injection_path = data_path / "injection.json"
    causal_graph_path = data_path / "causal_graph.json"

    injected_apps: list[str] = []
    if injection_path.exists():
        with open(injection_path) as f:
            injection = json.load(f)
        injected_apps = [
            e["app"] for e in injection.get("engine_config_summary", [])
        ]
        fault_types = [
            e.get("chaos_type", "unknown")
            for e in injection.get("engine_config_summary", [])
        ]
    else:
        fault_types = []

    if not injected_apps and causal_graph_path.exists():
        with open(causal_graph_path) as f:
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


def _load_trajectory_summary(trajectory_path: str, max_chars: int = 30000) -> str:
    """Load OTLP JSONL and produce a condensed summary of tool calls."""
    path = Path(trajectory_path)
    if not path.exists():
        return "(trajectory file not found)"

    parts: list[str] = []
    turn = 0
    try:
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(rec, dict):
                continue

            rtype = rec.get("type", "")
            if rtype == "tool_call":
                name = rec.get("name", "?")
                args = rec.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        pass
                args_s = json.dumps(args, ensure_ascii=False)
                if len(args_s) > 400:
                    args_s = args_s[:400] + "..."
                parts.append(f"[T{turn}] CALL {name}({args_s})")
            elif rtype == "tool_result":
                name = rec.get("name", "?")
                result = str(rec.get("result", ""))
                if len(result) > 200:
                    result = result[:200] + "..."
                parts.append(f"[T{turn}] RESULT({name}): {result}")
                turn += 1
    except OSError:
        return "(could not read trajectory)"

    text = "\n".join(parts)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n... (truncated)"
    return text if text else "(empty trajectory)"


_ANALYSIS_PROMPT = """\
You are an expert RCA evaluator. Analyze this failed investigation.

## Ground Truth
Correct root cause services: {gt_services}
Fault types: {fault_types}

## Agent's Conclusion
Agent identified: {agent_services}

## Trajectory
{trajectory}

## Task
Identify WHERE the investigation went wrong. For each divergence point:
1. What the agent did
2. What it should have done
3. Category (missed_metric, red_herring, premature_conclusion, wrong_service_focus, insufficient_evidence, correlation_confusion, ignored_anomaly)

Respond as JSON:
{{"divergence_points": [{{"turn_index": 0, "description": "...", "should_have_done": "...", "category": "..."}}], "key_lesson": "one sentence"}}
"""


async def observe_case(
    *,
    case_id: str,
    data_dir: str,
    agent_response: str,
    trajectory_path: str,
    provider_tuple: tuple[str, dict[str, Any]],
) -> DivergenceReport:
    """Analyze one failed case using an AgentM session."""
    from agentm.core.abi.session_config import AgentSessionConfig
    from agentm.core.abi.loop import LoopConfig
    from agentm.core.runtime.session import AgentSession
    from agentm.core.runtime.session_factory import create_agent_session

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

    trajectory = _load_trajectory_summary(trajectory_path)
    prompt = _ANALYSIS_PROMPT.format(
        gt_services=", ".join(gt_services),
        fault_types=", ".join(fault_types),
        agent_services=", ".join(agent_services) or "(none)",
        trajectory=trajectory,
    )

    config = AgentSessionConfig(
        cwd=data_dir,
        provider=provider_tuple,
        scenario="local",
        loop_config=LoopConfig(max_turns=2),
    )
    session = await create_agent_session(AgentSession, config)
    try:
        messages = await session.prompt(prompt)
        text = ""
        from agentm.core.abi.messages import AssistantMessage, TextContent
        for msg in messages:
            if isinstance(msg, AssistantMessage):
                text += "".join(
                    b.text for b in msg.content if isinstance(b, TextContent)
                )
    finally:
        await session.shutdown()

    try:
        # Extract JSON from response (may be wrapped in markdown)
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        analysis = json.loads(text.strip())
    except (json.JSONDecodeError, IndexError):
        _logger.warning("Could not parse observer output for %s", case_id)
        analysis = {
            "divergence_points": [{"turn_index": 0, "description": "analysis failed", "should_have_done": "unknown", "category": "analysis_failed"}],
            "key_lesson": "Could not analyze.",
        }

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
            for dp in analysis.get("divergence_points", [])
        ],
        key_lesson=analysis.get("key_lesson", ""),
    )
