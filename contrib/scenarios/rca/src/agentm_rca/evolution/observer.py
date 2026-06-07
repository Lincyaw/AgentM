"""Observe step: analyze a single RCA case to identify where the agent diverged.

Given a trajectory, the ground-truth causal graph, and the agent's response,
produces a DivergenceReport describing what went wrong and why.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

_logger = logging.getLogger(__name__)


@dataclass
class DivergencePoint:
    turn_index: int
    description: str
    should_have_done: str
    category: str  # free-text: "missed_metric", "red_herring", "wrong_conclusion", etc.


@dataclass
class DivergenceReport:
    case_id: str
    correct: bool
    root_causes_gt: list[str]
    root_causes_agent: list[str]
    divergence_points: list[DivergencePoint] = field(default_factory=list)
    key_lesson: str = ""


def _extract_gt_root_causes(causal_graph: dict[str, Any]) -> list[str]:
    """Extract ground-truth root cause service names from causal_graph.json.

    The `root_causes` field contains entries like:
      {"component": "service|ts-basic-service", ...}
      {"component": "container|ts-basic-service", ...}

    We extract the service name after the pipe delimiter.
    """
    root_causes: list[str] = []
    for rc in causal_graph.get("root_causes", []):
        component = rc.get("component", "")
        if "|" in component:
            service_name = component.split("|", 1)[1]
            # container components may have pod suffixes; normalize to service
            # e.g. "ts-basic-service-67858c87fb-prt77" -> keep as-is since
            # the injection.json uses the short app name. Try to match against
            # injection apps for normalization.
            root_causes.append(service_name)
        elif component:
            root_causes.append(component)
    return root_causes


def _extract_agent_root_causes(agent_response: str) -> list[str]:
    """Extract service names from the agent's submit_final_report output."""
    try:
        data = json.loads(agent_response)
    except (json.JSONDecodeError, TypeError):
        return []

    services: list[str] = []
    for rc in data.get("root_causes", []):
        if isinstance(rc, dict):
            svc = rc.get("service", "")
            if svc:
                services.append(svc)
    return services


def _check_correctness(
    gt_services: list[str],
    agent_services: list[str],
    injection: dict[str, Any],
) -> bool:
    """Check if the agent identified the correct root cause services.

    Uses injection.json's engine_config_summary apps as the canonical
    GT service set. The agent is correct if it identifies at least all
    injected services (superset is acceptable).
    """
    # Primary GT: injected app names from injection.json
    injected_apps = {
        e["app"] for e in injection.get("engine_config_summary", [])
    }

    if not injected_apps:
        # Fallback to causal_graph root_causes
        injected_apps = set(gt_services)

    if not injected_apps:
        return len(agent_services) == 0

    agent_set = {s.lower().strip() for s in agent_services}
    gt_set = {s.lower().strip() for s in injected_apps}

    # Agent is correct if it identified all GT services
    return gt_set.issubset(agent_set)


def _load_trajectory_summary(trajectory_path: str) -> str:
    """Load and summarize the OTLP JSONL trajectory for LLM consumption.

    Extracts tool calls and their results to produce a readable summary
    of the agent's investigation steps.
    """
    path = Path(trajectory_path)
    if not path.exists():
        return "(trajectory file not found)"

    records: list[dict[str, Any]] = []
    try:
        with open(path) as f:
            for raw_line in f:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    parsed = json.loads(raw_line)
                    if isinstance(parsed, dict):
                        records.append(parsed)
                except json.JSONDecodeError:
                    continue
    except OSError:
        return "(could not read trajectory file)"

    # Build a condensed summary of tool calls and key messages
    summary_parts: list[str] = []
    turn_idx = 0
    for record in records:
        record_type = record.get("type", "")
        if record_type == "tool_call":
            name = record.get("name", "unknown")
            args = record.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    pass
            # Truncate large args
            args_str = json.dumps(args, ensure_ascii=False)
            if len(args_str) > 500:
                args_str = args_str[:500] + "..."
            summary_parts.append(f"[Turn {turn_idx}] TOOL CALL: {name}({args_str})")
        elif record_type == "tool_result":
            name = record.get("name", "unknown")
            result = record.get("result", "")
            if isinstance(result, str) and len(result) > 300:
                result = result[:300] + "..."
            summary_parts.append(f"[Turn {turn_idx}] TOOL RESULT ({name}): {result}")
            turn_idx += 1
        elif record_type == "assistant_message":
            content = record.get("content", "")
            if isinstance(content, str) and content.strip():
                if len(content) > 300:
                    content = content[:300] + "..."
                summary_parts.append(f"[Turn {turn_idx}] ASSISTANT: {content}")

    if not summary_parts:
        # Fallback: just dump raw lines (truncated)
        raw = path.read_text()
        if len(raw) > 5000:
            raw = raw[:5000] + "\n... (truncated)"
        return raw

    return "\n".join(summary_parts)


_ANALYSIS_PROMPT = """\
You are an expert RCA (Root Cause Analysis) evaluator. You are given a recorded \
trajectory of an AI agent performing root cause analysis on a microservice incident.

## Ground Truth
The correct root cause services are: {gt_services}
The fault types injected were: {fault_types}

## Agent's Conclusion
The agent concluded these root causes: {agent_services}

## Agent's Investigation Trajectory
{trajectory_summary}

## Task
The agent got the WRONG answer. Analyze the trajectory and identify the specific \
points where the investigation went wrong. For each divergence point, explain:
1. What the agent did at that point
2. What it should have done instead
3. A category for the error (e.g., "missed_metric", "red_herring", "premature_conclusion", \
"wrong_service_focus", "insufficient_evidence", "correlation_confusion", "ignored_anomaly")

Respond in JSON format:
{{
  "divergence_points": [
    {{
      "turn_index": <approximate turn number where the error occurred>,
      "description": "<what went wrong>",
      "should_have_done": "<what the correct action would have been>",
      "category": "<error category>"
    }}
  ],
  "key_lesson": "<one-sentence takeaway about what pattern led to failure>"
}}
"""


async def observe_case(
    *,
    trajectory_path: str,
    data_dir: str,
    agent_response: str,
    provider_config: dict[str, Any],
) -> DivergenceReport:
    """Analyze a single RCA case and produce a divergence report.

    Args:
        trajectory_path: Path to the OTLP JSONL observability file.
        data_dir: Path to the case data directory (has causal_graph.json, injection.json).
        agent_response: JSON string of agent's submit_final_report output.
        provider_config: Dict with keys: base_url, api_key, model.

    Returns:
        A DivergenceReport describing the agent's performance on this case.
    """
    data_path = Path(data_dir)
    case_id = data_path.name

    # Load GT
    causal_graph_path = data_path / "causal_graph.json"
    injection_path = data_path / "injection.json"

    with open(causal_graph_path) as f:
        causal_graph = json.load(f)
    with open(injection_path) as f:
        injection = json.load(f)

    gt_services = _extract_gt_root_causes(causal_graph)
    agent_services = _extract_agent_root_causes(agent_response)

    # Also use injection apps as GT reference
    injected_apps = [
        e["app"] for e in injection.get("engine_config_summary", [])
    ]
    # Combine: injection apps are the authoritative GT
    gt_display = injected_apps if injected_apps else gt_services

    correct = _check_correctness(gt_services, agent_services, injection)

    if correct:
        return DivergenceReport(
            case_id=case_id,
            correct=True,
            root_causes_gt=gt_display,
            root_causes_agent=agent_services,
            divergence_points=[],
            key_lesson="Agent correctly identified root causes.",
        )

    # Agent was wrong — use LLM to analyze where it diverged
    trajectory_summary = _load_trajectory_summary(trajectory_path)
    fault_types = [
        e.get("chaos_type", "unknown")
        for e in injection.get("engine_config_summary", [])
    ]

    prompt = _ANALYSIS_PROMPT.format(
        gt_services=", ".join(gt_display),
        fault_types=", ".join(fault_types),
        agent_services=", ".join(agent_services) if agent_services else "(none submitted)",
        trajectory_summary=trajectory_summary,
    )

    # Truncate if too long for context
    if len(prompt) > 60000:
        # Keep the structure, trim the trajectory
        trajectory_summary = trajectory_summary[:30000] + "\n... (truncated)"
        prompt = _ANALYSIS_PROMPT.format(
            gt_services=", ".join(gt_display),
            fault_types=", ".join(fault_types),
            agent_services=", ".join(agent_services) if agent_services else "(none submitted)",
            trajectory_summary=trajectory_summary,
        )

    client = AsyncOpenAI(
        base_url=provider_config.get("base_url", "http://100.114.89.62:8088/v1"),
        api_key=provider_config.get("api_key", "sk-DLbuXPx8tzeb29atiigWEIXoU9P0xkh_2amQNqJHpMk"),
    )

    try:
        response = await client.chat.completions.create(
            model=provider_config.get("model", "DeepSeek-V4-pro"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        analysis = json.loads(content)
    except Exception as exc:
        _logger.warning("LLM analysis failed for case %s: %s", case_id, exc)
        analysis = {
            "divergence_points": [{
                "turn_index": 0,
                "description": f"LLM analysis unavailable: {exc}",
                "should_have_done": "unknown",
                "category": "analysis_failed",
            }],
            "key_lesson": "Could not analyze trajectory.",
        }

    divergence_points = [
        DivergencePoint(
            turn_index=dp.get("turn_index", 0),
            description=dp.get("description", ""),
            should_have_done=dp.get("should_have_done", ""),
            category=dp.get("category", "unknown"),
        )
        for dp in analysis.get("divergence_points", [])
    ]

    return DivergenceReport(
        case_id=case_id,
        correct=False,
        root_causes_gt=gt_display,
        root_causes_agent=agent_services,
        divergence_points=divergence_points,
        key_lesson=analysis.get("key_lesson", ""),
    )
