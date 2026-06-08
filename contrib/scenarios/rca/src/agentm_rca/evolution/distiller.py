"""Distill step: spawn a distiller agent to synthesize a SKILL.md.

The distiller gets tools to browse failure reports (browse_reports,
get_report_summary) and submits a structured skill via submit_skill.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass
from typing import Any

from agentm_rca.evolution.observer import DivergenceReport

_logger = logging.getLogger(__name__)


@dataclass
class DistilledSkill:
    name: str
    content: str
    pattern_category: str
    train_cases: int
    pattern_frequency: int


def _build_report_summary(reports: list[DivergenceReport]) -> str:
    """Aggregate stats for get_report_summary tool."""
    failed = [r for r in reports if not r.correct]
    cats: Counter[str] = Counter()
    for r in failed:
        seen: set[str] = set()
        for dp in r.divergence_points:
            if dp.category not in seen:
                cats[dp.category] += 1
                seen.add(dp.category)

    lines = [
        f"Total reports: {len(reports)}",
        f"Failed: {len(failed)}",
        f"Correct: {len(reports) - len(failed)}",
        "",
        "Category frequencies:",
    ]
    for cat, count in cats.most_common():
        lines.append(f"  {cat}: {count} cases")

    lines.append("")
    lines.append("Per-case summaries:")
    for i, r in enumerate(failed):
        cats_str = ", ".join(dp.category for dp in r.divergence_points)
        lines.append(
            f"  [{i}] {r.case_id}: GT={r.root_causes_gt}, "
            f"Agent={r.root_causes_agent or ['(none)']}, "
            f"categories=[{cats_str}], lesson={r.key_lesson[:80]}"
        )

    return "\n".join(lines)


def _build_skill_content(args: dict[str, Any], train_cases: int, pattern_freq: int) -> str:
    tags_s = json.dumps(args.get("tags", ["rca"]))
    triggers_s = json.dumps(args.get("trigger_patterns", []))
    return f"""---
name: {args["name"]}
description: '{args.get("description", "")}'
tags: {tags_s}
trigger_patterns: {triggers_s}
type: skill
confidence: evolved
version: 1
evidence:
  train_cases: {train_cases}
  pattern_frequency: {pattern_freq}
---

{args.get("body", "")}
"""


_DISTILLER_PROMPT = """\
You are an expert at writing operational methodology for AI agents doing RCA \
(Root Cause Analysis) on microservice incidents.

You have access to failure analysis reports from cases where an RCA agent \
got the wrong answer. Your job:

1. Call ``get_report_summary`` to see the overall failure patterns.
2. Call ``browse_reports`` on the most interesting cases to understand details.
3. Identify the dominant failure pattern that a single SKILL.md could address.
4. Call ``submit_skill`` with an actionable, specific skill (≤300 words body) \
that would help the agent avoid this class of failure in future investigations.

The skill should give concrete guidance — not generic advice like "be thorough". \
Tell the agent exactly what to check, in what order, when it sees a specific \
pattern of symptoms.
"""


async def distill_skill(
    *,
    reports: list[DivergenceReport],
    provider_tuple: tuple[str, dict[str, Any]],
) -> DistilledSkill | None:
    """Spawn a distiller agent to synthesize a SKILL.md from failure reports."""
    from agentm.core.abi.session_config import AgentSessionConfig
    from agentm.core.abi.loop import LoopConfig
    from agentm.core.abi.messages import AssistantMessage, ToolCallBlock
    from agentm.core.runtime.session import AgentSession
    from agentm.core.runtime.session_factory import create_agent_session

    failed = [r for r in reports if not r.correct]
    if not failed:
        return None

    cats: Counter[str] = Counter()
    for r in failed:
        seen: set[str] = set()
        for dp in r.divergence_points:
            if dp.category not in seen:
                cats[dp.category] += 1
                seen.add(dp.category)

    if not cats:
        return None

    top_cat, cat_count = cats.most_common(1)[0]
    if cat_count < 2:
        _logger.info("Top category %r only %d case(s), need ≥2.", top_cat, cat_count)
        return None

    report_dicts = [r.to_dict() for r in failed]
    summary = _build_report_summary(reports)

    config = AgentSessionConfig(
        cwd=".",
        provider=provider_tuple,
        scenario="local",
        loop_config=LoopConfig(max_turns=10),
        extra_extensions=[
            ("agentm_rca.evolution.distiller_atom", {
                "reports": report_dicts,
                "report_summary": summary,
            }),
        ],
    )

    session = await create_agent_session(AgentSession, config)
    try:
        messages = await session.prompt(_DISTILLER_PROMPT)
    finally:
        await session.shutdown()

    skill_args: dict[str, Any] | None = None
    for msg in reversed(messages):
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, ToolCallBlock) and block.name == "submit_skill":
                    skill_args = block.arguments
                    break
            if skill_args:
                break

    if skill_args is None:
        _logger.warning("Distiller did not submit a skill.")
        return None

    name = skill_args.get("name", f"evolved-{top_cat}")
    content = _build_skill_content(skill_args, len(reports), cat_count)

    return DistilledSkill(
        name=name,
        content=content,
        pattern_category=top_cat,
        train_cases=len(reports),
        pattern_frequency=cat_count,
    )
