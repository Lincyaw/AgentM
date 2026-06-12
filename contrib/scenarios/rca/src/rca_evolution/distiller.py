"""Distill step: spawn a distiller agent to synthesize a SKILL.md.

The distiller gets tools to browse failure reports (browse_reports,
get_report_summary) and submits a structured skill via submit_skill.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import Any

from loguru import logger

from rca_evolution.observer import DivergenceReport

@dataclass
class DistilledSkill:
    name: str
    content: str
    pattern_category: str
    train_cases: int
    pattern_frequency: int
    action: str = "create"
    reason: str = ""

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
got the wrong answer. Your job is to maintain a high-quality skill library — \
not to blindly accumulate skills.

## Workflow

1. Call ``get_existing_skills`` to see what skills already exist.
2. Call ``get_report_summary`` to see the overall failure patterns.
3. Call ``browse_reports`` on the most interesting cases to understand details.
4. Decide the best action:
   - **update** an existing skill if the new failures reveal the same pattern \
but with sharper or broader guidance
   - **create** a new skill only if the failure pattern is genuinely distinct \
from all existing skills
   - **retire** a skill if it is redundant, too generic, or actively harmful
5. Call ``submit_skill`` with the chosen action.

Prefer fewer, stronger skills over many weak ones. A skill should give \
concrete guidance — not generic advice like "be thorough". \
Tell the agent exactly what to check, in what order, when it sees a specific \
pattern of symptoms.
"""

def _collect_existing_skills(skill_dir: str) -> list[dict[str, Any]]:
    """Read all SKILL.md files under the evolved skills directory."""
    from pathlib import Path

    result: list[dict[str, Any]] = []
    base = Path(skill_dir)
    if not base.exists():
        return result
    for skill_path in sorted(base.glob("*/SKILL.md")):
        text = skill_path.read_text()
        name = skill_path.parent.name
        body = text.split("---", 2)[-1].strip() if text.count("---") >= 2 else text
        result.append({"name": name, "content": text, "body_preview": body[:500]})
    return result

async def distill_skill(
    *,
    reports: list[DivergenceReport],
    provider_tuple: tuple[str, dict[str, Any]],
    skill_output_dir: str = "",
) -> DistilledSkill | None:
    """Spawn a distiller agent to synthesize/update/retire a SKILL.md."""
    from agentm.core.abi import (
        AgentSessionConfig,
        AssistantMessage,
        LoopConfig,
        ToolCallBlock,
    )
    from agentm.core.runtime import AgentSession
    from agentm.core.runtime import create_agent_session

    failed = [
        r for r in reports
        if not r.correct
        and not all(dp.category == "analysis_failed" for dp in r.divergence_points)
    ]
    if not failed:
        logger.info("No usable failure reports (all correct or analysis_failed).")
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
        logger.info("Top category %r only %d case(s), need ≥2.", top_cat, cat_count)
        return None

    report_dicts = [r.to_dict() for r in failed]
    summary = _build_report_summary(reports)
    existing_skills = _collect_existing_skills(skill_output_dir) if skill_output_dir else []

    config = AgentSessionConfig(
        cwd=".",
        provider=provider_tuple,
        loop_config=LoopConfig(max_turns=10),
        extensions=[
            ("agentm.extensions.builtin.operations", {"backend": "local"}),
            ("rca_evolution.distiller_atom", {
                "reports": report_dicts,
                "report_summary": summary,
                "existing_skills": existing_skills,
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
        logger.warning("Distiller did not submit a skill.")
        return None

    action = skill_args.get("action", "create")
    name = skill_args.get("name", f"evolved-{top_cat}")

    if action == "retire":
        logger.info("Distiller retiring skill %r: %s", name, skill_args.get("reason", ""))
        return DistilledSkill(
            name=name, content="", pattern_category=top_cat,
            train_cases=len(reports), pattern_frequency=cat_count,
            action="retire", reason=skill_args.get("reason", ""),
        )

    content = _build_skill_content(skill_args, len(reports), cat_count)

    return DistilledSkill(
        name=name,
        content=content,
        pattern_category=top_cat,
        train_cases=len(reports),
        pattern_frequency=cat_count,
        action=action,
        reason=skill_args.get("reason", ""),
    )
