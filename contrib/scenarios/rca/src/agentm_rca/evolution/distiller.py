"""Distill step: cluster failure patterns and synthesize SKILL.md files.

Takes a collection of DivergenceReports from failed cases and produces
a candidate SKILL.md that addresses the most common failure pattern.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI

from agentm_rca.evolution.observer import DivergenceReport

_logger = logging.getLogger(__name__)


@dataclass
class DistilledSkill:
    name: str
    content: str  # full SKILL.md content with frontmatter
    pattern_category: str
    train_cases: int
    pattern_frequency: int


_DISTILL_PROMPT = """\
You are an expert at writing operational methodology guides for AI agents \
performing Root Cause Analysis (RCA) on microservice incidents.

## Context
I have analyzed {total_cases} RCA cases where an AI agent failed to identify \
the correct root causes. The most common failure pattern is:

**Category: {top_category}** ({category_count} occurrences out of {total_failures} failures)

## Examples of this failure pattern:

{examples}

## Task
Write a SKILL.md file that will help an RCA agent avoid this specific failure \
pattern in future investigations. The skill should be:
1. Actionable — give concrete steps the agent should follow
2. Specific — address this exact pattern, not generic advice
3. Concise — no more than 300 words in the body

Respond in JSON format:
{{
  "name": "<kebab-case-name for the skill, e.g. 'verify-before-conclude'>",
  "description": "<one-line description>",
  "tags": ["rca", "<relevant-tags>"],
  "trigger_patterns": ["<situations that should activate this skill>"],
  "body": "<the markdown body content of the skill>"
}}
"""


def _format_examples(reports: list[DivergenceReport], category: str, max_examples: int = 5) -> str:
    """Format example divergence points for the distillation prompt."""
    examples: list[str] = []
    count = 0
    for report in reports:
        for dp in report.divergence_points:
            if dp.category == category and count < max_examples:
                examples.append(
                    f"- Case {report.case_id}:\n"
                    f"  - GT root causes: {', '.join(report.root_causes_gt)}\n"
                    f"  - Agent concluded: {', '.join(report.root_causes_agent) or '(nothing)'}\n"
                    f"  - What went wrong: {dp.description}\n"
                    f"  - Should have done: {dp.should_have_done}\n"
                    f"  - Lesson: {report.key_lesson}"
                )
                count += 1
                break  # one example per case
    return "\n\n".join(examples) if examples else "(no detailed examples available)"


def _build_skill_md(
    name: str,
    description: str,
    tags: list[str],
    trigger_patterns: list[str],
    body: str,
    train_cases: int,
    pattern_frequency: int,
) -> str:
    """Assemble a complete SKILL.md with YAML frontmatter."""
    tags_str = json.dumps(tags)
    triggers_str = json.dumps(trigger_patterns)

    return f"""---
name: {name}
description: '{description}'
tags: {tags_str}
trigger_patterns: {triggers_str}
type: skill
confidence: evolved
version: 1
evidence:
  train_cases: {train_cases}
  pattern_frequency: {pattern_frequency}
---

{body}
"""


async def distill_skill(
    *,
    reports: list[DivergenceReport],
    provider_config: dict[str, Any],
) -> DistilledSkill | None:
    """Distill common failure patterns into a candidate SKILL.md.

    Args:
        reports: DivergenceReports (only incorrect cases should be passed).
        provider_config: Dict with keys: base_url, api_key, model.

    Returns:
        A DistilledSkill if a clear pattern was found, None otherwise.
    """
    # Filter to only incorrect cases
    failed_reports = [r for r in reports if not r.correct]
    if not failed_reports:
        _logger.info("No failed cases to distill from.")
        return None

    # Count category frequencies across all divergence points
    category_counts: Counter[str] = Counter()
    for report in failed_reports:
        # Count each category once per case (not per divergence point)
        seen_categories: set[str] = set()
        for dp in report.divergence_points:
            if dp.category not in seen_categories:
                category_counts[dp.category] += 1
                seen_categories.add(dp.category)

    if not category_counts:
        _logger.info("No divergence points found in failed reports.")
        return None

    # Pick the most common category
    top_category, category_count = category_counts.most_common(1)[0]

    # Need at least 2 cases showing the pattern to distill
    if category_count < 2:
        _logger.info(
            "Top category %r only appeared in %d case(s); need >= 2 to distill.",
            top_category,
            category_count,
        )
        return None

    examples = _format_examples(failed_reports, top_category)

    prompt = _DISTILL_PROMPT.format(
        total_cases=len(reports),
        top_category=top_category,
        category_count=category_count,
        total_failures=len(failed_reports),
        examples=examples,
    )

    client = AsyncOpenAI(
        base_url=provider_config.get("base_url", "http://100.114.89.62:8088/v1"),
        api_key=provider_config.get("api_key", "sk-DLbuXPx8tzeb29atiigWEIXoU9P0xkh_2amQNqJHpMk"),
    )

    try:
        response = await client.chat.completions.create(
            model=provider_config.get("model", "DeepSeek-V4-pro"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        result = json.loads(content)
    except Exception as exc:
        _logger.error("LLM distillation failed: %s", exc)
        return None

    name = result.get("name", f"evolved-{top_category}")
    description = result.get("description", f"Skill addressing {top_category} failures")
    tags = result.get("tags", ["rca", top_category])
    trigger_patterns = result.get("trigger_patterns", [top_category])
    body = result.get("body", "")

    if not body:
        _logger.warning("LLM returned empty body for skill distillation.")
        return None

    skill_content = _build_skill_md(
        name=name,
        description=description,
        tags=tags,
        trigger_patterns=trigger_patterns,
        body=body,
        train_cases=len(reports),
        pattern_frequency=category_count,
    )

    return DistilledSkill(
        name=name,
        content=skill_content,
        pattern_category=top_category,
        train_cases=len(reports),
        pattern_frequency=category_count,
    )
