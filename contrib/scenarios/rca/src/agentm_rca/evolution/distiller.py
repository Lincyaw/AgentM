"""Distill step: cluster failure patterns → synthesize a SKILL.md.

Uses an AgentM session for the LLM synthesis call.
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


_DISTILL_PROMPT = """\
You are an expert at writing operational methodology for AI agents doing RCA.

## Context
{total_cases} cases analyzed; {total_failures} failures. Most common pattern:

**{top_category}** ({category_count} occurrences)

## Examples

{examples}

## Task
Write a SKILL.md that prevents this failure pattern. Be actionable, specific, ≤300 words.

Respond as JSON:
{{"name": "kebab-case-name", "description": "one line", "tags": ["rca", "..."], "trigger_patterns": ["..."], "body": "markdown body"}}
"""


def _format_examples(reports: list[DivergenceReport], category: str, max_ex: int = 5) -> str:
    examples: list[str] = []
    for r in reports:
        for dp in r.divergence_points:
            if dp.category == category and len(examples) < max_ex:
                examples.append(
                    f"- {r.case_id}: GT={r.root_causes_gt}, Agent={r.root_causes_agent or '(none)'}\n"
                    f"  Wrong: {dp.description}\n"
                    f"  Should: {dp.should_have_done}\n"
                    f"  Lesson: {r.key_lesson}"
                )
                break
    return "\n\n".join(examples) or "(no examples)"


async def distill_skill(
    *,
    reports: list[DivergenceReport],
    provider_tuple: tuple[str, dict[str, Any]],
) -> DistilledSkill | None:
    """Distill failure patterns into a SKILL.md using an AgentM session."""
    from agentm.core.abi.session_config import AgentSessionConfig
    from agentm.core.abi.loop import LoopConfig
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

    prompt = _DISTILL_PROMPT.format(
        total_cases=len(reports),
        total_failures=len(failed),
        top_category=top_cat,
        category_count=cat_count,
        examples=_format_examples(failed, top_cat),
    )

    config = AgentSessionConfig(
        cwd=".",
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
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        result = json.loads(text.strip())
    except (json.JSONDecodeError, IndexError):
        _logger.error("Could not parse distiller output.")
        return None

    name = result.get("name", f"evolved-{top_cat}")
    body = result.get("body", "")
    if not body:
        return None

    tags_s = json.dumps(result.get("tags", ["rca", top_cat]))
    triggers_s = json.dumps(result.get("trigger_patterns", [top_cat]))
    description = result.get("description", f"Address {top_cat} failures")

    content = f"""---
name: {name}
description: '{description}'
tags: {tags_s}
trigger_patterns: {triggers_s}
type: skill
confidence: evolved
version: 1
evidence:
  train_cases: {len(reports)}
  pattern_frequency: {cat_count}
---

{body}
"""

    return DistilledSkill(
        name=name, content=content,
        pattern_category=top_cat,
        train_cases=len(reports),
        pattern_frequency=cat_count,
    )
