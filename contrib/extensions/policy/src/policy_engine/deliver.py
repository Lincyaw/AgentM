"""Delivery: how a trigger's finding reaches the working agent."""

from __future__ import annotations

import json
import time
from functools import lru_cache
from pathlib import Path

import yaml
from loguru import logger

from agentm.core.abi import text_message
from agentm.core.abi.events import Inject

from .evidence import gather_evidence
from .llm import call_llm
from .pg_query import PgQuerySource
from .triggers import ChecklistItem

_CRITIC_MANIFEST = Path(__file__).parent / "agents" / "critic.yaml"


@lru_cache(maxsize=1)
def _load_critic_system() -> str:
    raw = yaml.safe_load(_CRITIC_MANIFEST.read_text(encoding="utf-8"))
    return str(raw.get("system", ""))


def build_injection(message: str) -> Inject:
    return Inject(messages=(text_message(message, timestamp=time.time()),))


def verify_item(
    source: PgQuerySource,
    item: ChecklistItem,
    *,
    schema: str = "harbor_live",
    model_name: str | None = None,
) -> tuple[bool, str]:
    """Verify one checklist item against targeted evidence.

    Uses direct LLM call, no child session.
    Returns (violated, reasoning).
    """
    system = _load_critic_system()
    if not system:
        logger.warning("deliver: critic manifest missing system prompt")
        return False, ""

    evidence_prompt = gather_evidence(source, item, schema=schema)
    if not evidence_prompt:
        return False, "no evidence turns found"

    result = call_llm(system, evidence_prompt, max_tokens=300, model_name=model_name)
    if result is None:
        return False, ""

    return _parse_critic_result(result)


def _parse_critic_result(text: str) -> tuple[bool, str]:
    if not text:
        return False, ""
    clean = text.strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        parsed = json.loads(clean)
        violated = bool(parsed.get("violated", False))
        reasoning = str(parsed.get("reasoning", ""))
        return violated, reasoning
    except json.JSONDecodeError:
        if clean.upper() == "OK":
            return False, ""
        return True, clean[:500]
