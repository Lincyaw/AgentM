# code-health: ignore-file[AM025] -- model JSON is untyped at the boundary
"""Delivery: how a finding reaches the working agent, and the acceptance pass."""

from __future__ import annotations

import json
import time
from functools import lru_cache
from pathlib import Path

import yaml
from loguru import logger

from agentm.core.abi import MessageEnd, Model, StreamFn, text_message
from agentm.core.abi.events import Inject

from .evidence import gather_evidence
from .jsonio import json_object
from .pg_query import PgQuerySource
from .tagger import _content_text
from .triggers import ChecklistItem

_CRITIC_MANIFEST = Path(__file__).parent / "agents" / "critic.yaml"


@lru_cache(maxsize=1)
def _load_critic_system() -> str:
    raw = yaml.safe_load(_CRITIC_MANIFEST.read_text(encoding="utf-8"))
    return str(raw.get("system", ""))


def build_injection(message: str) -> Inject:
    return Inject(messages=(text_message(message, timestamp=time.time()),))


async def verify_item(
    source: PgQuerySource,
    item: ChecklistItem,
    *,
    stream_fn: StreamFn,
    model: Model,
    schema: str = "harbor_live",
) -> tuple[bool, str]:
    """Verify one checklist item against targeted evidence.

    Runs through the session's registered provider, so it shares the host's
    retry policy and token accounting rather than opening a second path to the
    model. Returns (violated, reasoning).
    """
    system = _load_critic_system()
    if not system:
        logger.warning("deliver: critic manifest missing system prompt")
        return False, ""

    evidence_prompt = gather_evidence(source, item, schema=schema)
    if not evidence_prompt:
        return False, "no evidence turns found"

    try:
        stream = stream_fn(
            messages=[text_message(evidence_prompt, timestamp=time.time())],
            model=model,
            tools=[],
            system=system,
        )
        async for event in stream:
            if isinstance(event, MessageEnd):  # code-health: ignore[AM025]
                return _parse_critic_result(_content_text(event.message, 2000))
    except Exception as exc:  # noqa: BLE001
        logger.warning("deliver: critic call failed: {}", exc)
    return False, ""


def _parse_critic_result(text: str) -> tuple[bool, str]:
    if not text:
        return False, ""
    clean = json_object(text)
    try:
        parsed = json.loads(clean)
        violated = bool(parsed.get("violated", False))
        reasoning = str(parsed.get("reasoning", ""))
        return violated, reasoning
    except json.JSONDecodeError:
        if clean.upper() == "OK":
            return False, ""
        return True, clean[:500]
