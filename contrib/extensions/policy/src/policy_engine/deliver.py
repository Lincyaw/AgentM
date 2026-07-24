"""Delivery: how a trigger's finding reaches the working agent."""

from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path

import yaml
from loguru import logger

from agentm.core.abi import AtomAPI, text_message
from agentm.core.abi.events import Inject

from .tagger import _extract_result_text

_CRITIC_MANIFEST = Path(__file__).parent / "agents" / "critic.yaml"


@lru_cache(maxsize=1)
def _load_critic_system() -> str:
    raw = yaml.safe_load(_CRITIC_MANIFEST.read_text(encoding="utf-8"))
    return str(raw.get("system", ""))


def build_injection(message: str) -> Inject:
    return Inject(messages=(text_message(message, timestamp=time.time()),))


async def run_critic(
    api: AtomAPI,
    *,
    questions: list[str],
    evidence: list[str],
) -> str | None:
    system = _load_critic_system()
    if not system:
        logger.warning("deliver: critic manifest missing system prompt")
        return None

    prompt_lines = ["Checklist questions:"]
    prompt_lines.extend(f"- {q}" for q in questions)
    if evidence:
        prompt_lines.append("")
        prompt_lines.append("Structural observations:")
        prompt_lines.extend(f"- {e}" for e in evidence)
    prompt = "\n".join(prompt_lines)

    try:
        child = await api.spawn(
            purpose="policy-critic", system=system, tools=[], max_turns=1
        )
        result = await child.prompt(prompt, origin="policy_engine")
    except Exception as exc:  # noqa: BLE001
        logger.warning("deliver: critic failed: {}", exc)
        return None

    text = _extract_result_text(result)
    if not text or text.strip() == "OK":
        return None
    return (
        "Process review from the checklist critic (based on your own "
        f"commands and edits):\n{text.strip()[:1200]}"
    )
