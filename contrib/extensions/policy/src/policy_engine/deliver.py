# code-health: ignore-file[AM025] -- model JSON is untyped at the boundary
"""Delivery: how a finding reaches the working agent, and the acceptance pass."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml
from loguru import logger

from agentm.core.abi import MessageEnd, Model, StreamFn, text_message
from agentm.core.abi.events import Inject

from .evidence import gather_evidence
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


_ACCEPTANCE_MANIFEST = Path(__file__).parent / "agents" / "acceptance.yaml"


@lru_cache(maxsize=1)
def _load_acceptance_system() -> str:
    raw = yaml.safe_load(_ACCEPTANCE_MANIFEST.read_text(encoding="utf-8"))
    return str(raw.get("system", ""))


@dataclass(frozen=True, slots=True)
class AcceptanceVerdict:
    accepted: bool
    finding: str = ""
    next_step: str = ""

    def as_message(self) -> str:
        """What the agent is told when the submission does not hold up."""
        parts = ["Not quite — one thing is still open before this can go in."]
        if self.finding:
            parts += ["", self.finding]
        if self.next_step:
            parts += ["", f"Next: {self.next_step}"]
        parts += [
            "",
            "Do that, then call `submit` again. If you think this is already "
            "covered, call `submit` and tell me where.",
        ]
        return "\n".join(parts)


def build_acceptance_prompt(
    *,
    task: str,
    summary: str,
    events: Sequence[str],
    concerns: Sequence[str],
) -> str:
    parts = [f"## Task\n{task.strip()}"]
    if summary.strip():
        parts.append(f"## The agent's closing summary\n{summary.strip()}")
    if concerns:
        joined = "\n\n".join(f"- {c.strip()}" for c in concerns)
        parts.append(f"## Process concerns raised during the run\n{joined}")
    parts.append("## The run\n" + "\n".join(events))
    return "\n\n".join(parts)


async def review_submission(
    prompt: str,
    *,
    stream_fn: StreamFn,
    model: Model,
) -> AcceptanceVerdict:
    """Acceptance pass at submit time.

    Accepts on any failure of its own — a broken reviewer must not be able to
    hold a session open.
    """
    system = _load_acceptance_system()
    if not system:
        logger.warning("deliver: acceptance manifest missing system prompt")
        return AcceptanceVerdict(accepted=True)

    try:
        stream = stream_fn(
            messages=[text_message(prompt, timestamp=time.time())],
            model=model,
            tools=[],
            system=system,
        )
        async for event in stream:
            if isinstance(event, MessageEnd):  # code-health: ignore[AM025]
                return _parse_acceptance(_content_text(event.message, 4000))
    except Exception as exc:  # noqa: BLE001
        logger.warning("deliver: acceptance review failed: {}", exc)
    return AcceptanceVerdict(accepted=True)


def _parse_acceptance(text: str) -> AcceptanceVerdict:
    clean = text.strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        parsed = json.loads(clean)
    except json.JSONDecodeError:
        logger.warning("deliver: acceptance verdict was not JSON")
        return AcceptanceVerdict(accepted=True)
    if not isinstance(parsed, Mapping):
        return AcceptanceVerdict(accepted=True)
    return AcceptanceVerdict(
        accepted=bool(parsed.get("accepted", True)),
        finding=str(parsed.get("finding", "")),
        next_step=str(parsed.get("next_step", "")),
    )
