# code-health: ignore-file[AM025] -- event payloads are untyped at the boundary
"""Delivery: how a trigger's finding reaches the working agent.

Two channels, matching the two intervention forms:

- ``inject``: append a user message into the main agent's own loop via the
  ``Inject`` action (zero latency, zero extra tokens; the message is a
  template quoting the agent's own commands).
- ``subagent``: spawn a reviewer child session that reads the checklist
  questions plus a digest of the trajectory, and inject its verdict only
  when it confirms concrete violations.
"""

from __future__ import annotations

import time

from loguru import logger

from agentm.core.abi import AtomAPI, TextContent, UserMessage
from agentm.core.abi.events import Inject

from .signals import TrajectoryState
from .triggers import ChecklistItem

_CRITIC_SYSTEM = """You are a process reviewer for a software-engineering agent.
You receive checklist questions, structural observations, and a digest of the
agent's own actions (commands it ran, files it edited). Judge each question
strictly against that evidence.

Rules:
- You review process, not the task solution. Never propose code, name fixes,
  or hint at what the correct change is.
- Cite the agent's own commands or files for every violation you report.
- If the evidence satisfies a question, do not mention it.
- Output at most 700 characters. If nothing is violated, output exactly: OK
"""

_DIGEST_COMMANDS = 30


def build_injection(message: str) -> Inject:
    return Inject(
        messages=(
            UserMessage(
                role="user",
                content=[TextContent(type="text", text=message)],
                timestamp=time.time(),
            ),
        )
    )


def trajectory_digest(state: TrajectoryState) -> str:
    lines: list[str] = []
    edited = state.mutated_paths[-20:]
    if edited:
        lines.append(
            f"files edited ({len(state.mutated_paths)} total, last {len(edited)}):"
        )
        lines.extend(f"  {path}" for path in edited)
    lines.append("commands executed (most recent):")
    for record in state.execs[-_DIGEST_COMMANDS:]:
        status = "ok" if record.exit_code == 0 else f"exit={record.exit_code}"
        lines.append(f"  [{status}] {record.raw[:160]}")
    return "\n".join(lines)


def self_check_message(
    items: tuple[ChecklistItem, ...], evidence: tuple[str, ...]
) -> str:
    """Form one: hand the checklist to the agent for its own review."""

    lines = [
        "Process check from the validation monitor: before concluding, audit "
        "your own process against each point below. Where a point is not "
        "satisfied, act on it first."
    ]
    for index, item in enumerate(items, start=1):
        lines.append(f"{index}. {item.check}")
    if evidence:
        lines.append("")
        lines.append("Structural observations from this session:")
        lines.extend(f"- {fact}" for fact in evidence)
    return "\n".join(lines)


async def run_critic(
    api: AtomAPI,
    *,
    items: tuple[ChecklistItem, ...],
    evidence: tuple[str, ...],
    state: TrajectoryState,
) -> str | None:
    """Form two: spawn a reviewer child; returns its verdict text or None."""

    prompt_lines = ["Checklist questions:"]
    prompt_lines.extend(f"- ({item.item_id}) {item.check}" for item in items)
    if evidence:
        prompt_lines.append("")
        prompt_lines.append("Structural observations:")
        prompt_lines.extend(f"- {fact}" for fact in evidence)
    prompt_lines.append("")
    prompt_lines.append(trajectory_digest(state))
    prompt = "\n".join(prompt_lines)

    try:
        child = await api.spawn(
            purpose="policy-critic",
            system=_CRITIC_SYSTEM,
            tools=[],
            max_turns=1,
        )
        result = await child.prompt(prompt, origin="policy_engine")
    except Exception as exc:  # noqa: BLE001 -- critic failure must never kill the loop
        logger.warning("policy critic failed: {}", exc)
        return None
    text = _result_text(result)
    if not text or text.strip() == "OK":
        return None
    return (
        "Process review from the checklist critic (based on your own "
        f"commands and edits):\n{text.strip()[:1200]}"
    )


def _result_text(result: object) -> str:
    text = getattr(result, "text", None)  # code-health: ignore[AM021]
    if isinstance(text, str):  # code-health: ignore[AM025]
        return text
    return str(result) if result is not None else ""
