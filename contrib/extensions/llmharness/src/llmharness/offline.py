"""Shared offline primitives for spawning standalone auditor sessions.

This module contains only the low-level building blocks (session spawn,
terminal-tool extraction). The higher-level audit pipeline, offline_audit
convenience wrapper, and StandaloneChildRunner live in
``agentm_eval.harness.offline_audit`` — eval-only infrastructure.
"""

from __future__ import annotations

import contextlib
import json
import time
from dataclasses import dataclass
from typing import Any, Literal

from agentm.core.abi import (
    AgentMessage,
    AgentSessionConfig,
    AssistantMessage,
    ToolCallBlock,
)
from agentm.core.runtime import AgentSession
from loguru import logger

PhaseStatus = Literal["ok", "no_call", "spawn_error", "prompt_error"]


@dataclass(slots=True)
class PhaseResult:
    """Outcome of one standalone auditor phase invocation."""

    output: dict[str, Any] | None
    status: PhaseStatus
    error: str | None
    latency_ms: int
    messages: list[AgentMessage]


@dataclass(frozen=True, slots=True)
class AuditorSettings:
    """Minimal config needed for one offline auditor firing."""

    base_prompt: str | None = None
    tools: tuple[str, ...] | None = None

    @classmethod
    def default(cls) -> AuditorSettings:
        from llmharness.agents.auditor.context import load_auditor_prompt

        return cls(base_prompt=load_auditor_prompt("index"))


def terminal_tool_arguments(
    messages: list[AgentMessage],
    tool_name: str,
) -> dict[str, Any] | None:
    """Extract the arguments of the last call to ``tool_name``."""

    for msg in reversed(messages):
        if not isinstance(msg, AssistantMessage):
            continue
        for block in reversed(msg.content):
            if isinstance(block, ToolCallBlock) and block.name == tool_name:
                return dict(block.arguments)
    return None


async def run_phase_standalone(
    *,
    cwd: str,
    extensions: list[tuple[str, dict[str, Any]]],
    provider: tuple[str, dict[str, Any]] | None = None,
    model: str | None = None,
    payload: dict[str, Any] | str,
    terminal_tool: str,
    purpose: str = "cognitive_audit_offline",
    parent_session_id: str | None = None,
    trace_id: str | None = None,
) -> PhaseResult:
    """Spawn a top-level auditor session and return terminal-tool arguments."""

    config = AgentSessionConfig(
        cwd=cwd,
        model=model,
        provider=provider,
        extensions=extensions,
        purpose=purpose,
        parent_session_id=parent_session_id,
        root_session_id=trace_id,
    )
    t0 = time.monotonic()
    try:
        session = await AgentSession.create(config)
    except Exception as exc:
        logger.debug("offline: caught exception: {}", exc)
        return PhaseResult(
            output=None,
            status="spawn_error",
            error=str(exc),
            latency_ms=int((time.monotonic() - t0) * 1000),
            messages=[],
        )

    try:
        if isinstance(payload, str):
            user_message = payload
        else:
            user_message = json.dumps(payload, ensure_ascii=False, default=str)
        messages = await session.prompt(user_message)
    except Exception as exc:
        logger.debug("offline: caught exception: {}", exc)
        with contextlib.suppress(Exception):
            await session.shutdown()
        return PhaseResult(
            output=None,
            status="prompt_error",
            error=str(exc),
            latency_ms=int((time.monotonic() - t0) * 1000),
            messages=[],
        )

    with contextlib.suppress(Exception):
        await session.shutdown()
    latency_ms = int((time.monotonic() - t0) * 1000)
    args = terminal_tool_arguments(messages, terminal_tool)
    if args is None:
        return PhaseResult(
            output=None,
            status="no_call",
            error=f"terminal tool {terminal_tool!r} was not called",
            latency_ms=latency_ms,
            messages=messages,
        )
    return PhaseResult(
        output=args,
        status="ok",
        error=None,
        latency_ms=latency_ms,
        messages=messages,
    )


def flatten_assistant_blocks(messages: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, AssistantMessage):
            continue
        for block in msg.content:
            if isinstance(block, ToolCallBlock):
                out.append(
                    {
                        "type": "tool_call",
                        "name": block.name,
                        "arguments": dict(block.arguments),
                    }
                )
            elif hasattr(block, "text"):
                btype = getattr(block, "type", "text")
                out.append({"type": btype, "text": block.text})
    return out


__all__ = [
    "AuditorSettings",
    "PhaseResult",
    "flatten_assistant_blocks",
    "run_phase_standalone",
    "terminal_tool_arguments",
]
