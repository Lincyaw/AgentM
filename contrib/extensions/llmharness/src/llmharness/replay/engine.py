"""Run a single audit phase as a top-level AgentM session.

Live path: parent session spawns a child via ``api.spawn_child_session``.
Replay path: this module creates a brand-new top-level session with the
same extensions and pumps the recorded payload through it.

Both paths share :mod:`llmharness.audit.{extractor,auditor}.extensions`
composers, so the system prompt, tool surface, and payload schema are
identical to a live firing.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from agentm.core.abi.messages import AgentMessage, AssistantMessage, ToolCallBlock
from agentm.core.abi.session_config import AgentSessionConfig
from agentm.core.runtime.session import AgentSession
from agentm.core.runtime.session_factory import create_agent_session

_logger = logging.getLogger(__name__)


@dataclass
class PhaseResult:
    """Outcome of one phase invocation.

    ``output`` carries the ``submit_*`` tool arguments on success;
    ``status`` enumerates the failure modes so replay tooling can diff
    them against the recorded record's ``status`` field.
    """

    output: dict[str, Any] | None
    status: str  # "ok" | "no_call" | "spawn_error" | "prompt_error"
    error: str | None
    latency_ms: int
    messages: list[AgentMessage]


async def run_phase_standalone(
    *,
    cwd: str,
    extensions: list[tuple[str, dict[str, Any]]],
    provider: tuple[str, dict[str, Any]] | None,
    payload: dict[str, Any],
    terminal_tool: str,
    purpose: str = "cognitive_audit_replay",
) -> PhaseResult:
    """Spawn a top-level session with ``extensions``, send ``payload`` as
    a user message, and return the ``terminal_tool`` arguments.

    Exists so the offline replay path and the live path share one
    invocation primitive. The live path uses ``api.spawn_child_session``
    (it has a parent bus); replay calls this and gets an isolated root
    session — same tool surface, same prompt, same payload shape.
    """
    config = AgentSessionConfig(
        cwd=cwd,
        provider=provider,
        extensions=extensions,
        purpose=purpose,
    )
    t0 = time.monotonic()
    try:
        session = await create_agent_session(AgentSession, config)
    except Exception as exc:
        return PhaseResult(
            output=None,
            status="spawn_error",
            error=str(exc),
            latency_ms=int((time.monotonic() - t0) * 1000),
            messages=[],
        )

    try:
        messages = await session.prompt(
            json.dumps(payload, ensure_ascii=False, default=str)
        )
    except Exception as exc:
        await _safe_shutdown(session)
        return PhaseResult(
            output=None,
            status="prompt_error",
            error=str(exc),
            latency_ms=int((time.monotonic() - t0) * 1000),
            messages=[],
        )

    await _safe_shutdown(session)
    latency_ms = int((time.monotonic() - t0) * 1000)

    args = _find_terminal_tool_arguments(messages, terminal_tool)
    if args is None:
        return PhaseResult(
            output=None,
            status="no_call",
            error=f"terminal tool {terminal_tool!r} was not called",
            latency_ms=latency_ms,
            messages=messages,
        )
    return PhaseResult(
        output=args, status="ok", error=None, latency_ms=latency_ms, messages=messages
    )


def _find_terminal_tool_arguments(
    messages: list[AgentMessage], tool_name: str
) -> dict[str, Any] | None:
    for msg in messages:
        if not isinstance(msg, AssistantMessage):
            continue
        for block in msg.content:
            if isinstance(block, ToolCallBlock) and block.name == tool_name:
                args = block.arguments
                return dict(args) if isinstance(args, dict) else None
    return None


async def _safe_shutdown(session: Any) -> None:
    try:
        shutdown = getattr(session, "shutdown", None)
        if shutdown is not None:
            await shutdown()
    except Exception:
        _logger.debug("replay session shutdown failed", exc_info=True)
