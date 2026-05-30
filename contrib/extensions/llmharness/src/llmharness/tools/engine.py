"""Run a single audit phase as a top-level AgentM session.

Live path: parent session spawns a child via ``api.spawn_child_session``.
Replay path: this module creates a brand-new top-level session with the
same extensions and pumps the recorded payload through it.

Both paths share :mod:`llmharness.audit.{extractor,auditor}.extensions`
composers, the terminal-tool scraper in
:mod:`llmharness.child_collect`, and the shutdown helper in
:mod:`llmharness.audit.seams.session`, so the system prompt, tool
surface, payload shape, and collect contract are identical to a live
firing.

Boundary: this module is a host-side driver (not a §11 atom — no
``MANIFEST`` / ``install`` pair, never named in a scenario manifest), so
the ``agentm.core.runtime.*`` imports below are intentional and must not
be removed. If this file is ever promoted to an atom, route session
construction through ``ExtensionAPI`` instead.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from agentm.core.abi.messages import AgentMessage
from agentm.core.abi.session_config import AgentSessionConfig

# Both imports are runtime-required: ``AgentSession`` is passed as a
# positional argument to ``create_agent_session(AgentSession, config)``,
# so it cannot be hidden behind ``TYPE_CHECKING``.
from agentm.core.runtime.session import AgentSession
from agentm.core.runtime.session_factory import create_agent_session

from ..audit.seams.session import safe_shutdown
from ..child_collect import nudge_until_tool_call, terminal_tool_arguments
from ..replay.record import Status


@dataclass
class PhaseResult:
    """Outcome of one phase invocation.

    ``status`` is keyed to :data:`record.Status` so chain replay can
    diff against the recorded record's ``status`` field directly.
    """

    output: dict[str, Any] | None
    status: Status
    error: str | None
    latency_ms: int
    messages: list[AgentMessage]


async def run_phase_standalone(
    *,
    cwd: str,
    extensions: list[tuple[str, dict[str, Any]]],
    provider: tuple[str, dict[str, Any]] | None,
    payload: dict[str, Any] | str,
    terminal_tool: str,
    purpose: str = "cognitive_audit_replay",
    parent_session_id: str | None = None,
    trace_id: str | None = None,
) -> PhaseResult:
    """Spawn a top-level session, send ``payload`` as a user message,
    return the ``terminal_tool`` arguments.

    The live path's ``api.spawn_child_session`` produces a bus-parented
    child; this produces an isolated root session — same tool surface
    and prompt, no parent linkage.

    ``payload`` is either a dict (JSON-serialised and sent as the user
    message) or a string (sent verbatim). Callers that need a per-phase
    directive prefix construct the combined string themselves and pass
    it in — this engine no longer special-cases any terminal tool.

    ``parent_session_id`` / ``trace_id`` are optional linkage fields:
    when set, the child session's observability file carries the parent
    and trace ids so ``agentm trace index`` can associate it with the
    originating experiment.
    """
    config = AgentSessionConfig(
        cwd=cwd,
        provider=provider,
        extensions=extensions,
        purpose=purpose,
        parent_session_id=parent_session_id,
        root_session_id=trace_id,
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
        if isinstance(payload, str):
            user_message = payload
        else:
            user_message = json.dumps(payload, ensure_ascii=False, default=str)
        messages = await session.prompt(user_message)
    except Exception as exc:
        await safe_shutdown(session)
        return PhaseResult(
            output=None,
            status="prompt_error",
            error=str(exc),
            latency_ms=int((time.monotonic() - t0) * 1000),
            messages=[],
        )

    messages = await nudge_until_tool_call(session.prompt, messages, terminal_tool)

    await safe_shutdown(session)
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
        output=args, status="ok", error=None, latency_ms=latency_ms, messages=messages
    )
