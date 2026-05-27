"""Single-shot orchestration helper: spawn a child :class:`AgentSession`,
send it one prompt, drive it to completion, collect its result.

This is a thin convenience over the ``api.spawn_child_session`` ABI seam —
the one place AgentM exposes nested-session creation. It is **not** an
atom (no ``MANIFEST`` / ``install``, never auto-discovered) and **not** a
core module: its sole consumer is llmharness's
:class:`~llmharness.audit.seams.live.LiveChildRunner` (extractor +
auditor), so it lives here in contrib rather than the core tree. It is a
*convenience layer*, not a new pluggability axis — the axis is
``spawn_child_session`` itself. Should a second in-session consumer with a
genuine need to swap child-running strategy appear, promote it to a
registered service (consumed via ``api.get_service``, no cross-package
import) instead of moving the function.

It imports only :mod:`agentm.core.abi` (``ExtensionAPI``,
``AgentSessionConfig``, messages) and the sibling
:mod:`llmharness.child_collect` — never :mod:`agentm.core.runtime`. The
caller supplies the policy (which extensions / provider, free-text vs.
terminal-tool collect); this helper only composes the config, runs the
child, always shuts it down, and scrapes the result. It does NOT support
multi-turn / mid-flight instruction injection — a consumer that needs
that (e.g. the interactive ``sub_agent`` dispatcher) drives its own loop
over ``spawn_child_session`` directly.

The single knob that unifies the two collect modes is ``terminal_tool``:
when ``None`` the child's final free text is collected into
``final_text``; when set, that terminal tool-call's arguments are
collected into ``terminal_args`` (with ``terminal_called`` recording
whether the call was seen).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Final

from agentm.core.abi.extension import ExtensionAPI
from agentm.core.abi.messages import AgentMessage
from agentm.core.abi.session_config import AgentSessionConfig

from .child_collect import final_assistant_text, terminal_tool_arguments


@dataclass(frozen=True)
class ChildTaskResult:
    """Outcome of one :func:`run_child_task` invocation.

    ``messages`` is the full child trajectory. On the happy path
    ``run_child_task`` populates ``final_text`` when ``terminal_tool`` is
    ``None``, else ``terminal_args`` (with ``terminal_called`` recording
    whether the terminal call was seen). ``error`` is set (and the result
    fields left empty) when spawn / prompt raised.
    """

    messages: list[AgentMessage]
    terminal_called: bool
    terminal_args: dict[str, Any] | None
    final_text: str | None
    error: str | None
    latency_ms: int


async def _safe_shutdown(session: Any) -> None:
    """Swallow shutdown errors — child cleanup is best-effort."""
    try:
        shutdown = getattr(session, "shutdown", None)
        if shutdown is not None:
            await shutdown()
    except Exception:
        pass


async def run_child_task(
    api: ExtensionAPI,
    *,
    extensions: list[tuple[str, dict[str, Any]]],
    provider: tuple[str, dict[str, Any]] | None,
    prompt: str,
    purpose: str,
    terminal_tool: str | None = None,
) -> ChildTaskResult:
    """Spawn a child session, send it ``prompt``, run it to completion,
    collect the result, and always shut the child down.

    Spawn / prompt exceptions are captured into ``ChildTaskResult.error``
    rather than raised, so the happy path never throws; callers that need
    typed error routing inspect ``error`` themselves.
    """
    config = AgentSessionConfig(
        cwd=api.cwd,
        provider=provider,
        extensions=extensions,
        purpose=purpose,
    )
    t0 = time.monotonic()

    def _elapsed_ms() -> int:
        return int((time.monotonic() - t0) * 1000)

    try:
        child = await api.spawn_child_session(config)
    except Exception as exc:
        return ChildTaskResult(
            messages=[],
            terminal_called=False,
            terminal_args=None,
            final_text=None,
            error=str(exc),
            latency_ms=_elapsed_ms(),
        )

    try:
        messages = await child.prompt(prompt)
    except Exception as exc:
        await _safe_shutdown(child)
        return ChildTaskResult(
            messages=[],
            terminal_called=False,
            terminal_args=None,
            final_text=None,
            error=str(exc),
            latency_ms=_elapsed_ms(),
        )

    await _safe_shutdown(child)
    latency_ms = _elapsed_ms()

    if terminal_tool is None:
        return ChildTaskResult(
            messages=messages,
            terminal_called=False,
            terminal_args=None,
            final_text=final_assistant_text(messages),
            error=None,
            latency_ms=latency_ms,
        )

    terminal_args = terminal_tool_arguments(messages, terminal_tool)
    return ChildTaskResult(
        messages=messages,
        terminal_called=terminal_args is not None,
        terminal_args=terminal_args,
        final_text=None,
        error=None,
        latency_ms=latency_ms,
    )


__all__: Final = ["ChildTaskResult", "run_child_task"]
