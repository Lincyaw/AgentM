"""Session-orchestration utilities shared by the live adapter and the
offline replay runner.

The two paths spawn audit children with the same extension list and
drive them to the same terminal tool — the only difference is that the
live adapter calls ``api.spawn_child_session`` (bus-parented) while
replay calls ``create_agent_session`` (top-level). Extension binding,
terminal-tool lookup, and shutdown semantics are identical between
them, so they live here.
"""

from __future__ import annotations

import logging
from typing import Any

from agentm.core.abi.messages import AgentMessage, AssistantMessage, ToolCallBlock

from ._atom_constants import (
    EXTRACTOR_STATE_SERVICE_KEY,
    EXTRACTOR_TOOLS_MODULE,
    SYSTEM_PROMPT_MODULE,
)
from .extractor.state import ExtractionState

_logger = logging.getLogger(__name__)


def find_terminal_tool_arguments(
    messages: list[AgentMessage], tool_name: str
) -> dict[str, Any] | None:
    """Last-match-wins scan for a ``tool_name`` ToolCallBlock's arguments.

    Reverse-iteration matches the live adapter's choice — if a child
    session somehow emitted the terminal tool twice (kernel re-issue,
    flaky stream), we want the *latest* submission.
    """
    for msg in reversed(messages):
        if not isinstance(msg, AssistantMessage):
            continue
        for block in reversed(msg.content):
            if isinstance(block, ToolCallBlock) and block.name == tool_name:
                return dict(block.arguments)
    return None


async def safe_shutdown(session: Any) -> None:
    """Swallow shutdown errors — audit children are best-effort cleanup."""
    try:
        shutdown = getattr(session, "shutdown", None)
        if shutdown is not None:
            await shutdown()
    except Exception:
        _logger.debug("audit-child shutdown failed", exc_info=True)


def bind_extractor_state(
    base_extensions: list[tuple[str, dict[str, Any]]],
    *,
    state: ExtractionState,
    turn_window_json: str,
) -> list[tuple[str, dict[str, Any]]]:
    """Inject a fresh ``ExtractionState`` + substitute ``{TURN_WINDOW_JSON}``.

    Returns a copy; the input list and its config dicts are never mutated.
    Called once per extractor firing, both in live spawning and in replay.
    """
    out: list[tuple[str, dict[str, Any]]] = []
    for module, cfg in base_extensions:
        new_cfg = dict(cfg)
        if module == EXTRACTOR_TOOLS_MODULE:
            new_cfg["state"] = state
            new_cfg.setdefault(EXTRACTOR_STATE_SERVICE_KEY, state)
        elif module == SYSTEM_PROMPT_MODULE:
            prompt = new_cfg.get("prompt")
            if isinstance(prompt, str) and "{TURN_WINDOW_JSON}" in prompt:
                new_cfg["prompt"] = prompt.replace(
                    "{TURN_WINDOW_JSON}", turn_window_json
                )
        out.append((module, new_cfg))
    return out
