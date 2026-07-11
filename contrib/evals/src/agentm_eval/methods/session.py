"""Agent session method adapter.

Runs a full agent session with a given scenario and prompt, captures the
session ID, and extracts the terminal tool call result.

Usage::

    from agentm_eval.methods.session import run_session, SessionResult

    result = await run_session(
        scenario="rca:baseline",
        prompt="Investigate the incident...",
        model="azure-gpt",
        cwd="/path/to/sandbox",
    )
    print(result.session_id, result.tool_output)
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any

from agentm.core.abi import (
    AgentMessage,
    AgentSessionConfig,
    AssistantMessage,
    ToolCallBlock,
)
from loguru import logger


@dataclass(frozen=True, slots=True)
class SessionResult:
    """Outcome of a scenario agent session."""

    session_id: str
    root_session_id: str
    messages: list[AgentMessage]
    tool_output: dict[str, Any] | None = None
    error: str | None = None


def extract_tool_args(
    messages: list[AgentMessage], tool_name: str
) -> dict[str, Any] | None:
    """Arguments of the last call to ``tool_name`` in ``messages``."""
    for msg in reversed(messages):
        if not isinstance(msg, AssistantMessage):
            continue
        for block in reversed(msg.content):
            if isinstance(block, ToolCallBlock) and block.name == tool_name:
                return dict(block.arguments)
    return None


async def run_session(
    *,
    scenario: str,
    prompt: str,
    model: str | None = None,
    provider: tuple[str, dict[str, Any]] | None = None,
    cwd: str = ".",
    terminal_tool: str | None = None,
    max_turns: int = 60,
    max_tool_calls_per_turn: int = 20,
    initial_messages: list[Any] | None = None,
    atom_config_overrides: dict[str, dict[str, Any]] | None = None,
    env: dict[str, str] | None = None,
) -> SessionResult:
    """Run a scenario agent session and return the result.

    When ``terminal_tool`` is set, extracts its last invocation arguments
    into ``SessionResult.tool_output``.
    """
    import os

    from agentm.core.abi import LoopConfig
    from agentm.core.runtime import AgentSession

    prev_env: dict[str, str | None] = {}
    if env:
        for k, v in env.items():
            prev_env[k] = os.environ.get(k)
            os.environ[k] = v

    config = AgentSessionConfig(
        cwd=cwd,
        model=model,
        provider=provider,
        scenario=scenario,
        initial_messages=initial_messages or [],
        atom_config_overrides=atom_config_overrides or {},
        loop_config=LoopConfig(
            max_turns=max_turns,
            max_tool_calls_per_turn=max_tool_calls_per_turn,
        ),
    )

    try:
        session = await AgentSession.create(config)
    except Exception as exc:
        logger.warning("session creation failed: {}", exc)
        if env:
            _restore_env(prev_env)
        return SessionResult(
            session_id="",
            root_session_id="",
            messages=[],
            error=f"create failed: {exc}",
        )

    try:
        messages = await session.prompt(prompt)
    except Exception as exc:
        logger.warning("session prompt failed: {}", exc)
        with contextlib.suppress(Exception):
            await session.shutdown()
        if env:
            _restore_env(prev_env)
        return SessionResult(
            session_id=session.session_id,
            root_session_id=session.root_session_id,
            messages=[],
            error=f"prompt failed: {exc}",
        )

    with contextlib.suppress(Exception):
        await session.shutdown()
    if env:
        _restore_env(prev_env)

    tool_output = None
    if terminal_tool:
        tool_output = extract_tool_args(messages, terminal_tool)

    return SessionResult(
        session_id=session.session_id,
        root_session_id=session.root_session_id,
        messages=messages,
        tool_output=tool_output,
    )


def _restore_env(prev: dict[str, str | None]) -> None:
    import os

    for k, v in prev.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
