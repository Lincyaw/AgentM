"""Shared oracle infrastructure — one plain-JSON model call, used by every
model-judgment pass (Pass 2 identity/edges, Pass 3 constraints/value-fidelity).

The passes own the *questions* (their instructions + payload shaping); this
module owns the *mechanism*: spawn a one-shot child session, prompt it, pull
the JSON list under a key. Kept separate so no pass depends on another just to
reach the model, and so the §11 rule (offline callers import
``AgentSession.create`` on their own side) has a single home.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Coroutine
from typing import Any

from loguru import logger

# Session factory: ``async (config) -> session`` where session has
# ``.prompt(text)`` and ``.shutdown()``. The atom passes
# ``api.spawn_child_session``; offline callers pass ``AgentSession.create``.
type SessionFactory = Callable[[Any], Coroutine[Any, Any, Any]]


def load_prompt(name: str) -> str:
    from importlib.resources import files
    text = files("trajectory_index.prompts").joinpath(f"{name}.md").read_text(encoding="utf-8")
    return text.strip()


def _index_by_id(raw: list[Any]) -> dict[int, dict[str, Any]]:
    """Index a list of id-keyed dicts by their ``id`` field (missing → skipped)."""
    by_id: dict[int, dict[str, Any]] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        with contextlib.suppress(TypeError, ValueError, KeyError):
            by_id[int(item["id"])] = item
    return by_id


def _safe_float(item: dict[str, Any], key: str, default: float = 0.0) -> float:
    """Coerce a model-returned field to float, defaulting on garbage."""
    try:
        return float(item.get(key, default))
    except (TypeError, ValueError):
        return default


async def _ask_model(
    prompt_name: str,
    payload: str,
    model: str | None,
    session_factory: SessionFactory,
    purpose: str = "alias_resolution",
    key: str = "verdicts",
) -> list[Any] | None:
    """One plain-JSON model call; returns the list under ``key`` or None.

    ``session_factory`` creates a session from an ``AgentSessionConfig``.
    The atom passes ``api.spawn_child_session``; offline callers pass
    ``AgentSession.create`` (imported on their side, not here — §11).
    """
    from pathlib import Path

    from agentm.core.abi import (
        AgentSessionConfig,
        AssistantMessage,
        LoopConfig,
        TextContent,
    )

    from .pass1_nodes.serialize import extract_json

    instructions = load_prompt(prompt_name)

    config = AgentSessionConfig(
        cwd=str(Path.cwd()),
        model=model,
        scenario="minimal",
        purpose=purpose,
        loop_config=LoopConfig(max_turns=1),
        log_trace_command=False,
    )
    session = await session_factory(config)
    try:
        messages = await session.prompt(f"{instructions}\n\n{payload}")
    finally:
        with contextlib.suppress(Exception):
            await session.shutdown()

    text = "".join(
        block.text
        for msg in messages
        if isinstance(msg, AssistantMessage)
        for block in msg.content
        if isinstance(block, TextContent)
    )
    obj = extract_json(text)
    if obj is None:
        logger.warning("{}: model returned no parseable JSON", purpose)
        return None
    verdicts = obj.get(key)
    if not isinstance(verdicts, list):
        logger.warning("{}: JSON missing '{}' list", purpose, key)
        return None
    return verdicts
