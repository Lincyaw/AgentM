"""Integration: resume-without-prompt + reminder_seed delivers REMINDER_DELIVERED.

Fail-stop position: when ``agentm --resume <sid>`` is called without a
positional prompt and ``llmharness.replay.reminder_seed`` is mounted, the
seed atom must observe the synthetic ``decide_turn_action`` that
``AgentSession.tick`` fires, return ``Inject([reminder_msg])``, persist a
``REMINDER_DELIVERED`` entry, and let the loop run one assistant turn.

This is the bridge between AgentM's tick semantics and the harness's
prefix-replay flow — if either side regresses, the entire
``agent-from-reminder`` CLI silently emits a command that exits with
``NoPendingInput`` and writes nothing.
"""

from __future__ import annotations

import sys
import types
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from agentm.core.abi import (
    AssistantMessage,
    MessageEnd,
    Model,
    TextContent,
)
from agentm.core.abi.extension import ProviderConfig
from agentm.core.abi.session_config import AgentSessionConfig
from agentm.core.runtime.session import AgentSession

from llmharness.audit.entry_types import REMINDER_DELIVERED


async def _stream_fn(
    *,
    messages: list[Any],
    model: Model,
    tools: list[Any],
    system: str | None = None,
    signal: Any = None,
    thinking: str = "off",
) -> AsyncIterator[Any]:
    del messages, model, tools, system, signal, thinking
    yield MessageEnd(
        message=AssistantMessage(
            role="assistant",
            content=[TextContent(type="text", text="stub-ack")],
            timestamp=0.0,
            stop_reason="end_turn",
        )
    )


def _register_stub_provider(module_name: str) -> None:
    module = types.ModuleType(module_name)

    def install(api: Any, config: dict[str, Any]) -> None:
        del config
        api.register_provider(
            "stub",
            ProviderConfig(
                stream_fn=_stream_fn,
                model=Model(
                    id="stub-model",
                    provider="stub",
                    context_window=1024,
                    max_output_tokens=64,
                ),
                name="stub",
            ),
        )

    module.install = install
    sys.modules[module_name] = module


@pytest.mark.asyncio
async def test_tick_with_reminder_seed_delivers_reminder(tmp_path: Path) -> None:
    module_name = f"tests._reminder_seed_tick_{id(tmp_path)}"
    _register_stub_provider(module_name)
    try:
        session = await AgentSession.create(
            AgentSessionConfig(
                cwd=str(tmp_path),
                provider=(module_name, {}),
                extensions=[
                    ("agentm.extensions.builtin.operations_local", {}),
                    (
                        "llmharness.replay.reminder_seed",
                        {"text": "do the thing"},
                    ),
                ],
            )
        )
        try:
            result = await session.tick()
        finally:
            await session.shutdown()

        # An assistant turn ran post-inject.
        assert any(
            isinstance(m, AssistantMessage)
            and any(
                isinstance(b, TextContent) and "stub-ack" in b.text
                for b in m.content
            )
            for m in result
        )

        # A REMINDER_DELIVERED entry was persisted on the session log.
        entries = list(session.session_manager.get_active_branch())
        kinds = [e.type for e in entries]
        assert REMINDER_DELIVERED in kinds
        delivered = next(e for e in entries if e.type == REMINDER_DELIVERED)
        assert delivered.payload == {"text": "do the thing"}
    finally:
        sys.modules.pop(module_name, None)
