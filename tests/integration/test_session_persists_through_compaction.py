"""Regression test for the slice-vs-identity bug in ``AgentSession.prompt``.

Bug this prevents: when ``micro_compact`` rewrites the in-flight message
list in place via ``messages[:] = compacted`` (from a
``before_send_to_llm`` handler), the previous slice-by-index logic
``final_messages[pre_run_count:]`` silently dropped the new turn's
assistant / tool_result messages from the SessionManager because the
list shrank below the original length. Per
``.claude/designs/extension-as-scenario.md`` §10b.2 SessionManager owns
durable history; the per-turn context is ephemeral, so identity-based
diff is the only correct way to harvest "what was added".
"""

from __future__ import annotations

import sys
import types
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from agentm.core.kernel import (
    AssistantMessage,
    AssistantStreamEvent,
    MessageEnd,
    Model,
    TextContent,
    ToolCallBlock,
)
from agentm.harness.extension import ProviderConfig
from agentm.harness.resource_loader import InMemoryResourceLoader
from agentm.harness.session import AgentSession, AgentSessionConfig


class _ScriptedProvider:
    """Mirrors ``tests/integration/extension_composition.py`` fixture style."""

    def __init__(self, scripted: list[AssistantMessage]) -> None:
        self._scripted = scripted
        self.calls = 0

    def __call__(
        self,
        *,
        messages: list[Any],
        model: Model,
        tools: list[Any],
        system: str | None = None,
        signal: Any = None,
        thinking: str = "off",
    ) -> AsyncIterator[AssistantStreamEvent]:
        del messages, model, tools, system, signal, thinking
        index = self.calls
        self.calls += 1
        if index < len(self._scripted):
            return self._iter(self._scripted[index])
        return self._iter(
            AssistantMessage(
                role="assistant",
                content=[TextContent(type="text", text="terminal")],
                timestamp=float(index + 1),
                stop_reason="end_turn",
            )
        )

    async def _iter(self, msg: AssistantMessage) -> AsyncIterator[AssistantStreamEvent]:
        yield MessageEnd(message=msg)


def _install_provider_module(name: str, provider: _ScriptedProvider) -> str:
    module = types.ModuleType(name)

    def install(api: Any, _config: dict[str, Any]) -> None:
        api.register_provider(
            "fake-compaction-persist",
            ProviderConfig(
                stream_fn=provider,
                model=Model(
                    id="fake-compaction-persist",
                    provider="fake",
                    context_window=10_000,
                    max_output_tokens=1_000,
                ),
                name="fake-compaction-persist",
            ),
        )

    module.install = install  # type: ignore[attr-defined]
    sys.modules[name] = module
    return name


@pytest.mark.asyncio
async def test_new_turn_messages_persist_when_micro_compact_fires_mid_turn(
    tmp_path: Path,
) -> None:
    """Compaction during a turn must not erase that turn's reply from history.

    Two-turn script: turn 1 calls a tool then ends; turn 2 (under a
    ridiculously low ``threshold_pct``) trips ``micro_compact`` so the
    in-flight message list is rewritten before the LLM call. After the
    turn returns, ``session_manager.get_messages()`` must contain the new
    assistant text "post compact" — under the old slice-by-index logic
    this assertion fails because the slice is empty.
    """

    # Turn 1 (prompt #1): warm-up tool call then end-turn text. This populates
    # SessionManager with several entries so prompt #2's pre_run_count is
    # comfortably larger than the post-compaction list length.
    # Turn 2 (prompt #2): end-turn text only. micro_compact (threshold_pct
    # absurdly small, keep_last=1) rewrites the list before the LLM call,
    # shrinking it well below pre_run_count. Under the old slice-by-index
    # logic, "post compact" would be lost.
    scripted = [
        AssistantMessage(
            role="assistant",
            content=[
                ToolCallBlock(
                    type="tool_call",
                    id="warm",
                    name="add_hypothesis",
                    arguments={"id": "W1", "description": "warm"},
                )
            ],
            timestamp=1.0,
            stop_reason="tool_use",
        ),
        AssistantMessage(
            role="assistant",
            content=[TextContent(type="text", text="warm done")],
            timestamp=2.0,
            stop_reason="end_turn",
        ),
        AssistantMessage(
            role="assistant",
            content=[TextContent(type="text", text="post compact")],
            timestamp=3.0,
            stop_reason="end_turn",
        ),
    ]
    provider = _ScriptedProvider(scripted)
    provider_module = _install_provider_module(
        "tests.integration._fake_compaction_persist_provider", provider
    )

    config = AgentSessionConfig(
        cwd=str(tmp_path),
        extensions=[
            (
                "agentm.extensions.builtin.micro_compact",
                {"threshold_pct": 0.0001, "keep_last": 1},
            ),
            ("agentm.extensions.builtin.tool_hypothesis_store", {}),
        ],
        provider=(provider_module, {}),
        resource_loader=InMemoryResourceLoader(),
    )
    session = await AgentSession.create(config)

    await session.prompt("warm up")
    await session.prompt("now finish")

    persisted = session.session_manager.get_messages()
    texts = [
        block.text
        for msg in persisted
        for block in getattr(msg, "content", [])
        if isinstance(block, TextContent)
    ]
    assert "post compact" in texts, (
        f"final assistant text was lost from SessionManager: {texts!r}"
    )

    await session.shutdown()
