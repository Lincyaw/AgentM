"""Integration coverage for the LLM-driven compaction extension."""

from __future__ import annotations

import sys
import types
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from agentm.core.compaction import SUMMARIZATION_SYSTEM_PROMPT
from agentm.core.kernel import (
    AgentMessage,
    AssistantMessage,
    AssistantStreamEvent,
    MessageEnd,
    Model,
    TextContent,
    text_message,
)
from agentm.harness.extension import ProviderConfig
from agentm.harness.resource_loader import InMemoryResourceLoader
from agentm.harness.session import AgentSession, AgentSessionConfig


class _SummaryThenReplyProvider:
    def __init__(self, summary_text: str, reply_text: str) -> None:
        self._summary = summary_text
        self._reply = reply_text
        self.calls = 0
        self.seen_messages: list[list[Any]] = []
        self.seen_systems: list[str | None] = []

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
        del model, tools, signal, thinking
        self.calls += 1
        self.seen_messages.append(list(messages))
        self.seen_systems.append(system)
        text = self._summary if self.calls == 1 else self._reply
        return self._iter(
            AssistantMessage(
                role="assistant",
                content=[TextContent(type="text", text=text)],
                timestamp=float(self.calls),
                stop_reason="end_turn",
            )
        )

    async def _iter(self, msg: AssistantMessage) -> AsyncIterator[AssistantStreamEvent]:
        yield MessageEnd(message=msg)


def _install_provider_module(name: str, provider: _SummaryThenReplyProvider) -> str:
    module = types.ModuleType(name)

    def install(api: Any, _config: dict[str, Any]) -> None:
        api.register_provider(
            "fake-llm-compaction",
            ProviderConfig(
                stream_fn=provider,
                model=Model(
                    id="fake-llm-compaction",
                    provider="fake",
                    context_window=1000,
                    max_output_tokens=1000,
                ),
                name="fake-llm-compaction",
            ),
        )

    module.install = install  # type: ignore[attr-defined]
    sys.modules[name] = module
    return name


@pytest.mark.asyncio
async def test_llm_compaction_replaces_a_prefix_with_one_summary_message(
    tmp_path: Path,
) -> None:
    summary_text = """## Goal
Keep working on the compaction port.

## Constraints & Preferences
- Preserve exact file paths.

## Progress
### Done
- [x] Investigated the repo.

### In Progress
- [ ] Wiring the semantic compaction layer.

### Blocked
- (none)

## Key Decisions
- **Use session-tree compaction**: Keep durable history in the tree.

## Next Steps
1. Finish the port.

## Critical Context
- src/agentm/extensions/builtin/llm_compaction.py"""
    provider = _SummaryThenReplyProvider(summary_text, "semantic compaction complete")
    provider_module = _install_provider_module(
        "tests.integration._fake_llm_compaction_provider",
        provider,
    )

    initial_messages: list[AgentMessage] = [
        text_message((f"turn-{index} " + ("x" * 390)), timestamp=float(index))
        for index in range(8)
    ]

    session = await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            extensions=[
                (
                    "agentm.extensions.builtin.micro_compact",
                    {"threshold_pct": 0.99, "keep_last": 20},
                ),
                (
                    "agentm.extensions.builtin.llm_compaction",
                    {"enabled": True, "reserve_tokens": 200, "keep_recent_tokens": 150},
                ),
            ],
            provider=(provider_module, {}),
            initial_messages=initial_messages,
            resource_loader=InMemoryResourceLoader(),
        )
    )

    await session.prompt("finish the port")

    assert provider.calls == 2
    assert provider.seen_systems[0] == SUMMARIZATION_SYSTEM_PROMPT
    assert provider.seen_systems[1] in {"", None}

    llm_messages = provider.seen_messages[1]
    first_assistant = llm_messages[0]
    assert isinstance(first_assistant, AssistantMessage)
    first_block = first_assistant.content[0]
    assert isinstance(first_block, TextContent)
    assert first_block.text == summary_text

    persisted_texts = [
        block.text
        for msg in session.session_manager.get_messages()
        for block in getattr(msg, "content", [])
        if isinstance(block, TextContent)
    ]
    assert persisted_texts[0] == summary_text
    assert "semantic compaction complete" in persisted_texts
    assert all("turn-0" not in text for text in persisted_texts)
    assert any("turn-7" in text for text in persisted_texts)

    branch = session.session_manager.get_active_branch()
    compaction_entries = [entry for entry in branch if entry.type == "compaction"]
    assert len(compaction_entries) == 1
    payload = compaction_entries[0].payload
    assert payload["first_kept_entry_id"] != ""
    assert payload["summary"] == summary_text

    await session.shutdown()
