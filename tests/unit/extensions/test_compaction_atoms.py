from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from agentm.core.abi import AssistantMessage, MessageEnd, Model, TextContent
from agentm.extensions.builtin import llm_compaction
from agentm.core.abi.extension import ProviderConfig


@pytest.mark.asyncio
async def test_llm_compaction_provider_summarizer_uses_provider_result_text() -> None:
    seen: dict[str, Any] = {}

    async def _stream_fn(**kwargs: Any) -> AsyncIterator[Any]:
        seen.update(kwargs)
        yield MessageEnd(
            message=AssistantMessage(
                role="assistant",
                content=[TextContent(type="text", text="summary text")],
                timestamp=0.0,
                stop_reason="end_turn",
            )
        )

    provider = ProviderConfig(
        stream_fn=_stream_fn,
        model=Model(
            id="parent",
            provider="fake",
            context_window=4096,
            max_output_tokens=50,
        ),
        name="fake",
    )
    summarizer = llm_compaction._ProviderSummarizer(
        provider,
        Model(id="parent", provider="fake", context_window=4096, max_output_tokens=50),
    )

    result = await summarizer("system prompt", "body", 10)

    assert result == "summary text"
    assert seen["system"] == "system prompt"
    assert seen["tools"] == []
    assert seen["model"].max_output_tokens == 10
