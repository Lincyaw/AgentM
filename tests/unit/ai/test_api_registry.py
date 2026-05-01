"""Tests for the api-registry — port behavior of pi-mono `api-registry.ts`."""

from __future__ import annotations

from typing import Any

import pytest

from agentm.ai import (
    Model,
    clear_api_providers,
    get_api_provider,
    get_api_providers,
    register_api_provider,
    unregister_api_providers,
)


class _FakeProvider:
    def __init__(self, api: str) -> None:
        self.api = api
        self.stream_calls: list[Model] = []

    async def stream(self, model: Model, context: Any, options: Any | None = None) -> list[str]:
        self.stream_calls.append(model)
        return ["streamed", model.id]

    async def stream_simple(
        self, model: Model, context: Any, options: Any | None = None
    ) -> list[str]:
        return ["simple", model.id]


@pytest.fixture(autouse=True)
def _clean_registry() -> None:
    clear_api_providers()
    yield
    clear_api_providers()


def test_register_and_lookup_round_trips() -> None:
    provider = _FakeProvider("anthropic-messages")
    register_api_provider(provider)

    found = get_api_provider("anthropic-messages")
    assert found is not None
    assert found.api == "anthropic-messages"
    assert get_api_provider("openai-completions") is None


def test_register_replaces_prior_entry_for_same_api() -> None:
    first = _FakeProvider("anthropic-messages")
    second = _FakeProvider("anthropic-messages")
    register_api_provider(first)
    register_api_provider(second)

    assert len(get_api_providers()) == 1


def test_unregister_by_source_id_removes_only_matching_entries() -> None:
    register_api_provider(_FakeProvider("anthropic-messages"), source_id="ext-a")
    register_api_provider(_FakeProvider("openai-completions"), source_id="ext-b")

    unregister_api_providers("ext-a")

    assert get_api_provider("anthropic-messages") is None
    assert get_api_provider("openai-completions") is not None


@pytest.mark.asyncio
async def test_stream_rejects_model_with_mismatched_api() -> None:
    register_api_provider(_FakeProvider("anthropic-messages"))
    wrapped = get_api_provider("anthropic-messages")
    assert wrapped is not None

    bad_model = Model(id="m", provider="openai", api="openai-completions")
    with pytest.raises(ValueError, match="Mismatched api"):
        await wrapped.stream(bad_model, context=None)


@pytest.mark.asyncio
async def test_stream_passes_through_when_api_matches() -> None:
    register_api_provider(_FakeProvider("anthropic-messages"))
    wrapped = get_api_provider("anthropic-messages")
    assert wrapped is not None

    model = Model(id="claude-3", provider="anthropic", api="anthropic-messages")
    out = await wrapped.stream(model, context={"hello": "world"})
    assert out == ["streamed", "claude-3"]
