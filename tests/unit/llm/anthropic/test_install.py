"""Tests for the registry-backed Anthropic provider definition."""

from __future__ import annotations

import pytest

from agentm.ai.providers.anthropic import PROVIDER
from agentm.ai.types import ResolvedAuth
from agentm.llm.anthropic import AnthropicStreamFn


def test_provider_definition_builds_known_model() -> None:
    model = PROVIDER.build_model("claude-opus-4-7")

    assert model.id == "claude-opus-4-7"
    assert model.provider == "anthropic"
    assert model.context_window == 200_000
    assert model.max_output_tokens == 32_768


def test_provider_definition_uses_defaults_for_unknown_model() -> None:
    model = PROVIDER.build_model("claude-future-9000")

    assert model.context_window == 200_000
    assert model.max_output_tokens == 8_192


def test_provider_definition_requires_auth() -> None:
    model = PROVIDER.build_model("claude-sonnet-4-6")

    with pytest.raises(RuntimeError, match="No API key found for anthropic"):
        PROVIDER.stream_factory(model, {}, None)


def test_provider_definition_builds_stream_fn_from_resolved_auth() -> None:
    model = PROVIDER.build_model("claude-sonnet-4-6")
    stream_fn = PROVIDER.stream_factory(
        model,
        {"base_url": "https://example.invalid"},
        ResolvedAuth(api_key="sk-test", source="stored", label="stored OAuth token"),
    )

    assert isinstance(stream_fn, AnthropicStreamFn)
    assert stream_fn.api_key == "sk-test"
    assert stream_fn.base_url == "https://example.invalid"
