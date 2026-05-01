from __future__ import annotations

import pytest

from agentm.harness.auth_storage import AuthStorage
from agentm.harness.model_registry import ModelRegistry
from agentm.harness.model_resolver import ModelResolver


def test_model_registry_includes_pi_mono_provider_catalog() -> None:
    registry = ModelRegistry()
    provider_ids = {provider.id for provider in registry.get_providers()}

    assert {
        "amazon-bedrock",
        "anthropic",
        "openai",
        "azure-openai-responses",
        "openai-codex",
        "deepseek",
        "google",
        "google-gemini-cli",
        "google-antigravity",
        "google-vertex",
        "github-copilot",
        "xai",
        "groq",
        "cerebras",
        "openrouter",
        "vercel-ai-gateway",
        "zai",
        "mistral",
        "minimax",
        "minimax-cn",
        "huggingface",
        "fireworks",
        "opencode",
        "opencode-go",
        "kimi-coding",
        "cloudflare-workers-ai",
    } <= provider_ids


@pytest.mark.asyncio
async def test_model_resolver_builds_registry_backed_provider_config() -> None:
    auth = AuthStorage.in_memory()
    auth.store_api_key("anthropic", "sk-test")
    resolver = ModelResolver(auth_storage=auth)

    resolved = await resolver.resolve("anthropic", model_id="claude-opus-4-7")

    assert resolved.config.name == "anthropic"
    assert resolved.model.id == "claude-opus-4-7"
    assert resolved.auth is not None
    assert resolved.auth.api_key == "sk-test"


@pytest.mark.asyncio
async def test_model_resolver_rejects_missing_auth() -> None:
    resolver = ModelResolver(auth_storage=AuthStorage.in_memory())

    with pytest.raises(RuntimeError, match="No API key found for openai"):
        await resolver.resolve("openai")
