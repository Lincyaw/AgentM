from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import pytest

from agentm.ai.api_registry import (
    clear_api_providers,
    get_api_provider,
    register_api_provider,
)
from agentm.ai.env_api_keys import find_env_keys, get_env_api_key, resolve
from agentm.ai.types import (
    DEFAULT_PROVIDER_REGISTRY,
    KNOWN_APIS,
    KNOWN_PROVIDERS,
    Model,
    ProviderDescriptor,
    ProviderRegistry,
)


@dataclass(slots=True)
class _Provider:
    api: str

    async def stream(
        self, model: Model, context: Any, options: Any | None = None
    ) -> Iterable[Any]:
        return [model.id, context, options]

    async def stream_simple(
        self, model: Model, context: Any, options: Any | None = None
    ) -> Iterable[Any]:
        return [model.id, context, options]


@pytest.fixture(autouse=True)
def _clear_api_registry() -> None:
    clear_api_providers()


def test_env_key_resolution_uses_descriptor_precedence_with_in_memory_env() -> None:
    env = {"ANTHROPIC_API_KEY": "api-key", "ANTHROPIC_OAUTH_TOKEN": "oauth-token"}

    assert find_env_keys("anthropic", env) == [
        "ANTHROPIC_OAUTH_TOKEN",
        "ANTHROPIC_API_KEY",
    ]
    assert get_env_api_key("anthropic", env) == "oauth-token"
    assert resolve("anthropic", env) == "oauth-token"
    assert get_env_api_key("unknown", env) is None


def test_default_provider_metadata_covers_builtin_vendors() -> None:
    providers = set(KNOWN_PROVIDERS)

    assert "amazon-bedrock" in providers
    assert "openai-codex" in providers
    assert DEFAULT_PROVIDER_REGISTRY.resolve("bedrock").id == "amazon-bedrock"
    assert DEFAULT_PROVIDER_REGISTRY.resolve("codex").id == "openai-codex"
    assert (
        tuple(dict.fromkeys(d.api for d in DEFAULT_PROVIDER_REGISTRY.descriptors()))
        == KNOWN_APIS
    )


def test_registering_new_vendor_only_requires_descriptor_and_factory() -> None:
    registry = ProviderRegistry()
    descriptor = ProviderDescriptor(
        id="fake-provider",
        api="fake-api",
        env_var_precedence=("FAKE_API_KEY",),
        aliases=("fake",),
    )
    provider = _Provider(api="fake-api")
    registry.register(descriptor, lambda: provider)

    assert registry.resolve("fake").id == "fake-provider"
    assert (
        registry.get_env_api_key("fake-provider", {"FAKE_API_KEY": "secret"})
        == "secret"
    )
    assert registry.factory_for("fake") is not None


def test_provider_registry_builds_cli_extension_config_from_descriptor_env() -> None:
    registry = ProviderRegistry()
    descriptor = ProviderDescriptor(
        id="third",
        api="openai-chat",
        env_var_precedence=("THIRD_API_KEY",),
        extension_module="agentm.llm.openai",
        default_model="third-default",
        base_url_env="THIRD_BASE_URL",
        verify_ssl_env="THIRD_VERIFY_SSL",
        default_query_ticket_env="THIRD_TICKET",
    )
    registry.register(descriptor, lambda: _Provider(api="openai-chat"))

    assert registry.build(
        "third",
        {},
        env={
            "THIRD_BASE_URL": "https://third.example",
            "THIRD_VERIFY_SSL": "false",
            "THIRD_TICKET": "ticket",
        },
    ) == (
        "agentm.llm.openai",
        {
            "model": "third-default",
            "name": "third",
            "base_url": "https://third.example",
            "default_query": {"warpgate-ticket": "ticket"},
            "verify_ssl": False,
        },
    )


@pytest.mark.asyncio
async def test_validating_provider_uses_canonical_api_for_stream_methods() -> None:
    register_api_provider(_Provider(api="openai"))
    provider = get_api_provider("openai-responses")
    assert provider is not None

    model = Model(id="gpt", provider="openai", api="openai-responses")
    assert await provider.stream(model, "ctx") == ["gpt", "ctx", None]
    assert await provider.stream_simple(model, "ctx", {"temperature": 0}) == [
        "gpt",
        "ctx",
        {"temperature": 0},
    ]

    with pytest.raises(ValueError, match="Mismatched api"):
        await provider.stream(
            Model(id="claude", provider="anthropic", api="anthropic-messages"), "ctx"
        )
