"""Type definitions and provider metadata for the AI provider boundary."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Iterable, Protocol

Api = str
Provider = str
ThinkingLevel = str


@dataclass(slots=True)
class Model:
    """Minimal model descriptor used to route calls through the registry."""

    id: str
    provider: Provider
    api: Api
    options: dict[str, Any] = field(default_factory=dict)


StreamFunction = Callable[[Model, Any], Awaitable[Iterable[Any]]]


class ApiProvider(Protocol):
    """Protocol every concrete provider implementation must satisfy."""

    api: Api

    def stream(self, model: Model, context: Any, options: Any | None = None) -> Awaitable[Iterable[Any]]:
        ...

    def stream_simple(
        self, model: Model, context: Any, options: Any | None = None
    ) -> Awaitable[Iterable[Any]]:
        ...


ProviderFactory = Callable[[], ApiProvider]


@dataclass(frozen=True, slots=True)
class ProviderDescriptor:
    """Single source of truth for a provider's identity and ambient credentials."""

    id: str
    api: Api
    env_var_precedence: tuple[str, ...]
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _ProviderRecord:
    descriptor: ProviderDescriptor
    factory: ProviderFactory


class ProviderRegistry:
    """Registry of provider descriptors keyed by canonical id and aliases."""

    def __init__(self) -> None:
        self._records: dict[str, _ProviderRecord] = {}
        self._aliases: dict[str, str] = {}
        self._api_aliases: dict[str, str] = {}

    def register(
        self, descriptor: ProviderDescriptor, factory: ProviderFactory
    ) -> None:
        key = _normalize_key(descriptor.id)
        record = _ProviderRecord(descriptor=descriptor, factory=factory)
        self._records[key] = record
        self._api_aliases[_normalize_key(descriptor.api)] = descriptor.api
        self._api_aliases[key] = descriptor.api
        for alias in descriptor.aliases:
            self._aliases[_normalize_key(alias)] = key
            self._api_aliases[_normalize_key(alias)] = descriptor.api

    def resolve(self, id_or_alias: str) -> ProviderDescriptor:
        key = _normalize_key(id_or_alias)
        record = self._records.get(key)
        if record is None:
            record = self._records.get(self._aliases.get(key, ""))
        if record is None:
            raise KeyError(id_or_alias)
        return record.descriptor

    def descriptors(self) -> tuple[ProviderDescriptor, ...]:
        return tuple(record.descriptor for record in self._records.values())

    def factory_for(self, id_or_alias: str) -> ProviderFactory:
        descriptor = self.resolve(id_or_alias)
        return self._records[_normalize_key(descriptor.id)].factory

    def find_env_keys(
        self, provider: str, env: Mapping[str, str] | None = None
    ) -> list[str] | None:
        try:
            descriptor = self.resolve(provider)
        except KeyError:
            return None
        source = os.environ if env is None else env
        found = [name for name in descriptor.env_var_precedence if source.get(name)]
        return found or None

    def get_env_api_key(
        self, provider: str, env: Mapping[str, str] | None = None
    ) -> str | None:
        found = self.find_env_keys(provider, env)
        if not found:
            return None
        source = os.environ if env is None else env
        return source.get(found[0])

    def canonical_api(self, api_or_alias: str) -> str:
        return self._api_aliases.get(_normalize_key(api_or_alias), _normalize_key(api_or_alias))


def _normalize_key(value: str) -> str:
    return value.strip().lower()


def _unavailable_provider_factory(provider_id: str) -> ProviderFactory:
    def _factory() -> ApiProvider:
        raise NotImplementedError(f"Provider {provider_id!r} has no built-in implementation")

    return _factory


def _build_default_provider_registry() -> ProviderRegistry:
    registry = ProviderRegistry()
    for descriptor in DEFAULT_PROVIDER_DESCRIPTORS:
        registry.register(descriptor, _unavailable_provider_factory(descriptor.id))
    return registry


DEFAULT_PROVIDER_DESCRIPTORS: tuple[ProviderDescriptor, ...] = (
    ProviderDescriptor(
        id="anthropic",
        api="anthropic-messages",
        env_var_precedence=("ANTHROPIC_OAUTH_TOKEN", "ANTHROPIC_API_KEY"),
    ),
    ProviderDescriptor(
        id="openai",
        api="openai-responses",
        env_var_precedence=("OPENAI_API_KEY",),
        aliases=("openai-responses",),
    ),
    ProviderDescriptor(
        id="amazon-bedrock",
        api="bedrock-converse-stream",
        env_var_precedence=("AWS_BEARER_TOKEN_BEDROCK", "AWS_ACCESS_KEY_ID"),
        aliases=("bedrock",),
    ),
    ProviderDescriptor(
        id="google",
        api="google-generative-ai",
        env_var_precedence=("GEMINI_API_KEY",),
        aliases=("gemini",),
    ),
    ProviderDescriptor(
        id="google-vertex",
        api="google-vertex-ai",
        env_var_precedence=("GOOGLE_CLOUD_API_KEY",),
        aliases=("vertex",),
    ),
    ProviderDescriptor(
        id="azure-openai-responses",
        api="openai-responses",
        env_var_precedence=("AZURE_OPENAI_API_KEY",),
        aliases=("azure-openai",),
    ),
    ProviderDescriptor(
        id="openai-codex",
        api="openai-responses",
        env_var_precedence=("OPENAI_API_KEY",),
        aliases=("codex",),
    ),
    ProviderDescriptor(id="deepseek", api="openai-chat", env_var_precedence=("DEEPSEEK_API_KEY",)),
    ProviderDescriptor(
        id="github-copilot",
        api="openai-chat",
        env_var_precedence=("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"),
        aliases=("copilot",),
    ),
    ProviderDescriptor(id="xai", api="openai-chat", env_var_precedence=("XAI_API_KEY",)),
    ProviderDescriptor(id="groq", api="openai-chat", env_var_precedence=("GROQ_API_KEY",)),
    ProviderDescriptor(id="cerebras", api="openai-chat", env_var_precedence=("CEREBRAS_API_KEY",)),
    ProviderDescriptor(id="openrouter", api="openai-chat", env_var_precedence=("OPENROUTER_API_KEY",)),
    ProviderDescriptor(id="mistral", api="openai-chat", env_var_precedence=("MISTRAL_API_KEY",)),
    ProviderDescriptor(id="huggingface", api="openai-chat", env_var_precedence=("HF_TOKEN",)),
    ProviderDescriptor(id="fireworks", api="openai-chat", env_var_precedence=("FIREWORKS_API_KEY",)),
)

DEFAULT_PROVIDER_REGISTRY = _build_default_provider_registry()
KNOWN_PROVIDERS = tuple(descriptor.id for descriptor in DEFAULT_PROVIDER_DESCRIPTORS)
KNOWN_APIS = tuple(dict.fromkeys(descriptor.api for descriptor in DEFAULT_PROVIDER_DESCRIPTORS))
