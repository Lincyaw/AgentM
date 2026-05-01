from __future__ import annotations

from dataclasses import dataclass

import agentm.ai.providers  # noqa: F401 - registers built-ins on import
from agentm.ai.api_registry import get_api_provider, get_api_providers
from agentm.ai.types import ProviderDefinition
from agentm.core.kernel.stream import Model


@dataclass(frozen=True, slots=True)
class RegisteredModel:
    provider: ProviderDefinition
    model: Model


class ModelRegistry:
    def get_provider(self, provider_id: str) -> ProviderDefinition | None:
        return get_api_provider(provider_id)

    def get_providers(self) -> list[ProviderDefinition]:
        return get_api_providers()

    def get_available(self) -> list[RegisteredModel]:
        return [
            RegisteredModel(provider=provider, model=provider.build_model())
            for provider in self.get_providers()
        ]

    def get_model(self, provider_id: str, model_id: str | None = None) -> Model:
        provider = self.get_provider(provider_id)
        if provider is None:
            raise KeyError(f"Unknown provider: {provider_id}")
        return provider.build_model(model_id)


def get_default_model(provider_id: str) -> str:
    registry = ModelRegistry()
    provider = registry.get_provider(provider_id)
    if provider is None:
        raise KeyError(f"Unknown provider: {provider_id}")
    return provider.default_model
