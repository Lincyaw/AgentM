from __future__ import annotations

from agentm.ai.types import ProviderDefinition

_REGISTRY: dict[str, ProviderDefinition] = {}
_BUILTINS: dict[str, ProviderDefinition] = {}


def register_api_provider(provider: ProviderDefinition, source_id: str | None = None) -> None:
    del source_id  # mirrored surface; unused for now
    _REGISTRY[provider.id] = provider
    _BUILTINS.setdefault(provider.id, provider)


def get_api_provider(provider_id: str) -> ProviderDefinition | None:
    return _REGISTRY.get(provider_id)


def get_api_providers() -> list[ProviderDefinition]:
    return list(_REGISTRY.values())


def unregister_api_provider(provider_id: str) -> None:
    built_in = _BUILTINS.get(provider_id)
    if built_in is None:
        _REGISTRY.pop(provider_id, None)
    else:
        _REGISTRY[provider_id] = built_in


def clear_api_providers() -> None:
    _REGISTRY.clear()


def reset_api_providers() -> None:
    _REGISTRY.clear()
    _REGISTRY.update(_BUILTINS)
