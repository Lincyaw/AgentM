from __future__ import annotations

from agentm.ai.types import ProviderDefinition

_REGISTRY: dict[str, ProviderDefinition] = {}
_BUILTINS: dict[str, ProviderDefinition] = {}
_SOURCES: dict[str, str] = {}


def register_api_provider(provider: ProviderDefinition, source_id: str | None = None) -> None:
    _REGISTRY[provider.id] = provider
    _BUILTINS.setdefault(provider.id, provider)
    if source_id is not None:
        _SOURCES[provider.id] = source_id


def get_api_provider(provider_id: str) -> ProviderDefinition | None:
    return _REGISTRY.get(provider_id)


def get_api_providers() -> list[ProviderDefinition]:
    return list(_REGISTRY.values())


def unregister_api_provider(provider_id: str) -> None:
    built_in = _BUILTINS.get(provider_id)
    if built_in is None:
        _REGISTRY.pop(provider_id, None)
        _SOURCES.pop(provider_id, None)
    else:
        _REGISTRY[provider_id] = built_in
        _SOURCES.pop(provider_id, None)


def unregister_api_providers(source_id: str) -> None:
    for provider_id, registered_source_id in tuple(_SOURCES.items()):
        if registered_source_id == source_id:
            unregister_api_provider(provider_id)


def clear_api_providers() -> None:
    _REGISTRY.clear()
    _SOURCES.clear()


def reset_api_providers() -> None:
    _REGISTRY.clear()
    _REGISTRY.update(_BUILTINS)
    _SOURCES.clear()
