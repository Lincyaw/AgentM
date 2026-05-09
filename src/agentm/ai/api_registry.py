"""Runtime API provider registry."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Iterable

from agentm.ai.types import DEFAULT_PROVIDER_REGISTRY, Api, ApiProvider, Model


@dataclass(slots=True)
class _RegisteredProvider:
    provider: ApiProvider
    source_id: str | None = None


@dataclass(slots=True)
class _ValidatingProvider:
    inner: ApiProvider
    api: Api

    def _check_api(self, model: Model) -> None:
        expected = DEFAULT_PROVIDER_REGISTRY.canonical_api(self.api)
        actual = DEFAULT_PROVIDER_REGISTRY.canonical_api(model.api)
        if actual != expected:
            raise ValueError(f"Mismatched api: {model.api} expected {self.api}")

    async def stream(
        self, model: Model, context: Any, options: Any | None = None
    ) -> Iterable[Any]:
        self._check_api(model)
        return await _await(self.inner.stream(model, context, options))

    async def stream_simple(
        self, model: Model, context: Any, options: Any | None = None
    ) -> Iterable[Any]:
        self._check_api(model)
        return await _await(self.inner.stream_simple(model, context, options))


_registry: dict[Api, _RegisteredProvider] = {}


async def _await(value: Any) -> Any:
    """Await ``value`` if it is awaitable; otherwise return it as-is."""

    if inspect.isawaitable(value):
        return await value
    return value


def _canonical_api(api: Api) -> Api:
    return DEFAULT_PROVIDER_REGISTRY.canonical_api(api)


def register_api_provider(provider: ApiProvider, source_id: str | None = None) -> None:
    """Register ``provider`` under its canonical ``api`` key."""

    expected_api = _canonical_api(provider.api)
    _registry[expected_api] = _RegisteredProvider(
        provider=_ValidatingProvider(inner=provider, api=provider.api),
        source_id=source_id,
    )


def get_api_provider(api: Api) -> ApiProvider | None:
    """Look up a registered provider by api string, or ``None`` if unknown."""

    entry = _registry.get(_canonical_api(api))
    return entry.provider if entry is not None else None


def get_api_providers() -> list[ApiProvider]:
    """Return all currently registered providers (snapshot)."""

    return [entry.provider for entry in _registry.values()]


def unregister_api_providers(source_id: str) -> None:
    """Remove every provider that was registered with ``source_id``."""

    stale = [api for api, entry in _registry.items() if entry.source_id == source_id]
    for api in stale:
        del _registry[api]


def clear_api_providers() -> None:
    """Drop the entire registry. Intended for tests and process shutdown."""

    _registry.clear()
