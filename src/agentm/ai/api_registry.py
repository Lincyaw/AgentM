"""Provider registry — port of pi-mono `api-registry.ts`.

Mutable module-level map keyed by `Api` string. `register_api_provider`
wraps `stream` / `stream_simple` so each call validates the model's
`api` matches the provider's `api` (matches pi-mono's mismatched-api
error). `source_id` lets a caller bulk-unregister everything one
extension or test fixture installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Iterable

from agentm.ai.types import Api, ApiProvider, Model


@dataclass(slots=True)
class _RegisteredProvider:
    provider: ApiProvider
    source_id: str | None = None


_registry: dict[Api, _RegisteredProvider] = {}


def _wrap(provider: ApiProvider) -> ApiProvider:
    """Return a thin wrapper that enforces `model.api == provider.api`."""

    expected = provider.api

    class _Wrapped:
        api = expected

        async def stream(
            self, model: Model, context: Any, options: Any | None = None
        ) -> Iterable[Any]:
            if model.api != expected:
                raise ValueError(f"Mismatched api: {model.api} expected {expected}")
            result = provider.stream(model, context, options)
            return await _await(result)

        async def stream_simple(
            self, model: Model, context: Any, options: Any | None = None
        ) -> Iterable[Any]:
            if model.api != expected:
                raise ValueError(f"Mismatched api: {model.api} expected {expected}")
            result = provider.stream_simple(model, context, options)
            return await _await(result)

    return _Wrapped()  # type: ignore[return-value]


async def _await(value: Any) -> Any:
    """Await ``value`` if it is awaitable; otherwise return as-is."""

    if isinstance(value, Awaitable):
        return await value
    return value


def register_api_provider(provider: ApiProvider, source_id: str | None = None) -> None:
    """Register ``provider`` under its ``api`` key. Replaces any prior entry."""

    _registry[provider.api] = _RegisteredProvider(provider=_wrap(provider), source_id=source_id)


def get_api_provider(api: Api) -> ApiProvider | None:
    """Look up a registered provider by api string, or ``None`` if unknown."""

    entry = _registry.get(api)
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
