"""Provider selection port."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from .stream import Model, StreamFn


@dataclass(frozen=True, slots=True)
class ProviderConfig:
    """Session-local LLM provider registration.

    Provider atoms install through the normal ``ExtensionManifest`` path, then
    register one of these records with the session. Host selection policy lives
    separately in ``ProviderResolver``.
    """

    stream_fn: StreamFn
    model: Model
    name: str


ProviderRegistry = Mapping[str, ProviderConfig]


class ProviderResolver(Protocol):
    """Tree-scoped host policy for selecting the active provider."""

    def resolve_provider(self, providers: ProviderRegistry) -> str | None: ...


__all__ = ["ProviderConfig", "ProviderRegistry", "ProviderResolver"]
