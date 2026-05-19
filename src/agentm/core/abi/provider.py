"""Provider selection port."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

from .stream import Model, StreamFn

ProviderT = TypeVar("ProviderT", contravariant=True)


@dataclass(frozen=True, slots=True)
class ProviderConfig:
    """LLM provider registration record shared across provider and runtime layers."""

    stream_fn: StreamFn
    model: Model
    name: str


@dataclass(frozen=True, slots=True)
class ProviderManifest:
    """Provider extension metadata that stays inside the core ABI boundary."""

    name: str
    description: str
    registers: tuple[str, ...]
    config_schema: dict[str, Any] | None = None


class ProviderResolver(Protocol[ProviderT]):
    """Select the active provider registration from a provider registry."""

    def resolve_provider(self, providers: Mapping[str, ProviderT]) -> str | None: ...


__all__ = ["ProviderConfig", "ProviderManifest", "ProviderResolver"]
