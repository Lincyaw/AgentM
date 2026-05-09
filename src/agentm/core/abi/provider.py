"""Provider selection port."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, TypeVar

ProviderT = TypeVar("ProviderT", contravariant=True)


class ProviderResolver(Protocol[ProviderT]):
    """Select the active provider registration from a provider registry."""

    def resolve_provider(self, providers: Mapping[str, ProviderT]) -> str | None: ...


__all__ = ["ProviderResolver"]
