"""Default runtime provider resolver."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeVar

ProviderT = TypeVar("ProviderT")


class LastRegisteredWins:
    """Resolve providers using the historical dict insertion-order policy."""

    def resolve_provider(self, providers: Mapping[str, ProviderT]) -> str | None:
        if not providers:
            return None
        return next(reversed(list(providers)))


__all__ = ["LastRegisteredWins"]
