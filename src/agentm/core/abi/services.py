"""Published service protocols shared by extensions and presenters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class CostBreakdown:
    amount: float
    currency: str = "usd"


class CostQueryService(Protocol):
    def estimate(self, usage: Any, *, provider: str | None = None) -> CostBreakdown: ...


__all__ = ["CostBreakdown", "CostQueryService"]
