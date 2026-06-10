"""Generic mutable single-slot container."""

from __future__ import annotations

from typing import Generic, TypeVar

T = TypeVar("T")


class Ref(Generic[T]):
    """Typed mutable reference cell.

    Replaces the ``dict[str, X] = {"value": None}`` pattern used in
    ``session_factory`` to share a mutable slot across closures.
    """

    __slots__ = ("value",)

    def __init__(self, value: T) -> None:
        self.value = value


__all__ = ["Ref"]
