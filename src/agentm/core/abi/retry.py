"""Retry policy port for async provider operations."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Protocol, TypeVar

T = TypeVar("T")


class RetryPolicy(Protocol):
    """Run an awaitable factory with provider-supplied retryability checks."""

    async def run(
        self,
        fn: Callable[[], Awaitable[T]],
        *,
        is_retryable: Callable[[BaseException], bool],
    ) -> T: ...
