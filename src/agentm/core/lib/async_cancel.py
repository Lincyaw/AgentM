"""Async helpers for composing awaitables with SDK cancellation signals."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import TypeVar

from agentm.core.abi.cancel import CancelSignal

T = TypeVar("T")


class OperationCancelledBySignal(Exception):
    """Raised when a ``CancelSignal`` wins a race against an awaitable."""


async def await_with_cancel_signal(
    awaitable: Awaitable[T],
    signal: CancelSignal | None,
) -> T:
    """Await *awaitable*, cancelling it if *signal* fires first."""

    if signal is None:
        return await awaitable
    if signal.is_set():
        raise OperationCancelledBySignal

    value_task = asyncio.ensure_future(awaitable)
    signal_task = asyncio.create_task(signal.wait())
    try:
        done, _pending = await asyncio.wait(
            (value_task, signal_task),
            return_when=asyncio.FIRST_COMPLETED,
        )
        if value_task in done:
            signal_task.cancel()
            await asyncio.gather(signal_task, return_exceptions=True)
            return await value_task
        value_task.cancel()
        await asyncio.gather(value_task, return_exceptions=True)
        raise OperationCancelledBySignal
    finally:
        if not value_task.done():
            value_task.cancel()
        if not signal_task.done():
            signal_task.cancel()
        await asyncio.gather(value_task, signal_task, return_exceptions=True)


__all__ = [
    "OperationCancelledBySignal",
    "await_with_cancel_signal",
]
