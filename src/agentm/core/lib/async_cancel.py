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


async def await_known_outcome(awaitable: Awaitable[T]) -> T:
    """Settle a non-cancellable mutation before propagating task cancellation.

    Cancelling an awaiter for ``asyncio.to_thread`` does not stop the backend
    operation. Durable state transitions must therefore reach success or raise
    their backend error before the caller can observe cancellation.
    """

    result, cancelled = await settle_known_outcome(awaitable)
    if cancelled:
        raise asyncio.CancelledError
    return result


async def settle_known_outcome(awaitable: Awaitable[T]) -> tuple[T, bool]:
    """Return an operation result together with observed caller cancellation.

    Ownership-producing operations need the result even when their caller was
    cancelled so they can explicitly release the resource before propagating
    cancellation.
    """

    task = asyncio.ensure_future(awaitable)
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    try:
        result = task.result()
    except asyncio.CancelledError:
        raise
    except BaseException as operation_error:
        if cancelled:
            raise BaseExceptionGroup(
                "operation failed after caller cancellation",
                (asyncio.CancelledError(), operation_error),
            ) from operation_error
        raise
    return result, cancelled


__all__ = [
    "OperationCancelledBySignal",
    "await_known_outcome",
    "await_with_cancel_signal",
    "settle_known_outcome",
]
