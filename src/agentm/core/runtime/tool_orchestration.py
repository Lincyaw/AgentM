"""Default batch tool orchestrator."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterable

from agentm.core.abi.cancel import (
    CancelSignal,
    EventCancelSource,
    cancel_reason,
)
from agentm.core.abi.tool_orchestration import (
    ToolOrchestrationRequest,
    ToolOrchestrationResult,
    ToolOrchestrator,
    ToolWorkItem,
)
from agentm.core.abi.tool_executor import ToolExecutor
from agentm.core.lib.tool_executor import execute_tool_call


class _CombinedCancelSignal:
    def __init__(self, *signals: CancelSignal | None) -> None:
        self._signals = tuple(signal for signal in signals if signal is not None)

    @property
    def reason(self) -> str | None:
        for signal in self._signals:
            if signal.is_set():
                return cancel_reason(signal) or "unknown"
        return None

    def is_set(self) -> bool:
        return any(signal.is_set() for signal in self._signals)

    async def wait(self) -> object:
        if self.is_set():
            return None
        waiters: list[asyncio.Task[object]] = [
            asyncio.create_task(signal.wait())
            for signal in self._signals
        ]
        try:
            await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for waiter in waiters:
                if not waiter.done():
                    waiter.cancel()
            await asyncio.gather(*waiters, return_exceptions=True)
        return None


def _partition(items: Iterable[ToolWorkItem]) -> list[tuple[ToolWorkItem, ...]]:
    batches: list[tuple[ToolWorkItem, ...]] = []
    parallel: list[ToolWorkItem] = []
    for item in items:
        if item.requirements.concurrency == "parallel_safe":
            parallel.append(item)
            continue
        if parallel:
            batches.append(tuple(parallel))
            parallel = []
        batches.append((item,))
    if parallel:
        batches.append(tuple(parallel))
    return batches


async def _run_item(
    item: ToolWorkItem,
    *,
    signal: CancelSignal | None,
    executor: ToolExecutor | None,
) -> ToolOrchestrationResult:
    try:
        output = await execute_tool_call(
            item.tool,
            item.args,
            signal=signal,
            executor=executor,
            requirements=item.requirements,
        )
        return ToolOrchestrationResult(
            item=item,
            status="completed",
            output=output,
        )
    except asyncio.CancelledError as exc:
        current = asyncio.current_task()
        if current is not None and current.cancelling():
            raise
        return ToolOrchestrationResult(
            item=item,
            status="cancelled",
            cancel_reason=cancel_reason(signal) or str(exc) or "tool_cancelled",
        )
    except Exception as exc:
        return ToolOrchestrationResult(
            item=item,
            status="failed",
            error=exc,
        )


class DefaultToolOrchestrator:
    """Default scheduler matching the SDK's conservative execution semantics."""

    async def stream_batch(
        self,
        request: ToolOrchestrationRequest,
        *,
        signal: CancelSignal | None = None,
        executor: ToolExecutor | None = None,
    ) -> AsyncIterator[ToolOrchestrationResult]:
        for batch in _partition(request.items):
            if len(batch) == 1:
                yield await _run_item(
                    batch[0],
                    signal=signal,
                    executor=executor,
                )
                continue
            async for result in self._stream_parallel(
                batch,
                signal=signal,
                executor=executor,
            ):
                yield result

    async def _stream_parallel(
        self,
        batch: tuple[ToolWorkItem, ...],
        *,
        signal: CancelSignal | None,
        executor: ToolExecutor | None,
    ) -> AsyncIterator[ToolOrchestrationResult]:
        sibling = EventCancelSource()
        child_signal = _CombinedCancelSignal(signal, sibling)
        tasks: dict[asyncio.Task[ToolOrchestrationResult], ToolWorkItem] = {
            asyncio.create_task(
                _run_item(item, signal=child_signal, executor=executor),
                name=f"agentm-tool-orchestrate-{item.call.name}",
            ): item
            for item in batch
        }
        pending = set(tasks)
        try:
            while pending:
                done, pending = await asyncio.wait(
                    pending,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in done:
                    result = task.result()
                    yield result
                    if result.status == "failed" and not sibling.is_set():
                        sibling.set("sibling_error")
        finally:
            unfinished = [task for task in tasks if not task.done()]
            for task in unfinished:
                task.cancel()
            if unfinished:
                await asyncio.gather(*unfinished, return_exceptions=True)


_DEFAULT_ORCHESTRATOR = DefaultToolOrchestrator()


def default_tool_orchestrator() -> ToolOrchestrator:
    return _DEFAULT_ORCHESTRATOR


__all__ = [
    "DefaultToolOrchestrator",
    "default_tool_orchestrator",
]
