"""Substrate-owned tool execution boundaries.

The agent loop must never call ``tool.execute`` inline. This module centralizes
the execution-domain policy so kernel foreground calls and wrapper atoms such
as ``background_exec`` share the same task/thread/process boundary semantics.
"""

from __future__ import annotations

import asyncio
import multiprocessing
import threading
import traceback
from typing import Any, cast

from loguru import logger

from .tool import (
    TOOL_EXECUTION_DOMAIN_EVENT_LOOP,
    TOOL_EXECUTION_DOMAIN_METADATA_KEY,
    TOOL_EXECUTION_DOMAIN_PROCESS,
    TOOL_EXECUTION_DOMAIN_SANDBOX,
    TOOL_EXECUTION_DOMAIN_THREAD,
    Tool,
    ToolExecutionDomain,
    ToolOutcome,
    ToolResult,
)

_SUPPORTED_DOMAINS = {
    TOOL_EXECUTION_DOMAIN_EVENT_LOOP,
    TOOL_EXECUTION_DOMAIN_THREAD,
    TOOL_EXECUTION_DOMAIN_PROCESS,
    TOOL_EXECUTION_DOMAIN_SANDBOX,
}
_PROCESS_TERMINATE_GRACE_SECONDS = 0.25
_PROCESS_KILL_GRACE_SECONDS = 1.0
_PROCESS_RESULT_JOIN_SECONDS = 1.0


class ToolExecutionDomainUnavailable(RuntimeError):
    """Raised when a tool requests an execution domain the substrate lacks."""


class ToolProcessFailed(RuntimeError):
    """Raised when a process-domain tool exits without a usable result."""


class ToolProcessTerminated(RuntimeError):
    """Raised when the parent killed a process-domain tool."""


class _ThreadForwardedSignal:
    """asyncio.Event-like signal bridge usable inside a worker thread loop."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def is_set(self) -> bool:
        return self._event.is_set()

    def set(self) -> None:
        self._event.set()

    async def wait(self) -> None:
        while not self._event.is_set():
            await asyncio.sleep(0.05)


class _ProcessForwardedSignal:
    """asyncio.Event-like signal bridge backed by a multiprocessing event."""

    def __init__(self, event: Any) -> None:
        self._event = event

    def is_set(self) -> bool:
        return bool(self._event.is_set())

    def set(self) -> None:
        self._event.set()

    async def wait(self) -> None:
        while not self.is_set():
            await asyncio.sleep(0.05)


def tool_execution_domain(tool: Tool) -> ToolExecutionDomain:
    """Return the declared execution domain for ``tool``.

    Missing or malformed metadata preserves the historical event-loop behavior.
    Unsupported values are downgraded with a warning so third-party atoms cannot
    crash catalog construction just by carrying stale metadata.
    """

    metadata = getattr(tool, "metadata", {})
    if not isinstance(metadata, dict):
        return TOOL_EXECUTION_DOMAIN_EVENT_LOOP
    raw = metadata.get(
        TOOL_EXECUTION_DOMAIN_METADATA_KEY,
        TOOL_EXECUTION_DOMAIN_EVENT_LOOP,
    )
    if raw in _SUPPORTED_DOMAINS:
        return cast(ToolExecutionDomain, raw)
    logger.warning(
        "tool {} declares unsupported {}={!r}; using {}",
        tool.name,
        TOOL_EXECUTION_DOMAIN_METADATA_KEY,
        raw,
        TOOL_EXECUTION_DOMAIN_EVENT_LOOP,
    )
    return TOOL_EXECUTION_DOMAIN_EVENT_LOOP


async def execute_tool_call(
    tool: Tool,
    args: dict[str, Any],
    *,
    signal: asyncio.Event | None,
) -> ToolResult | ToolOutcome:
    """Execute one tool call through its declared substrate boundary."""

    domain = tool_execution_domain(tool)
    if domain == TOOL_EXECUTION_DOMAIN_EVENT_LOOP:
        return await _execute_in_event_loop(tool, args, signal=signal)
    if domain == TOOL_EXECUTION_DOMAIN_THREAD:
        return await _execute_in_thread(tool, args, signal=signal)
    if domain == TOOL_EXECUTION_DOMAIN_PROCESS:
        return await _execute_in_process(tool, args, signal=signal)
    if domain == TOOL_EXECUTION_DOMAIN_SANDBOX:
        raise ToolExecutionDomainUnavailable(
            f"tool {tool.name!r} requested execution_domain={domain!r}, "
            "but that execution domain is not implemented yet"
        )
    raise AssertionError(f"unhandled tool execution domain: {domain!r}")


async def _execute_in_event_loop(
    tool: Tool,
    args: dict[str, Any],
    *,
    signal: asyncio.Event | None,
) -> ToolResult | ToolOutcome:
    """Run one tool behind an explicit asyncio task boundary."""

    task = asyncio.create_task(
        tool.execute(args, signal=signal),
        name=f"agentm-tool-{tool.name}",
    )
    try:
        return await task
    except asyncio.CancelledError:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        raise


async def _execute_in_thread(
    tool: Tool,
    args: dict[str, Any],
    *,
    signal: asyncio.Event | None,
) -> ToolResult | ToolOutcome:
    """Run one tool coroutine inside a worker-thread event loop.

    Cancelling the outer await cannot kill a Python thread. We forward the
    signal cooperatively and return control to the session loop immediately;
    process is the killable isolation layer; sandbox remains the future
    resource-isolated variant.
    """

    thread_signal = _ThreadForwardedSignal()
    forwarder: asyncio.Task[None] | None = None
    if signal is not None:
        forwarder = asyncio.create_task(
            _forward_signal(signal, thread_signal),
            name=f"agentm-tool-signal-forwarder-{tool.name}",
        )
    try:
        return await asyncio.to_thread(
            _run_tool_in_thread,
            tool,
            args,
            thread_signal,
        )
    except asyncio.CancelledError:
        thread_signal.set()
        raise
    finally:
        if forwarder is not None:
            forwarder.cancel()
            await asyncio.gather(forwarder, return_exceptions=True)


async def _forward_signal(
    source: asyncio.Event,
    target: _ThreadForwardedSignal,
) -> None:
    await source.wait()
    target.set()


def _run_tool_in_thread(
    tool: Tool,
    args: dict[str, Any],
    signal: _ThreadForwardedSignal,
) -> ToolResult | ToolOutcome:
    return asyncio.run(tool.execute(args, signal=cast(Any, signal)))


async def _execute_in_process(
    tool: Tool,
    args: dict[str, Any],
    *,
    signal: asyncio.Event | None,
) -> ToolResult | ToolOutcome:
    """Run one tool coroutine in a killable child process.

    The implementation is spawn-backed. Process-domain tools must therefore be
    pickleable/importable and pure with respect to the parent runtime: child
    writes to the event bus or session state do not
    mutate the parent process.
    """

    ctx = _process_context()
    child_signal = ctx.Event()
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    process = ctx.Process(
        target=_run_tool_in_process,
        args=(tool, args, child_signal, child_conn),
        name=f"agentm-tool-process-{tool.name}",
    )
    try:
        await asyncio.to_thread(process.start)
    except Exception:
        parent_conn.close()
        child_conn.close()
        raise
    child_conn.close()

    recv_task = asyncio.create_task(
        asyncio.to_thread(parent_conn.recv),
        name=f"agentm-tool-process-recv-{tool.name}",
    )
    signal_task: asyncio.Task[bool] | None = None
    if signal is not None:
        signal_task = asyncio.create_task(
            signal.wait(),
            name=f"agentm-tool-process-signal-{tool.name}",
        )

    try:
        waiters: set[asyncio.Task[Any]] = {recv_task}
        if signal_task is not None:
            waiters.add(signal_task)
        done, _pending = await asyncio.wait(
            waiters,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if recv_task in done:
            payload = await _receive_process_payload(
                recv_task,
                process=process,
                tool_name=tool.name,
            )
            return _decode_process_payload(payload, tool_name=tool.name)

        child_signal.set()
        await _terminate_process(process, tool_name=tool.name, reason="signal")
        raise ToolProcessTerminated(
            f"tool {tool.name!r} process was terminated after signal"
        )
    except asyncio.CancelledError:
        child_signal.set()
        await _terminate_process(process, tool_name=tool.name, reason="cancelled")
        raise
    finally:
        if signal_task is not None:
            signal_task.cancel()
            await asyncio.gather(signal_task, return_exceptions=True)
        if not recv_task.done():
            recv_task.cancel()
        await asyncio.gather(recv_task, return_exceptions=True)
        parent_conn.close()


def _process_context() -> Any:
    if "spawn" not in multiprocessing.get_all_start_methods():
        raise ToolExecutionDomainUnavailable(
            "execution_domain='process' requires the multiprocessing spawn "
            "start method"
        )
    return multiprocessing.get_context("spawn")


def _run_tool_in_process(
    tool: Tool,
    args: dict[str, Any],
    signal_event: Any,
    conn: Any,
) -> None:
    signal = _ProcessForwardedSignal(signal_event)
    try:
        outcome = asyncio.run(tool.execute(args, signal=cast(Any, signal)))
    except BaseException as exc:  # noqa: BLE001 - surfaced to the parent.
        conn.send(
            (
                "error",
                type(exc).__module__,
                type(exc).__name__,
                str(exc),
                traceback.format_exc(),
            )
        )
    else:
        conn.send(("ok", outcome))
    finally:
        conn.close()


async def _receive_process_payload(
    recv_task: asyncio.Task[Any],
    *,
    process: multiprocessing.Process,
    tool_name: str,
) -> Any:
    try:
        payload = await recv_task
    except EOFError as exc:
        await _join_process_after_result(process, tool_name=tool_name)
        raise ToolProcessFailed(
            f"tool {tool_name!r} process exited without a result "
            f"(exitcode={process.exitcode})"
        ) from exc
    except OSError as exc:
        await _join_process_after_result(process, tool_name=tool_name)
        raise ToolProcessFailed(
            f"tool {tool_name!r} process result pipe failed: {exc}"
        ) from exc
    await _join_process_after_result(process, tool_name=tool_name)
    return payload


def _decode_process_payload(payload: Any, *, tool_name: str) -> ToolResult | ToolOutcome:
    if (
        isinstance(payload, tuple)
        and len(payload) == 2
        and payload[0] == "ok"
    ):
        outcome = payload[1]
        if isinstance(outcome, (ToolResult, ToolOutcome)):
            return outcome
        raise ToolProcessFailed(
            f"tool {tool_name!r} process returned invalid payload "
            f"{type(outcome).__name__}"
        )
    if (
        isinstance(payload, tuple)
        and len(payload) == 5
        and payload[0] == "error"
    ):
        _tag, module, name, message, tb = payload
        raise ToolProcessFailed(
            f"tool {tool_name!r} process raised {module}.{name}: "
            f"{message}\n{tb}"
        )
    raise ToolProcessFailed(
        f"tool {tool_name!r} process returned malformed payload: {payload!r}"
    )


async def _join_process_after_result(
    process: multiprocessing.Process,
    *,
    tool_name: str,
) -> None:
    await asyncio.to_thread(process.join, _PROCESS_RESULT_JOIN_SECONDS)
    if not process.is_alive():
        return
    logger.warning(
        "tool {} process sent a result but did not exit; terminating",
        tool_name,
    )
    await _terminate_process(process, tool_name=tool_name, reason="post-result")


async def _terminate_process(
    process: multiprocessing.Process,
    *,
    tool_name: str,
    reason: str,
) -> None:
    if process.is_alive():
        logger.debug(
            "terminating tool {} process pid={} ({})",
            tool_name,
            process.pid,
            reason,
        )
        process.terminate()
        await asyncio.to_thread(process.join, _PROCESS_TERMINATE_GRACE_SECONDS)
    if process.is_alive() and hasattr(process, "kill"):
        logger.debug(
            "killing tool {} process pid={} ({})",
            tool_name,
            process.pid,
            reason,
        )
        process.kill()
        await asyncio.to_thread(process.join, _PROCESS_KILL_GRACE_SECONDS)
    if process.is_alive():
        logger.warning(
            "tool {} process pid={} survived terminate/kill ({})",
            tool_name,
            process.pid,
            reason,
        )


__all__ = [
    "ToolExecutionDomainUnavailable",
    "ToolProcessFailed",
    "ToolProcessTerminated",
    "execute_tool_call",
    "tool_execution_domain",
]
