"""Auto-background long-running tool calls at the execution boundary.

The atom installs a wrapping :class:`ToolExecutor`. Fast calls preserve the
inner executor's result. Calls that exceed the configured foreground timeout
continue under an atom-owned cancellation source and return a task ticket;
their final result is delivered through ``BackgroundCompletion``.

The parent turn cancellation signal is forwarded only while a call is in the
foreground. Once a task has been detached, only ``cancel_background`` or
session shutdown can stop it. This is the same ownership split used for
asynchronous child agents.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, field, replace
import json
import time
from typing import Literal
import uuid

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from agentm.core.abi import (
    AtomAPI,
    AtomInstallPriority,
    BackgroundCompletion,
    CancelSignal,
    EventCancelSource,
    FunctionTool,
    JsonValue,
    SessionShutdownEvent,
    TextContent,
    ToolContinue,
    ToolExecutionCapabilities,
    ToolExecutionRequest,
    ToolExecutor,
    ToolOutcome,
    ToolResult,
    ToolTerminate,
)
from agentm.core.lib import BackgroundTask, BackgroundTaskRegistry
from agentm.core.lib.tool_executor import DirectToolExecutor
from agentm.extensions import ExtensionManifest

_RUNNING: Literal["running"] = "running"
_COMPLETED: Literal["completed"] = "completed"
_ERROR: Literal["error"] = "error"
_CANCELLED: Literal["cancelled"] = "cancelled"
_Status = Literal["running", "completed", "error", "cancelled"]

_BYPASS_TOOLS = frozenset(
    {
        "check_background",
        "cancel_background",
        # Child dispatch owns a separate foreground/background cancellation
        # choice. Wrapping it here would detach the tool call without changing
        # the child's parent-cancellation domain.
        "dispatch_agent",
    }
)
_DEFAULT_TIMEOUT_SECONDS = 60.0
_DEFAULT_SHUTDOWN_GRACE_SECONDS = 5.0
_MAX_ACTIVITY_LABEL_CHARS = 96
_COMPLETION_PREVIEW_CHARS = 2000
_CHECK_RESULT_TAIL_LINES = 10


class BackgroundExecConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    timeout: float = Field(
        default=_DEFAULT_TIMEOUT_SECONDS,
        gt=0,
        allow_inf_nan=False,
    )
    denylist: tuple[str, ...] = ()
    shutdown_grace_seconds: float = Field(
        default=_DEFAULT_SHUTDOWN_GRACE_SECONDS,
        gt=0,
        allow_inf_nan=False,
    )


MANIFEST = ExtensionManifest(
    name="background_exec",
    description=(
        "Move long-running tool calls to detached execution and deliver their "
        "result through BackgroundCompletion."
    ),
    registers=(
        "executor:background_exec",
        "tool:check_background",
        "tool:cancel_background",
        "event:session_shutdown",
    ),
    config_schema=BackgroundExecConfig,
    requires=(),
    # Install after ordinary context/tool atoms so this wrapper remains the
    # outer owner while delegated executors keep their locks and sandboxes for
    # the background task's full lifetime.
    priority=AtomInstallPriority.CONTEXT + 100,
)


@dataclass(slots=True, kw_only=True)
class _BackgroundToolTask(BackgroundTask):
    tool_name: str
    label: str
    status: _Status = _RUNNING
    started_at: float = field(default_factory=time.monotonic)
    outcome: ToolResult | ToolOutcome | None = None
    error: str | None = None
    work_bracket: AbstractContextManager[None] | None = None


def _tool_result(
    payload: Mapping[str, JsonValue],
    *,
    is_error: bool = False,
) -> ToolResult:
    rendered = json.dumps(payload, ensure_ascii=True, sort_keys=True)
    return ToolResult(
        content=[TextContent(type="text", text=rendered)],
        is_error=is_error,
        extras=payload,
    )


def _outcome_result(outcome: ToolResult | ToolOutcome) -> ToolResult:
    if isinstance(outcome, ToolResult):
        return outcome
    if isinstance(outcome, (ToolContinue, ToolTerminate)):
        return outcome.result
    raise TypeError(f"unexpected tool outcome: {type(outcome).__name__}")


def _result_text(result: ToolResult) -> str:
    chunks = [
        block.text for block in result.content if isinstance(block, TextContent)
    ]
    return "\n".join(chunks).strip()


def _last_lines(text: str, count: int) -> str:
    lines = text.splitlines(keepends=True)
    return text if len(lines) <= count else "".join(lines[-count:])


def _single_line(value: str) -> str:
    return " ".join(value.split())


def _truncate_label(value: str) -> str:
    if len(value) <= _MAX_ACTIVITY_LABEL_CHARS:
        return value
    return value[: _MAX_ACTIVITY_LABEL_CHARS - 3].rstrip() + "..."


def _activity_label(tool_name: str, args: Mapping[str, object]) -> str:
    if tool_name in {"bash", "shell"}:
        command = args.get("cmd", args.get("command"))
        if isinstance(command, str):
            rendered = _single_line(command)
            if rendered:
                return _truncate_label(f"{tool_name}: {rendered}")
    return tool_name


def _completion_note(state: _BackgroundToolTask) -> str:
    head = f"Background task {state.task_id} ({state.label})"
    if state.status == _COMPLETED and state.outcome is not None:
        body = _result_text(_outcome_result(state.outcome))
        if not body:
            return f"{head} finished."
        if len(body) > _COMPLETION_PREVIEW_CHARS:
            tail = body[-_COMPLETION_PREVIEW_CHARS:]
            return (
                f"{head} finished.\n\nResult (last "
                f"{_COMPLETION_PREVIEW_CHARS} chars of {len(body)}; use "
                f"check_background for the retained result):\n...{tail}"
            )
        return f"{head} finished.\n\nResult:\n{body}"
    if state.status == _ERROR:
        return f"{head} failed: {state.error}"
    return f"{head} was cancelled."


def _task_payload(state: _BackgroundToolTask) -> dict[str, JsonValue]:
    payload: dict[str, JsonValue] = {
        "task_id": state.task_id,
        "tool_name": state.tool_name,
        "label": state.label,
        "status": state.status,
        "elapsed_s": round(time.monotonic() - state.started_at, 1),
    }
    if state.error is not None:
        payload["error"] = state.error
    if state.status == _COMPLETED and state.outcome is not None:
        text = _result_text(_outcome_result(state.outcome))
        payload["result"] = _last_lines(text, _CHECK_RESULT_TAIL_LINES)
    return payload


async def _forward_cancel(
    source: CancelSignal,
    target: EventCancelSource,
) -> None:
    await source.wait()
    target.set(source.reason or "unknown")


async def _stop_forwarder(forwarder: asyncio.Task[None] | None) -> None:
    if forwarder is None:
        return
    if not forwarder.done():
        forwarder.cancel()
    await asyncio.gather(forwarder, return_exceptions=True)


class _BackgroundManager:
    def __init__(
        self,
        *,
        api: AtomAPI,
        timeout: float,
        denylist: frozenset[str],
        shutdown_grace_seconds: float,
    ) -> None:
        self._api = api
        self.timeout = timeout
        self._denylist = denylist
        self._shutdown_grace_seconds = shutdown_grace_seconds
        self._registry: BackgroundTaskRegistry[_BackgroundToolTask] = (
            BackgroundTaskRegistry(max_workers=None)
        )
        self._shutting_down = False

    def should_detach(self, tool_name: str) -> bool:
        return (
            tool_name not in _BYPASS_TOOLS
            and tool_name not in self._denylist
        )

    def prepare_call(
        self,
        tool_name: str,
        args: Mapping[str, object],
    ) -> tuple[dict[str, object], float]:
        clean_args = dict(args)
        immediate = clean_args.pop("background", None) is True
        foreground_timeout = 0.0 if immediate else self.timeout

        if tool_name not in {"bash", "shell"}:
            return clean_args, foreground_timeout
        raw_timeout = clean_args.get("timeout")
        if not isinstance(raw_timeout, (int, float)) or isinstance(raw_timeout, bool):
            return clean_args, foreground_timeout
        if raw_timeout <= 0 or immediate:
            return clean_args, foreground_timeout
        if raw_timeout >= self.timeout:
            return clean_args, self.timeout

        # A shorter bash timeout means "detach by this point". The delegated
        # operation retains the atom's configured timeout as its actual kill
        # deadline after detachment.
        clean_args["timeout"] = self.timeout
        return clean_args, float(raw_timeout)

    async def detach(
        self,
        *,
        tool_name: str,
        args: Mapping[str, object],
        inner_task: asyncio.Task[ToolResult | ToolOutcome],
        abort_source: EventCancelSource,
        note: str,
    ) -> ToolResult:
        if self._shutting_down:
            abort_source.set("shutdown")
            await self._drain_or_cancel(inner_task)
            return _tool_result(
                {"error": "session is shutting down; background task refused"},
                is_error=True,
            )

        task_id = uuid.uuid4().hex
        state = _BackgroundToolTask(
            task_id=task_id,
            tool_name=tool_name,
            label=_activity_label(tool_name, args),
            abort_signal=abort_source,
        )
        try:
            bracket = self._api.track_background()
            bracket.__enter__()
            state.work_bracket = bracket
        except BaseException:
            abort_source.set("unknown")
            await self._drain_or_cancel(inner_task)
            raise

        watcher = asyncio.create_task(
            self._watch(state, inner_task),
            name=f"agentm-background-{tool_name}-{task_id}",
        )
        state.task = watcher
        try:
            await self._registry.register(state)
        except BaseException:
            abort_source.set("unknown")
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)
            self._exit_work_bracket(state)
            raise
        return _tool_result(
            {
                "task_id": task_id,
                "status": _RUNNING,
                "note": note,
            }
        )

    async def _watch(
        self,
        state: _BackgroundToolTask,
        inner_task: asyncio.Task[ToolResult | ToolOutcome],
    ) -> None:
        try:
            try:
                state.outcome = await inner_task
            except asyncio.CancelledError:
                state.status = _CANCELLED
            except Exception as exc:  # noqa: BLE001
                state.status = _ERROR
                state.error = str(exc) or type(exc).__name__
                logger.warning(
                    "background tool {} failed: {}",
                    state.tool_name,
                    state.error,
                )
            else:
                state.status = (
                    _CANCELLED if state.abort_signal.is_set() else _COMPLETED
                )
            if not self._shutting_down:
                self._post_completion(state)
        finally:
            self._exit_work_bracket(state)

    def _post_completion(self, state: _BackgroundToolTask) -> None:
        if state.read:
            return
        state.read = True
        terminal = state.status == _COMPLETED and isinstance(
            state.outcome,
            ToolTerminate,
        )
        self._api.push_trigger(
            BackgroundCompletion(
                task_id=state.task_id,
                payload=_completion_note(state),
                terminal=terminal,
            )
        )

    def _exit_work_bracket(self, state: _BackgroundToolTask) -> None:
        bracket = state.work_bracket
        if bracket is None:
            return
        state.work_bracket = None
        bracket.__exit__(None, None, None)

    async def check_background(
        self,
        _args: dict[str, object],
    ) -> ToolResult:
        async with self._registry.lock:
            states = self._registry.values()
        return _tool_result(
            {"tasks": tuple(_task_payload(state) for state in states)}
        )

    async def cancel_background(
        self,
        args: dict[str, object],
    ) -> ToolResult:
        raw_task_id = args.get("task_id")
        task_id = raw_task_id if isinstance(raw_task_id, str) else ""
        async with self._registry.lock:
            state = self._registry.get(task_id)
            if state is None or state.status != _RUNNING:
                return _tool_result(
                    {
                        "error": (
                            f"cannot cancel {task_id}: unknown or already terminal"
                        )
                    },
                    is_error=True,
                )
            state.abort_signal.set("task_stop")
        return _tool_result({"task_id": task_id, "status": "cancelling"})

    async def on_session_shutdown(
        self,
        _event: SessionShutdownEvent,
    ) -> None:
        self._shutting_down = True
        async with self._registry.lock:
            states = self._registry.values()
            for state in states:
                if state.status == _RUNNING:
                    state.abort_signal.set("shutdown")
        running = [
            state.task
            for state in states
            if state.status == _RUNNING and state.task is not None
        ]
        if running:
            _, pending = await asyncio.wait(
                running,
                timeout=self._shutdown_grace_seconds,
            )
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        for state in states:
            self._exit_work_bracket(state)

    async def _drain_or_cancel(
        self,
        task: asyncio.Task[ToolResult | ToolOutcome],
    ) -> None:
        _, pending = await asyncio.wait(
            {task},
            timeout=self._shutdown_grace_seconds,
        )
        for remaining in pending:
            remaining.cancel()
        await asyncio.gather(task, return_exceptions=True)


class _BackgroundExecutor:
    def __init__(
        self,
        inner: ToolExecutor,
        manager: _BackgroundManager,
    ) -> None:
        self._inner = inner
        self._manager = manager

    def capabilities(self) -> ToolExecutionCapabilities:
        return self._inner.capabilities()

    async def execute(
        self,
        request: ToolExecutionRequest,
        *,
        signal: CancelSignal | None = None,
    ) -> ToolResult | ToolOutcome:
        if not self._manager.should_detach(request.tool.name):
            return await self._inner.execute(request, signal=signal)

        args, timeout = self._manager.prepare_call(
            request.tool.name,
            request.args,
        )
        abort_source = EventCancelSource()
        forwarder = (
            asyncio.create_task(
                _forward_cancel(signal, abort_source),
                name=f"agentm-background-forward-{request.tool.name}",
            )
            if signal is not None
            else None
        )
        inner_request = replace(request, args=args)
        inner_task = asyncio.create_task(
            self._inner.execute(inner_request, signal=abort_source),
            name=f"agentm-background-inner-{request.tool.name}",
        )
        try:
            if timeout > 0:
                await asyncio.wait_for(asyncio.shield(inner_task), timeout)
                await _stop_forwarder(forwarder)
                return inner_task.result()
        except TimeoutError:
            pass
        except asyncio.CancelledError:
            abort_source.set(signal.reason if signal is not None else "unknown")
            await _stop_forwarder(forwarder)
            if not inner_task.done():
                inner_task.cancel()
            await asyncio.gather(inner_task, return_exceptions=True)
            raise
        except BaseException:
            await _stop_forwarder(forwarder)
            raise

        # Detachment transfers cancellation ownership from the parent turn to
        # this atom before the ticket is returned.
        await _stop_forwarder(forwarder)
        note = (
            "running in background; the task is still active and its result "
            "will be delivered automatically"
            if timeout == 0
            else (
                f"moved to background after {timeout:g}s; the task is still "
                "active and its result will be delivered automatically"
            )
        )
        return await self._manager.detach(
            tool_name=request.tool.name,
            args=request.args,
            inner_task=inner_task,
            abort_source=abort_source,
            note=note,
        )


class _CheckBackgroundParams(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _CancelBackgroundParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(
        min_length=1,
        description="Task id returned by a background ticket.",
    )


class _BackgroundExecRuntime:
    def __init__(self, api: AtomAPI, config: BackgroundExecConfig) -> None:
        self._api = api
        self._manager = _BackgroundManager(
            api=api,
            timeout=config.timeout,
            denylist=frozenset(config.denylist),
            shutdown_grace_seconds=config.shutdown_grace_seconds,
        )

    def install(self) -> None:
        inner = self._api.get_tool_executor() or DirectToolExecutor()
        self._api.register_tool_executor(
            _BackgroundExecutor(inner, self._manager),
            replace=True,
        )
        self._api.on(
            SessionShutdownEvent.CHANNEL,
            self._manager.on_session_shutdown,
        )
        self._api.register_tool(
            FunctionTool(
                name="check_background",
                description=(
                    "List background tasks and their retained terminal result "
                    "tail without waiting."
                ),
                parameters=_CheckBackgroundParams,
                fn=self._manager.check_background,
            )
        )
        self._api.register_tool(
            FunctionTool(
                name="cancel_background",
                description=(
                    "Cooperatively stop a running background task by task id. "
                    "The terminal cancellation notification is delivered "
                    "asynchronously."
                ),
                parameters=_CancelBackgroundParams,
                fn=self._manager.cancel_background,
            )
        )


def install(api: AtomAPI, config: BackgroundExecConfig) -> None:
    _BackgroundExecRuntime(api, config).install()


__all__ = (
    "BackgroundExecConfig",
    "MANIFEST",
    "install",
)
