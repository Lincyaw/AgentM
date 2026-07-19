"""Builtin ``background_exec`` atom: auto-backgrounding + ticker.

Design: ``.claude/designs/session-inbox.md`` (the ``background_exec`` section
and producer wiring decisions).

What it does (opt-in; a scenario lists it):

- It wraps a registered tool in a transparent auto-bg shim (:class:`_BgTool`).
  The shim runs the inner tool in an ``asyncio.Task`` and waits for whichever
  happens first: completion or ``timeout`` seconds:
  - finished in time  → return the inner result **unchanged**;
  - overran → register the still-running task in a
    :class:`BackgroundTaskRegistry`, spin up a per-task *ticker*, and return an
    immediate ticket ``ToolResult``. The real result arrives later as a
    ``BackgroundCompletion`` trigger.
- Companion tools ``check_background`` / ``cancel_background`` expose direct
  controls for backgrounded tool calls.
- A per-task **ticker** posts a running/heartbeat ``BackgroundCompletion``
  trigger on a sparse interval while the task is still running.

TODO(migration): several ``main``-branch seams have no equivalent on this
branch and are disabled here (see inline markers):
  * ``api.tools`` is not exposed as a mutable list, so automatic wrapping of
    every registered tool at ``agent_start`` cannot happen. ``wrap_tools`` is a
    no-op; the machinery below (``_BgTool`` / ``_BgManager.background``) is fully
    translated and driven through ``_BgManager.background`` for whoever
    re-introduces a tool-list hook.
  * ``wait_inbox_nonempty`` (soft-preempt a foreground tool on pending user
    input) has no equivalent — the foreground wait is timeout-only.
  * ``BackgroundActivityEvent`` (presenter chrome) does not exist — activity
    emission is dropped.
  * The bash live-output tail / source-side log streaming (``bash_output_tails``
    service + resource-writer log files) is dropped for coherence.
  * Inbox delivery (``post_inbox`` with ``dedup_key``) is replaced by
    ``push_trigger(BackgroundCompletion(...))``; there is no dedup surface, so
    ticker posts are best-effort.

§11: single file; no atom→atom imports; ``core.lib`` / ``core.abi`` only.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any, Literal

from loguru import logger

from agentm.core.abi import (
    AtomAPI,
    AtomInstallPriority,
    BackgroundCompletion,
    BeforeRunEvent,
    CancelSignal,
    EventCancelSource,
    FunctionTool,
    SessionShutdownEvent,
    TextContent,
    Tool,
    ToolContinue,
    ToolOutcome,
    ToolResult,
    ToolTerminate,
)
from agentm.core.lib import (
    BackgroundTask,
    BackgroundTaskRegistry,
    pydantic_to_tool_schema,
    to_jsonable,
)
from agentm.core.lib.tool_executor import execute_tool_call
from agentm.extensions import ExtensionManifest
from pydantic import BaseModel, Field

# main imported this from core.lib; not exported on this branch.
DEFAULT_SHUTDOWN_GRACE_SECONDS = 5.0

_RUNNING: Literal["running"] = "running"
_COMPLETED: Literal["completed"] = "completed"
_ERROR: Literal["error"] = "error"
_CANCELLED: Literal["cancelled"] = "cancelled"
_Status = Literal["running", "completed", "error", "cancelled"]

# The companion tools never auto-background themselves — they are pure registry
# pokes that return promptly.
_COMPANION_TOOLS = frozenset({"check_background", "cancel_background"})

_DEFAULT_TIMEOUT = 60.0
_DEFAULT_HEARTBEAT = 480.0
_DEFAULT_SILENCE_WARNING = 900.0
_MAX_ACTIVITY_LABEL_CHARS = 96
_COMPLETION_PREVIEW_CHARS = 2000


class BackgroundExecConfig(BaseModel):
    timeout: float = _DEFAULT_TIMEOUT
    heartbeat_interval: float = _DEFAULT_HEARTBEAT
    silence_warning: float = _DEFAULT_SILENCE_WARNING
    denylist: list[str] = []
    shutdown_grace_seconds: float = DEFAULT_SHUTDOWN_GRACE_SECONDS


MANIFEST = ExtensionManifest(
    name="background_exec",
    description=(
        "Auto-background any tool call that overruns a timeout; report progress "
        "to the agent via BackgroundCompletion triggers. Opt-in per scenario."
    ),
    registers=(
        "tool:check_background",
        "tool:cancel_background",
        "event:before_run",
        "event:session_shutdown",
    ),
    config_schema=BackgroundExecConfig,
    requires=(),
    priority=AtomInstallPriority.TOOL,
)


@dataclass(slots=True, kw_only=True)
class _BgTask(BackgroundTask):
    """A backgrounded tool call carried as a :class:`BackgroundTask`."""

    tool_name: str
    label: str
    status: _Status = _RUNNING
    started_at: float = field(default_factory=time.monotonic)
    last_milestone_at: float = field(default_factory=time.monotonic)
    outcome: ToolOutcome | None = None
    error: str | None = None
    ticker: asyncio.Task[Any] | None = None
    forwarder: asyncio.Task[Any] | None = None
    # #179: the entered ``api.track_background`` bracket. Entered synchronously
    # in ``background`` (before the ticket returns); exited once the completion
    # has been posted (``_watch`` finally). ``None`` when tracking was skipped.
    work_bracket: AbstractContextManager[None] | None = None


def _tool_result(payload: dict[str, Any], *, is_error: bool = False) -> ToolResult:
    return ToolResult(
        content=[TextContent(type="text", text=json.dumps(to_jsonable(payload)))],
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
    chunks = [block.text for block in result.content if isinstance(block, TextContent)]
    return "\n".join(chunks).strip()


def _single_line(value: str) -> str:
    return " ".join(value.split())


def _truncate_label(value: str) -> str:
    if len(value) <= _MAX_ACTIVITY_LABEL_CHARS:
        return value
    return value[: _MAX_ACTIVITY_LABEL_CHARS - 3].rstrip() + "..."


def _activity_label(tool_name: str, args: dict[str, Any]) -> str:
    if tool_name in {"bash", "shell"}:
        raw_command = args.get("cmd")
        if raw_command is None:
            raw_command = args.get("command")
        if isinstance(raw_command, str):
            command = _single_line(raw_command)
            if command:
                return _truncate_label(f"{tool_name}: {command}")
    return tool_name


def _completion_note(state: _BgTask) -> str:
    if state.status == _COMPLETED:
        result = _outcome_result(state.outcome) if state.outcome is not None else None
        body = _result_text(result) if result is not None else ""
        head = f"Background task {state.task_id} ({state.label}) finished."
        if not body:
            return head
        if len(body) > _COMPLETION_PREVIEW_CHARS:
            tail = body[-_COMPLETION_PREVIEW_CHARS:]
            return (
                f"{head}\n\nResult (last {_COMPLETION_PREVIEW_CHARS} chars of "
                f"{len(body)} — full output via check_background):\n...{tail}"
            )
        return f"{head}\n\nResult:\n{body}"
    if state.status == _ERROR:
        return f"Background task {state.task_id} ({state.label}) failed: {state.error}"
    if state.status == _CANCELLED:
        return f"Background task {state.task_id} ({state.label}) was cancelled."
    return f"Background task {state.task_id} ({state.label}): {state.status}."


_CHECK_BG_TAIL_LINES: int = 10


def _last_n_lines(text: str, n: int) -> str:
    lines = text.splitlines(keepends=True)
    if len(lines) <= n:
        return text
    return "".join(lines[-n:])


def _task_payload(state: _BgTask) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "task_id": state.task_id,
        "tool_name": state.tool_name,
        "label": state.label,
        "status": state.status,
        "elapsed_s": round(time.monotonic() - state.started_at, 1),
    }
    if state.error is not None:
        payload["error"] = state.error
    if state.status == _COMPLETED and state.outcome is not None:
        full = _result_text(_outcome_result(state.outcome))
        payload["result"] = _last_n_lines(full, _CHECK_BG_TAIL_LINES)
    return payload


async def _forward_abort(source: CancelSignal, target: EventCancelSource) -> None:
    """Set ``target`` once ``source`` fires (one-directional bridge)."""

    await source.wait()
    target.set()


class _BgTool:
    """Transparent auto-bg shim wrapping one registered tool.

    Presents the same ``name`` / ``parameters`` surface as the wrapped tool;
    the wrapped ``description`` gets an auto-background note appended. On
    :meth:`execute` it runs the inner tool in a task and waits for completion
    or timeout: a fast call returns its real result unchanged; timeout hands
    the live task to the manager and returns a ticket.
    """

    def __init__(self, wrapped: Tool, manager: _BgManager) -> None:
        self._wrapped = wrapped
        self._manager = manager
        self.name = wrapped.name
        self.description = wrapped.description + self._bg_note(
            wrapped.name, manager.timeout
        )
        self.parameters = self._bg_parameters(
            wrapped.name, wrapped.parameters, manager.timeout
        )
        metadata = getattr(wrapped, "metadata", {})
        self.metadata = dict(metadata) if isinstance(metadata, dict) else {}
        # TODO(migration): main pinned the wrapper to the session event loop via
        # TOOL_EXECUTION_DOMAIN_METADATA_KEY (not present on this branch).

    @staticmethod
    def _bg_note(tool_name: str, timeout: float) -> str:
        note = (
            f"\n\nNote: if this call runs longer than {timeout:g}s, it is moved "
            "to the background and returns a {task_id, status: \"running\"} "
            "ticket instead of its normal result; the real result arrives later "
            "as an automatic notification. Use check_background / "
            "cancel_background to inspect or stop it."
        )
        if tool_name in {"bash", "shell"}:
            note += (
                f" A timeout argument below {timeout:g}s is treated as the "
                "move-to-background point, not a kill deadline — the command "
                f"then keeps running with a {timeout:g}s kill deadline."
            )
        return note

    @staticmethod
    def _bg_parameters(
        tool_name: str, parameters: dict[str, Any], timeout: float
    ) -> dict[str, Any]:
        if tool_name not in {"bash", "shell"}:
            return parameters
        props = parameters.get("properties")
        if not isinstance(props, dict) or "timeout" not in props:
            return parameters
        timeout_prop = props["timeout"]
        patched_prop = dict(timeout_prop) if isinstance(timeout_prop, dict) else {}
        patched_prop["description"] = (
            "Seconds before this command is moved to the background — NOT a "
            "kill deadline: the command keeps running there (kill deadline "
            f"{timeout:g}s) and its result arrives as a new message. Values of "
            f"{timeout:g}s or more act as a real kill deadline instead."
        )
        return {
            **parameters,
            "properties": {
                **props,
                "timeout": patched_prop,
                "background": {
                    "type": "boolean",
                    "description": (
                        "Run this command in the background immediately. "
                        "The result will be delivered automatically as a "
                        "new message when the command finishes."
                    ),
                },
            },
        }

    async def execute(
        self,
        args: dict[str, Any],
        *,
        signal: CancelSignal | None = None,
    ) -> ToolResult | ToolOutcome:
        run_in_background = bool(args.get("background"))
        clean_args = (
            {k: v for k, v in args.items() if k != "background"}
            if "background" in args
            else args
        )
        # Always run the inner tool against a fresh PER-TASK abort source, never
        # the shared kernel ``signal``. ``cancel_background`` only ever sets this
        # per-task source, so cancelling one background task cannot abort the
        # live turn or any unrelated in-flight work.
        abort = EventCancelSource()
        forwarder: asyncio.Task[None] | None = None
        if signal is not None:
            forwarder = asyncio.create_task(_forward_abort(signal, abort))
        inner_args, foreground_timeout = self._manager.prepare_foreground_call(
            self.name, clean_args
        )
        if run_in_background:
            foreground_timeout = 0
        task: asyncio.Task[ToolResult | ToolOutcome] = asyncio.create_task(
            execute_tool_call(self._wrapped, inner_args, signal=abort),
            name=f"agentm-bg-inner-{self.name}",
        )
        try:
            foreground_done, reason = await self._wait_foreground(
                task, timeout=foreground_timeout
            )
        except asyncio.CancelledError:
            if forwarder is not None:
                forwarder.cancel()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            raise
        if foreground_done:
            if forwarder is not None:
                forwarder.cancel()
            return task.result()
        return await self._manager.background(
            tool_name=self.name,
            args=args,
            task=task,
            abort_signal=abort,
            forwarder=forwarder,
            note=reason,
        )

    async def _wait_foreground(
        self,
        task: asyncio.Task[ToolResult | ToolOutcome],
        *,
        timeout: float,
    ) -> tuple[bool, str]:
        """Wait until the tool finishes or times out.

        TODO(migration): main also raced ``wait_inbox_nonempty`` to soft-preempt
        on pending user input; no equivalent exists on this branch.
        """

        if timeout <= 0:
            done, _pending = await asyncio.wait({task}, timeout=0)
            return task in done, (
                "Running in background — still running, NOT terminated. The "
                "result will be delivered to you automatically as a new message "
                "when it finishes. While waiting, continue with other work. "
                "Do not cancel or poll"
            )

        timeout_task = asyncio.create_task(asyncio.sleep(timeout))
        try:
            done, _pending = await asyncio.wait(
                {task, timeout_task}, return_when=asyncio.FIRST_COMPLETED
            )
            if task in done:
                return True, ""
            return (
                False,
                f"moved to background after {timeout:g}s — still running, NOT "
                "terminated. The result will be delivered to you automatically "
                "as a new message when it finishes. While waiting, continue with "
                "other work. Do not cancel or poll",
            )
        finally:
            timeout_task.cancel()


class _BgManager:
    """Per-session registry + ticker/completion machinery for backgrounded
    tool calls."""

    def __init__(
        self,
        *,
        api: AtomAPI,
        timeout: float,
        heartbeat_interval: float,
        silence_warning: float,
        denylist: set[str],
        shutdown_grace_seconds: float = DEFAULT_SHUTDOWN_GRACE_SECONDS,
    ) -> None:
        self._api = api
        self.timeout = timeout
        self._heartbeat = heartbeat_interval
        self._silence_warning = silence_warning
        self._denylist = denylist
        self._shutdown_grace_seconds = shutdown_grace_seconds
        self._registry: BackgroundTaskRegistry[_BgTask] = BackgroundTaskRegistry(
            max_workers=None
        )
        self._shutting_down = False

    # --- install-time tool wrapping ---------------------------------------

    def wrap_tools(self) -> None:
        """Replace every wrappable tool with a ``_BgTool``.

        TODO(migration): ``api.tools`` is not exposed as a mutable list on this
        branch, so there is no way to wrap the registered tools in place. This
        is a no-op; the ``_BgTool`` / ``background`` machinery below is fully
        translated and ready for whoever adds a tool-list mutation hook.
        """

        logger.debug(
            "background_exec: wrap_tools is a no-op — api.tools is not exposed "
            "on this branch (auto-backgrounding disabled)"
        )

    def prepare_foreground_call(
        self, tool_name: str, args: dict[str, Any]
    ) -> tuple[dict[str, Any], float]:
        """Return inner args plus the foreground handoff timeout."""

        foreground_timeout = self.timeout
        if tool_name not in {"bash", "shell"}:
            return args, foreground_timeout

        raw_timeout = args.get("timeout")
        if raw_timeout is None:
            return args, foreground_timeout
        try:
            requested_timeout = float(raw_timeout)
        except (TypeError, ValueError):
            return args, foreground_timeout
        if requested_timeout <= 0:
            return args, foreground_timeout

        foreground_timeout = min(requested_timeout, self.timeout)
        if requested_timeout >= self.timeout:
            return args, foreground_timeout

        inner_args = dict(args)
        inner_args["timeout"] = self.timeout
        return inner_args, foreground_timeout

    # --- backgrounding -----------------------------------------------------

    async def background(
        self,
        *,
        tool_name: str,
        args: dict[str, Any],
        task: asyncio.Task[ToolResult | ToolOutcome],
        abort_signal: EventCancelSource,
        forwarder: asyncio.Task[None] | None = None,
        note: str | None = None,
    ) -> ToolResult:
        """Register an overran tool task and return its immediate ticket."""

        if self._shutting_down:
            abort_signal.set()
            if forwarder is not None:
                forwarder.cancel()
            _done, still_running = await asyncio.wait(
                {task}, timeout=self._shutdown_grace_seconds
            )
            if still_running:
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            return _tool_result(
                {"error": "session is shutting down; background task refused"},
                is_error=True,
            )
        task_id = uuid.uuid4().hex
        state = _BgTask(
            task_id=task_id,
            tool_name=tool_name,
            label=_activity_label(tool_name, args=args),
            task=task,
            abort_signal=abort_signal,
            forwarder=forwarder,
        )
        # #179: enter the work-tracking bracket NOW, synchronously, before the
        # watcher task is scheduled and the ticket returns.
        try:
            bracket = self._api.track_background()
            bracket.__enter__()
            state.work_bracket = bracket
        except Exception as exc:  # noqa: BLE001
            logger.debug("background_exec: track_background skipped: {}", exc)
            state.work_bracket = None
        state.task = asyncio.create_task(self._watch(state, task))
        state.ticker = asyncio.create_task(self._ticker(state))
        await self._registry.register(state)
        ticket_note = note or f"moved to background after {self.timeout:g}s"
        return _tool_result(
            {
                "task_id": task_id,
                "status": _RUNNING,
                "note": ticket_note,
            }
        )

    async def _watch(
        self,
        state: _BgTask,
        inner: asyncio.Task[ToolResult | ToolOutcome],
    ) -> None:
        """Await the inner tool task, record its outcome, post completion."""

        try:
            try:
                outcome = await inner
            except asyncio.CancelledError:
                state.status = _CANCELLED
                self._finalize(state)
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "background_exec: tool {} raised: {}", state.tool_name, exc
                )
                state.status = _ERROR
                state.error = str(exc) or exc.__class__.__name__
                self._finalize(state)
                return
            state.outcome = (
                outcome
                if isinstance(outcome, ToolOutcome)
                else ToolContinue(result=outcome)
            )
            if state.abort_signal.is_set():
                state.status = _CANCELLED
                self._finalize(state)
                return
            state.status = _COMPLETED
            self._finalize(state)
        finally:
            self._exit_work_tracking(state)

    def _exit_work_tracking(self, state: _BgTask) -> None:
        bracket = state.work_bracket
        if bracket is None:
            return
        state.work_bracket = None
        bracket.__exit__(None, None, None)

    def _finalize(self, state: _BgTask) -> None:
        state.last_milestone_at = time.monotonic()
        self._stop_ticker(state)
        if state.forwarder is not None and not state.forwarder.done():
            state.forwarder.cancel()
        self._post_completion(state)

    def _post_completion(self, state: _BgTask) -> None:
        if state.read:
            return
        state.read = True
        note = _completion_note(state)
        # #177: a backgrounded ToolTerminate posts terminal=True so the runtime
        # stops the loop after delivering this completion.
        loop_terminal = state.status == _COMPLETED and isinstance(
            state.outcome, ToolTerminate
        )
        try:
            self._api.push_trigger(
                BackgroundCompletion(
                    task_id=state.task_id,
                    payload=note,
                    terminal=loop_terminal,
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("background_exec: completion push failed: {}", exc)

    # --- ticker ------------------------------------------------------------

    async def _ticker(self, state: _BgTask) -> None:
        """Per-task heartbeat ticker.

        TODO(migration): main used a per-task ``dedup_key`` so a new status
        REPLACED the prior undrained one; BackgroundCompletion triggers have no
        dedup surface, so heartbeat posts accumulate. Kept sparse via the
        heartbeat interval.
        """

        warned_silence = False
        try:
            while state.status == _RUNNING:
                await asyncio.sleep(self._heartbeat)
                if state.status != _RUNNING:
                    return
                silent_for = time.monotonic() - state.last_milestone_at
                if (
                    not warned_silence
                    and self._silence_warning > 0
                    and silent_for >= self._silence_warning
                ):
                    warned_silence = True
                    note = (
                        f"Background task {state.task_id} ({state.label}) "
                        f"has produced no output for {silent_for:.0f}s — it may "
                        f"be stuck."
                    )
                else:
                    note = (
                        f"Background task {state.task_id} ({state.label}) "
                        f"still running ({time.monotonic() - state.started_at:.0f}s)."
                    )
                try:
                    self._api.push_trigger(
                        BackgroundCompletion(
                            task_id=state.task_id, payload=note, terminal=False
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.debug("background_exec: ticker push failed: {}", exc)
                    return
        except asyncio.CancelledError:
            return

    def _stop_ticker(self, state: _BgTask) -> None:
        if state.ticker is not None and not state.ticker.done():
            state.ticker.cancel()

    # --- companion tools ---------------------------------------------------

    async def check_background(self, _args: dict[str, Any]) -> ToolResult:
        async with self._registry.lock:
            tasks = self._registry.values()
            for state in tasks:
                if state.status != _RUNNING:
                    state.read = True
        return _tool_result({"tasks": [_task_payload(state) for state in tasks]})

    async def cancel_background(self, args: dict[str, Any]) -> ToolResult:
        task_id = str(args.get("task_id", ""))
        cancelled = await self._registry.cancel(task_id)
        if not cancelled:
            return _tool_result(
                {"error": (f"cannot cancel {task_id}: unknown or already terminal")},
                is_error=True,
            )
        return _tool_result({"task_id": task_id, "status": "cancelling"})

    # --- session shutdown --------------------------------------------------

    async def on_session_shutdown(self, _event: SessionShutdownEvent) -> None:
        """Drain detached tasks so none leak past ``session.shutdown()``."""

        self._shutting_down = True
        async with self._registry.lock:
            states = self._registry.values()
        for state in states:
            self._stop_ticker(state)
            if state.forwarder is not None and not state.forwarder.done():
                state.forwarder.cancel()
        running = [state for state in states if state.status == _RUNNING]
        if running:
            watches = [state.task for state in running]
            _done, still_running = await asyncio.wait(
                watches, timeout=self._shutdown_grace_seconds
            )
            if still_running:
                for state in running:
                    if state.task in still_running:
                        state.abort_signal.set()
                await asyncio.gather(*still_running, return_exceptions=True)
        for state in states:
            self._exit_work_tracking(state)


# Tool schemas (Pydantic -> JSON Schema via pydantic_to_tool_schema)
# ---------------------------------------------------------------------------


class _CheckBackgroundParams(BaseModel):
    pass


class _CancelBackgroundParams(BaseModel):
    task_id: str = Field(
        description=(
            "Task id from the background ticket result or check_background."
        ),
    )


class _BackgroundExecRuntime:
    def __init__(self, api: AtomAPI, config: BackgroundExecConfig) -> None:
        self._api = api
        self._manager = _BgManager(
            api=api,
            timeout=config.timeout,
            heartbeat_interval=config.heartbeat_interval,
            silence_warning=config.silence_warning,
            denylist=set(config.denylist),
            shutdown_grace_seconds=config.shutdown_grace_seconds,
        )

    def install(self) -> None:
        self._api.on(BeforeRunEvent.CHANNEL, self.on_before_run)
        self._api.on(SessionShutdownEvent.CHANNEL, self._manager.on_session_shutdown)
        self._register_tools()

    def on_before_run(self, _: BeforeRunEvent) -> None:
        self._manager.wrap_tools()

    def _register_tools(self) -> None:
        self._api.register_tool(
            FunctionTool(
                name="check_background",
                description=(
                    "List background tasks (state + elapsed) without waiting. "
                    "Long-running tool calls are automatically moved to the "
                    "background — this does NOT mean they failed or timed out. "
                    "You will be notified automatically when a background task "
                    "completes, so you do not need to poll this tool repeatedly. "
                    "Returns {tasks: [{task_id, tool_name, label, status, "
                    "elapsed_s, result?, error?}]}. Seeing a finished task here "
                    "consumes its pending completion notification."
                ),
                parameters=pydantic_to_tool_schema(_CheckBackgroundParams),
                fn=self._manager.check_background,
            )
        )
        self._api.register_tool(
            FunctionTool(
                name="cancel_background",
                description=(
                    "Request cancellation of a running background task. "
                    "Only cancel if you are certain the task is stuck — "
                    "a task that was moved to background is still running "
                    "and may complete successfully. Cancellation is "
                    'cooperative: this returns {task_id, status: "cancelling"} '
                    "immediately, and the actual stop is confirmed later as a "
                    "new message. Unknown or already-finished task ids return "
                    "an error."
                ),
                parameters=pydantic_to_tool_schema(_CancelBackgroundParams),
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
