"""Builtin ``background_exec`` atom: auto-backgrounding + ticker.

Design: ``.claude/designs/session-inbox.md`` (the ``background_exec`` section
and the "Step-3 design decisions (2026-05-28)" block) — step 3 of the
Session Inbox work.

What it does (opt-in; a scenario lists it):

- At ``agent_start`` it wraps **every** registered tool (minus a ``denylist``
  and its own companion tools) in a transparent auto-bg shim. The shim runs the
  inner tool in an ``asyncio.Task`` and waits at most ``timeout`` seconds:
  - finished in time  → return the inner result **unchanged** (fast tools are
    byte-for-byte unaffected; existing tool tests must stay green);
  - overran           → register the still-running task in a
    :class:`BackgroundTaskRegistry`, spin up a per-task *ticker*, and return an
    immediate ticket ``ToolResult`` so the turn's "every tool_call gets a
    result" protocol is satisfied. The real result arrives later as a
    ``source="background"`` inbox item.
- Companion tools ``check_background`` / ``wait_background`` / ``cancel_background``
  generalize ``sub_agent``'s polling tools. ``cancel_background`` is the first
  caller of :meth:`BackgroundTaskRegistry.cancel`.
- A per-task **ticker** posts to the inbox on milestones (completion / error /
  new output / silence-too-long warning) plus a sparse heartbeat fallback, all
  under one ``dedup_key`` so a new status REPLACES the prior undrained one (no
  stacking).

Architecture: module-level helpers + a :class:`_BgManager` that owns the
registry and the wrap/ticker/completion machinery, mirroring ``sub_agent``'s
``_ChildTaskManager`` so ``install`` stays a thin wire-up entry point.

§11: single file; no atom→atom imports; ``core.lib`` / ``core.abi`` only; no
``core.runtime.*`` / ``core._internal``.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any, Literal

from agentm.core.abi import (
    AgentStartEvent,
    FunctionTool,
    TextContent,
    Tool,
    ToolContinue,
    ToolOutcome,
    ToolResult,
    ToolTerminate,
)
from agentm.core.abi.events import SessionShutdownEvent
from agentm.core.lib import DEFAULT_SHUTDOWN_GRACE_SECONDS, to_jsonable
from agentm.core.lib.background_tasks import (
    BackgroundTask,
    BackgroundTaskRegistry,
)
from agentm.extensions import ExtensionManifest
from agentm.core.abi.extension import ExtensionAPI, ExtensionStaleError

_RUNNING: Literal["running"] = "running"
_COMPLETED: Literal["completed"] = "completed"
_ERROR: Literal["error"] = "error"
_CANCELLED: Literal["cancelled"] = "cancelled"
_Status = Literal["running", "completed", "error", "cancelled"]

# The companion tools never auto-background themselves — they are pure registry
# pokes that return promptly, and backgrounding a poll would be nonsensical.
_COMPANION_TOOLS = frozenset({"check_background", "wait_background", "cancel_background"})

_DEFAULT_TIMEOUT = 60.0
_DEFAULT_HEARTBEAT = 120.0
_DEFAULT_SILENCE_WARNING = 300.0


MANIFEST = ExtensionManifest(
    name="background_exec",
    description=(
        "Auto-background any tool call that overruns a timeout; report progress "
        "to the agent via the session inbox (ticker). Opt-in per scenario."
    ),
    registers=(
        "tool:check_background",
        "tool:wait_background",
        "tool:cancel_background",
        "event:agent_start",
        "event:session_shutdown",
    ),
    config_schema={
        "type": "object",
        "properties": {
            "timeout": {
                "type": "number",
                "minimum": 0,
                "default": _DEFAULT_TIMEOUT,
                "description": (
                    f"Seconds a foreground call may run before it is moved to "
                    f"the background (default {_DEFAULT_TIMEOUT:g})."
                ),
            },
            "heartbeat_interval": {
                "type": "number",
                "minimum": 0,
                "default": _DEFAULT_HEARTBEAT,
                "description": (
                    f"Sparse heartbeat: post a 'still running' status every N "
                    f"seconds of no other milestone (default {_DEFAULT_HEARTBEAT:g})."
                ),
            },
            "silence_warning": {
                "type": "number",
                "minimum": 0,
                "default": _DEFAULT_SILENCE_WARNING,
                "description": (
                    f"Emit a silence-too-long warning once a backgrounded task "
                    f"produces no result for N seconds (default {_DEFAULT_SILENCE_WARNING:g})."
                ),
            },
            "denylist": {
                "type": "array",
                "items": {"type": "string"},
                "default": [],
                "description": "Tool names that must never be wrapped/backgrounded.",
            },
            "shutdown_grace_seconds": {
                "type": "number",
                "minimum": 0,
                "default": DEFAULT_SHUTDOWN_GRACE_SECONDS,
                "description": (
                    "Seconds the session_shutdown drain waits for still-"
                    "running background tasks to finish cooperatively before "
                    f"cancelling them (default {DEFAULT_SHUTDOWN_GRACE_SECONDS:g})."
                ),
            },
        },
        "additionalProperties": False,
    },
    requires=(),  # Defers wrapping to agent_start so tool atoms may load in any order.
)


@dataclass(slots=True, kw_only=True)
class _BgTask(BackgroundTask):
    """A backgrounded tool call carried as a :class:`BackgroundTask`.

    The generic asyncio bits (``task_id`` / ``task`` / ``abort_signal`` /
    ``status`` / ``read``) live on the base; everything below is specific to a
    detached tool coroutine and managed by this atom.
    """

    tool_name: str
    status: _Status = _RUNNING
    started_at: float = field(default_factory=time.monotonic)
    last_milestone_at: float = field(default_factory=time.monotonic)
    outcome: ToolOutcome | None = None
    error: str | None = None
    ticker: asyncio.Task[Any] | None = None
    # Optional task that forwards a host-supplied kernel signal into this
    # task's own ``abort_signal`` (see ``_BgTool.execute``). Cancelled once the
    # task is terminal so it never outlives the work it was guarding.
    forwarder: asyncio.Task[Any] | None = None
    # #179: the entered ``api.track_background`` bracket. Entered synchronously
    # in ``background`` (before the ticket returns) so a one-shot host never
    # sees a momentary "idle" while the watcher task is being scheduled; exited
    # once the completion has been posted (``_watch`` finally). ``None`` when
    # tracking was skipped (atom reloaded mid-dispatch).
    work_bracket: AbstractContextManager[None] | None = None


def _tool_result(payload: dict[str, Any], *, is_error: bool = False) -> ToolResult:
    return ToolResult(
        content=[TextContent(type="text", text=json.dumps(to_jsonable(payload)))],
        is_error=is_error,
        extras=payload,
    )


def _outcome_result(outcome: ToolResult | ToolOutcome) -> ToolResult:
    """Extract the :class:`ToolResult` from any tool return shape.

    A bare :class:`ToolResult` is treated as its own payload; a
    :class:`ToolContinue` / :class:`ToolTerminate` carries one. Inlined here
    (rather than reaching into ``core.abi.loop``'s private helper) so the atom
    depends only on the public ``core.abi`` data shapes.
    """

    if isinstance(outcome, ToolResult):
        return outcome
    if isinstance(outcome, (ToolContinue, ToolTerminate)):
        return outcome.result
    raise TypeError(f"unexpected tool outcome: {type(outcome).__name__}")


def _result_text(result: ToolResult) -> str:
    chunks = [
        block.text
        for block in result.content
        if isinstance(block, TextContent)
    ]
    return "\n".join(chunks).strip()


def _completion_note(state: _BgTask) -> str:
    """Human-readable completion / error note for the background inbox item."""

    if state.status == _COMPLETED:
        result = _outcome_result(state.outcome) if state.outcome is not None else None
        body = _result_text(result) if result is not None else ""
        head = (
            f"Background task {state.task_id} ({state.tool_name}) finished."
        )
        return f"{head}\n\nResult:\n{body}" if body else head
    if state.status == _ERROR:
        return (
            f"Background task {state.task_id} ({state.tool_name}) failed: "
            f"{state.error}"
        )
    if state.status == _CANCELLED:
        return (
            f"Background task {state.task_id} ({state.tool_name}) was cancelled."
        )
    return f"Background task {state.task_id} ({state.tool_name}): {state.status}."


def _task_payload(state: _BgTask) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "task_id": state.task_id,
        "tool_name": state.tool_name,
        "status": state.status,
        "elapsed_s": round(time.monotonic() - state.started_at, 1),
    }
    if state.error is not None:
        payload["error"] = state.error
    if state.status == _COMPLETED and state.outcome is not None:
        payload["result"] = _result_text(_outcome_result(state.outcome))
    return payload


async def _forward_abort(source: asyncio.Event, target: asyncio.Event) -> None:
    """Set ``target`` once ``source`` fires (one-directional bridge).

    Used to propagate a host-supplied kernel signal into a background task's
    own abort event WITHOUT the reverse coupling: setting ``target`` (via
    ``cancel_background``) never touches ``source``, so cancelling one task
    leaves the shared kernel signal — and every other in-flight call bound to
    it — untouched. Cancelled by the owner when the task goes terminal.
    """

    await source.wait()
    target.set()


class _BgTool:
    """Transparent auto-bg shim wrapping one registered tool.

    Presents the same ``name`` / ``description`` / ``parameters`` surface as
    the wrapped tool so the kernel and the LLM see no difference. On
    :meth:`execute` it runs the inner tool in a task and waits ``timeout``
    seconds: a fast call returns its real result unchanged; an overrun hands
    the live task to the manager and returns a ticket.
    """

    def __init__(self, wrapped: Tool, manager: _BgManager) -> None:
        self._wrapped = wrapped
        self._manager = manager
        self.name = wrapped.name
        self.description = wrapped.description
        self.parameters = wrapped.parameters

    async def execute(
        self,
        args: dict[str, Any],
        *,
        signal: asyncio.Event | None = None,
    ) -> ToolResult | ToolOutcome:
        # Always run the inner tool against a fresh PER-TASK abort event, never
        # the shared kernel ``signal``. ``cancel_background`` only ever sets this
        # per-task event, so cancelling one background task cannot abort the live
        # turn or any unrelated in-flight work that shares the kernel signal.
        abort = asyncio.Event()
        # The kernel's signal must still abort this call when it fires; forward
        # it INTO the per-task event (one direction only) rather than handing
        # the shared signal to the inner tool.
        forwarder: asyncio.Task[None] | None = None
        if signal is not None:
            forwarder = asyncio.create_task(_forward_abort(signal, abort))
        task: asyncio.Task[ToolResult | ToolOutcome] = asyncio.create_task(
            self._wrapped.execute(args, signal=abort)
        )
        done, _pending = await asyncio.wait(
            {task}, timeout=self._manager.timeout
        )
        if task in done:
            # Finished within the foreground window → byte-for-byte unchanged.
            # A foreground ToolTerminate (sub-timeout) passes through verbatim.
            if forwarder is not None:
                forwarder.cancel()
            return task.result()
        # Overran: leave it running in the background, return a ticket. The
        # manager takes ownership of the forwarder and tears it down on the
        # task's terminal transition.
        return await self._manager.background(
            tool_name=self.name,
            task=task,
            abort_signal=abort,
            forwarder=forwarder,
        )


class _BgManager:
    """Per-session registry + ticker/completion machinery for backgrounded
    tool calls."""

    def __init__(
        self,
        *,
        api: ExtensionAPI,
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
        # No max_workers cap for backgrounded tool calls (each is a single
        # coroutine, not a whole child session): the registry's documented
        # unbounded contract (``max_workers=None``) skips slot accounting
        # entirely, so this atom can register handles without ever paying for
        # a reserve_slot → release_slot round-trip.
        self._registry: BackgroundTaskRegistry[_BgTask] = BackgroundTaskRegistry(
            max_workers=None
        )
        # Guards the shutdown handler against a background() racing in after the
        # bus has been cleared: once set, background() refuses to register.
        self._shutting_down = False

    # --- install-time tool wrapping ---------------------------------------

    def wrap_tools(self) -> None:
        """Replace every wrappable tool in ``api.tools`` with a ``_BgTool``.

        Idempotent across repeated ``agent_start`` fires via the
        ``isinstance(tool, _BgTool)`` skip alone — no run-once flag — so tools
        registered BETWEEN prompts (e.g. by a later ``install_atom``) still get
        wrapped on the next ``agent_start``. Companion tools and denylisted
        names are left as-is; tools already wrapped are skipped.
        """

        tools = self._api.tools
        for index, tool in enumerate(tools):
            if tool.name in _COMPANION_TOOLS or tool.name in self._denylist:
                continue
            if isinstance(tool, _BgTool):
                continue
            tools[index] = _BgTool(tool, self)

    # --- backgrounding -----------------------------------------------------

    async def background(
        self,
        *,
        tool_name: str,
        task: asyncio.Task[ToolResult | ToolOutcome],
        abort_signal: asyncio.Event,
        forwarder: asyncio.Task[None] | None = None,
    ) -> ToolResult:
        """Register an overran tool task and return its immediate ticket."""

        if self._shutting_down:
            # The session is tearing down; do not strand new detached tasks on
            # a bus that is about to be cleared. ``on_session_shutdown`` has
            # already run and drained the registry, so nothing else will await
            # this never-registered inner task — we MUST bring it to a terminal
            # state here, or a non-cooperative inner tool leaks ("Task was
            # destroyed but it is pending"). Same bounded-grace-then-cancel
            # shape as the shutdown drain: ask cooperatively, give it the grace
            # window, then cancel and gather so nothing outlives the refusal.
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
            task=task,
            abort_signal=abort_signal,
            forwarder=forwarder,
        )
        # #179: enter the work-tracking bracket NOW, synchronously, before the
        # watcher task is scheduled and the ticket returns — so a one-shot host
        # (``agentm -p``) regards this detached tool as live work the instant it
        # is backgrounded, with no window where it could mistake the session for
        # idle. ``_watch`` exits the bracket once the completion has posted. A
        # mid-dispatch atom reload (ExtensionStaleError) skips tracking — the
        # inbox is gone, there is nothing to keep alive for.
        try:
            bracket = self._api.track_background()
            bracket.__enter__()
            state.work_bracket = bracket
        except ExtensionStaleError:
            state.work_bracket = None
        # Watch completion + drive the ticker from one wrapper task so the
        # original tool task stays exactly what the inner tool returned.
        state.task = asyncio.create_task(self._watch(state, task))
        state.ticker = asyncio.create_task(self._ticker(state))
        # NOTE: registered WITHOUT a paired reserve_slot() — the work is already
        # running, there is nothing to fail-fast on. The registry clamps its
        # reservation counter so this no-reservation register stays at zero
        # rather than going negative.
        await self._registry.register(state)
        return _tool_result(
            {
                "task_id": task_id,
                "status": _RUNNING,
                "note": f"moved to background after {self.timeout:g}s",
            }
        )

    async def _watch(
        self,
        state: _BgTask,
        inner: asyncio.Task[ToolResult | ToolOutcome],
    ) -> None:
        """Await the inner tool task, record its outcome, post completion.

        #179: the work-tracking bracket is ENTERED in :meth:`background`
        (synchronously, before the ticket returns) so a one-shot host never
        sees a momentary "idle" while this detached task is being scheduled.
        It is EXITED here in ``finally`` once the completion has been posted, so
        the session stays non-idle until the result has actually landed in the
        inbox. The bracket exit always runs, so the count cannot leak.
        """

        try:
            try:
                outcome = await inner
            except asyncio.CancelledError:
                state.status = _CANCELLED
                self._finalize(state)
                return
            except Exception as exc:  # noqa: BLE001
                state.status = _ERROR
                state.error = str(exc) or exc.__class__.__name__
                self._finalize(state)
                return
            # Normalize a bare ToolResult to ToolContinue so downstream
            # rendering has one shape (the kernel does the same for foreground
            # returns).
            state.outcome = (
                outcome
                if isinstance(outcome, ToolOutcome)
                else ToolContinue(result=outcome)
            )
            # #177: a backgrounded tool that ultimately returns ToolTerminate
            # carries a terminate intent. ``_post_completion`` posts the
            # completion with ``terminal=True`` so the runtime drain seam routes
            # it through loop termination (Stop(ToolTerminated)) once the message
            # has been delivered, instead of swallowing the intent as an ordinary
            # completion.
            state.status = _COMPLETED
            self._finalize(state)
        finally:
            self._exit_work_tracking(state)

    def _exit_work_tracking(self, state: _BgTask) -> None:
        """Exit the #179 work-tracking bracket entered in :meth:`background`.

        Idempotent: clears the stored bracket so a double ``_watch`` finally (it
        cannot happen, but defensively) does not double-decrement the counter.
        """

        bracket = state.work_bracket
        if bracket is None:
            return
        state.work_bracket = None
        bracket.__exit__(None, None, None)

    def _finalize(self, state: _BgTask) -> None:
        """Tear down per-task helpers and post the terminal completion.

        Called from ``_watch`` on every terminal transition. Cancels the ticker
        and the (optional) signal forwarder so neither outlives the work, then
        posts the completion. ``post_inbox`` can raise ``ExtensionStaleError``
        if the atom was reloaded mid-flight — that is swallowed in
        ``_post_completion`` so a detached task never dies with an unretrieved
        exception.
        """

        state.last_milestone_at = time.monotonic()
        self._stop_ticker(state)
        if state.forwarder is not None and not state.forwarder.done():
            state.forwarder.cancel()
        self._post_completion(state)

    def _post_completion(self, state: _BgTask) -> None:
        """Post the terminal background result to the inbox (milestone).

        Swallows ``ExtensionStaleError`` (atom reloaded between dispatch and
        completion): the inbox we hold is stale, there is nothing to deliver
        into, so we stop gracefully rather than crash the detached task.
        """

        # ``read`` is read/written here WITHOUT the registry lock, while
        # ``check_background`` flips it under the lock — intentionally. The
        # ``bg-complete-{task_id}`` dedup_key below is what makes the unlocked
        # partner safe: a racing double-post collapses on that key, so the worst
        # outcome is a harmless redundant post, never a duplicate inbox item. Do
        # NOT "fix" this into a lock here — taking the registry lock from a
        # finalize path that can run under that same lock risks a deadlock.
        if state.read:
            return
        state.read = True
        # #177: a backgrounded ToolTerminate posts terminal=True so the runtime
        # stops the loop after delivering this completion. Any other terminal
        # transition (completed / error / cancelled) is an ordinary completion.
        terminal = state.status == _COMPLETED and isinstance(
            state.outcome, ToolTerminate
        )
        try:
            self._api.post_inbox(
                source="background",
                payload=_completion_note(state),
                dedup_key=f"bg-complete-{state.task_id}",
                terminal=terminal,
            )
        except ExtensionStaleError:
            return

    # --- ticker ------------------------------------------------------------

    async def _ticker(self, state: _BgTask) -> None:
        """Per-task milestone-driven ticker with a sparse heartbeat fallback.

        Every push carries the same ``dedup_key`` so a new status REPLACES the
        prior undrained one (no stacking). Completion / error / cancel
        milestones are posted by :meth:`_post_completion` (a different,
        terminal dedup_key); this loop owns the silence-warning and heartbeat
        fallbacks while the task is still running.
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
                        f"Background task {state.task_id} ({state.tool_name}) "
                        f"has produced no output for {silent_for:.0f}s — it may "
                        f"be stuck."
                    )
                else:
                    note = (
                        f"Background task {state.task_id} ({state.tool_name}) "
                        f"still running ({time.monotonic() - state.started_at:.0f}s)."
                    )
                self._api.post_inbox(
                    source="background",
                    payload=note,
                    dedup_key=f"bg-ticker-{state.task_id}",
                )
        except asyncio.CancelledError:
            return
        except ExtensionStaleError:
            # Atom reloaded while this detached ticker was sleeping: the inbox
            # we hold is stale. Stop gracefully (no unretrieved exception).
            return

    def _stop_ticker(self, state: _BgTask) -> None:
        if state.ticker is not None and not state.ticker.done():
            state.ticker.cancel()

    # --- companion tools ---------------------------------------------------

    async def check_background(self, _args: dict[str, Any]) -> ToolResult:
        """List task states, blocking on the first running task to complete.

        A terminal task surfaced here is marked ``read`` (like
        ``sub_agent.check_tasks``) so the completion reported in this tool
        result is NOT also re-injected into the inbox by ``_watch`` — the agent
        would otherwise see the same completion twice.
        """

        await self._registry.poll_first_completed()
        async with self._registry.lock:
            tasks = self._registry.values()
            for state in tasks:
                if state.status != _RUNNING:
                    state.read = True
        return _tool_result({"tasks": [_task_payload(state) for state in tasks]})

    async def wait_background(self, args: dict[str, Any]) -> ToolResult:
        """Block until one task reaches a terminal state, then report it."""

        task_id = str(args.get("task_id", ""))
        async with self._registry.lock:
            if self._registry.get(task_id) is None:
                return _tool_result(
                    {"error": f"unknown task_id: {task_id}"}, is_error=True
                )
        await self._registry.wait_one(task_id)
        async with self._registry.lock:
            state = self._registry.get(task_id)
            assert state is not None
            payload = _task_payload(state)
        return _tool_result(payload)

    async def cancel_background(self, args: dict[str, Any]) -> ToolResult:
        """Cooperatively cancel a running task via the registry.

        First caller of :meth:`BackgroundTaskRegistry.cancel`: sets the task's
        abort signal. The status flip to ``cancelled`` happens in
        :meth:`_watch` once the inner task observes the signal (or is cancelled).
        """

        task_id = str(args.get("task_id", ""))
        cancelled = await self._registry.cancel(task_id)
        if not cancelled:
            return _tool_result(
                {
                    "error": (
                        f"cannot cancel {task_id}: unknown or already terminal"
                    )
                },
                is_error=True,
            )
        return _tool_result({"task_id": task_id, "status": "cancelling"})

    # --- session shutdown --------------------------------------------------

    async def on_session_shutdown(self, _event: SessionShutdownEvent) -> None:
        """Drain detached tasks so none leak past ``session.shutdown()``.

        Without this the inner tool, its ``_watch`` wrapper, the ticker, and any
        signal forwarder stay pending when the bus is cleared, producing "Task
        was destroyed but it is pending" and leaving the inner tool running
        detached. Mirrors ``sub_agent.on_session_shutdown``: cancel each ticker
        and forwarder, give the still-running watch tasks a grace window to
        finish on their own, then set their abort signal and gather everything.
        """

        self._shutting_down = True
        async with self._registry.lock:
            states = self._registry.values()
        for state in states:
            self._stop_ticker(state)
            if state.forwarder is not None and not state.forwarder.done():
                state.forwarder.cancel()
        running = [state for state in states if state.status == _RUNNING]
        if not running:
            return
        watches = [state.task for state in running]
        _done, still_running = await asyncio.wait(
            watches, timeout=self._shutdown_grace_seconds
        )
        if still_running:
            for state in running:
                if state.task in still_running:
                    state.abort_signal.set()
            await asyncio.gather(*still_running, return_exceptions=True)
        # #179 nit: ``_watch``'s ``finally`` is the normal bracket-exit, but a
        # task cancelled BEFORE its body first executes never runs that
        # ``finally`` and would leak the work counter. Exit any still-held
        # bracket here — ``_exit_work_tracking`` is idempotent (it nulls the
        # stored bracket), so a task that already exited in its own ``finally``
        # is a no-op. Guarantees the "exit always runs" invariant on shutdown.
        for state in states:
            self._exit_work_tracking(state)


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    manager = _BgManager(
        api=api,
        timeout=float(config.get("timeout", _DEFAULT_TIMEOUT)),
        heartbeat_interval=float(config.get("heartbeat_interval", _DEFAULT_HEARTBEAT)),
        silence_warning=float(config.get("silence_warning", _DEFAULT_SILENCE_WARNING)),
        denylist={str(name) for name in config.get("denylist", [])},
        shutdown_grace_seconds=float(
            config.get("shutdown_grace_seconds", DEFAULT_SHUTDOWN_GRACE_SECONDS)
        ),
    )

    def on_agent_start(_: AgentStartEvent) -> None:
        manager.wrap_tools()

    api.on(AgentStartEvent.CHANNEL, on_agent_start)
    api.on(SessionShutdownEvent.CHANNEL, manager.on_session_shutdown)
    api.register_tool(
        FunctionTool(
            name="check_background",
            description=(
                "List background tasks (state + elapsed). Blocks until the "
                "first still-running task completes, then returns all states."
            ),
            parameters={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            fn=manager.check_background,
        )
    )
    api.register_tool(
        FunctionTool(
            name="wait_background",
            description="Wait for one background task to reach a terminal state.",
            parameters={
                "type": "object",
                "properties": {"task_id": {"type": "string"}},
                "required": ["task_id"],
                "additionalProperties": False,
            },
            fn=manager.wait_background,
        )
    )
    api.register_tool(
        FunctionTool(
            name="cancel_background",
            description="Request cancellation of a running background task.",
            parameters={
                "type": "object",
                "properties": {"task_id": {"type": "string"}},
                "required": ["task_id"],
                "additionalProperties": False,
            },
            fn=manager.cancel_background,
        )
    )
