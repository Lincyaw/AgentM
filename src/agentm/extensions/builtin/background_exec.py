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
from agentm.core.lib import to_jsonable
from agentm.core.lib.background_tasks import (
    BackgroundTask,
    BackgroundTaskRegistry,
)
from agentm.extensions import ExtensionManifest
from agentm.core.abi.extension import ExtensionAPI

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
    ),
    config_schema={
        "type": "object",
        "properties": {
            "timeout": {
                "type": "number",
                "minimum": 0,
                "default": _DEFAULT_TIMEOUT,
                "description": (
                    "Seconds a foreground call may run before it is moved to "
                    "the background (default 60)."
                ),
            },
            "heartbeat_interval": {
                "type": "number",
                "minimum": 0,
                "default": _DEFAULT_HEARTBEAT,
                "description": (
                    "Sparse heartbeat: post a 'still running' status every N "
                    "seconds of no other milestone (default 120)."
                ),
            },
            "silence_warning": {
                "type": "number",
                "minimum": 0,
                "default": _DEFAULT_SILENCE_WARNING,
                "description": (
                    "Emit a silence-too-long warning once a backgrounded task "
                    "produces no result for N seconds (default 300)."
                ),
            },
            "denylist": {
                "type": "array",
                "items": {"type": "string"},
                "default": [],
                "description": "Tool names that must never be wrapped/backgrounded.",
            },
        },
        "additionalProperties": True,
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
        abort = asyncio.Event()
        # The kernel's signal must still abort the in-flight call; chain it into
        # the per-task abort so cancel_background and the kernel both work.
        inner_signal = signal if signal is not None else abort
        task: asyncio.Task[ToolResult | ToolOutcome] = asyncio.create_task(
            self._wrapped.execute(args, signal=inner_signal)
        )
        done, _pending = await asyncio.wait(
            {task}, timeout=self._manager.timeout
        )
        if task in done:
            # Finished within the foreground window → byte-for-byte unchanged.
            # A foreground ToolTerminate (sub-timeout) passes through verbatim.
            return task.result()
        # Overran: leave it running in the background, return a ticket.
        return await self._manager.background(
            tool_name=self.name,
            task=task,
            abort_signal=abort if signal is None else signal,
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
    ) -> None:
        self._api = api
        self.timeout = timeout
        self._heartbeat = heartbeat_interval
        self._silence_warning = silence_warning
        self._denylist = denylist
        # No max_workers cap for backgrounded tool calls (each is a single
        # coroutine, not a whole child session); use a large ceiling so
        # reserve_slot never trips — the manager creates tasks directly.
        self._registry: BackgroundTaskRegistry[_BgTask] = BackgroundTaskRegistry(
            max_workers=1_000_000
        )
        self._wrapped = False

    # --- install-time tool wrapping ---------------------------------------

    def wrap_tools(self) -> None:
        """Replace every wrappable tool in ``api.tools`` with a ``_BgTool``.

        Idempotent across repeated ``agent_start`` fires. Companion tools and
        denylisted names are left as-is; tools already wrapped are skipped.
        """

        if self._wrapped:
            return
        tools = self._api.tools
        for index, tool in enumerate(tools):
            if tool.name in _COMPANION_TOOLS or tool.name in self._denylist:
                continue
            if isinstance(tool, _BgTool):
                continue
            tools[index] = _BgTool(tool, self)
        self._wrapped = True

    # --- backgrounding -----------------------------------------------------

    async def background(
        self,
        *,
        tool_name: str,
        task: asyncio.Task[ToolResult | ToolOutcome],
        abort_signal: asyncio.Event,
    ) -> ToolResult:
        """Register an overran tool task and return its immediate ticket."""

        task_id = uuid.uuid4().hex
        state = _BgTask(
            task_id=task_id,
            tool_name=tool_name,
            task=task,
            abort_signal=abort_signal,
        )
        # Watch completion + drive the ticker from one wrapper task so the
        # original tool task stays exactly what the inner tool returned.
        state.task = asyncio.create_task(self._watch(state, task))
        state.ticker = asyncio.create_task(self._ticker(state))
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
        """Await the inner tool task, record its outcome, post completion."""

        try:
            outcome = await inner
        except asyncio.CancelledError:
            state.status = _CANCELLED
            state.last_milestone_at = time.monotonic()
            self._stop_ticker(state)
            self._post_completion(state)
            return
        except Exception as exc:  # noqa: BLE001
            state.status = _ERROR
            state.error = str(exc) or exc.__class__.__name__
            state.last_milestone_at = time.monotonic()
            self._stop_ticker(state)
            self._post_completion(state)
            return
        # Normalize a bare ToolResult to ToolContinue so downstream rendering
        # has one shape (the kernel does the same for foreground returns).
        state.outcome = (
            outcome if isinstance(outcome, ToolOutcome) else ToolContinue(result=outcome)
        )
        # Terminal-from-background simplification (step 3): a backgrounded tool
        # that ultimately returns ToolTerminate is injected as an ordinary
        # completion — the terminate intent does NOT stop the loop here.
        # TODO(step 5): once the persistent driver exists, route a backgrounded
        # ToolTerminate so it can actually end the loop.
        state.status = _COMPLETED
        state.last_milestone_at = time.monotonic()
        self._stop_ticker(state)
        self._post_completion(state)

    def _post_completion(self, state: _BgTask) -> None:
        """Post the terminal background result to the inbox (milestone)."""

        if state.read:
            return
        state.read = True
        self._api.post_inbox(
            source="background",
            payload=_completion_note(state),
            dedup_key=f"bg-complete-{state.task_id}",
        )

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

    def _stop_ticker(self, state: _BgTask) -> None:
        if state.ticker is not None and not state.ticker.done():
            state.ticker.cancel()

    # --- companion tools ---------------------------------------------------

    async def check_background(self, _args: dict[str, Any]) -> ToolResult:
        """List task states, blocking on the first running task to complete."""

        await self._registry.poll_first_completed()
        async with self._registry.lock:
            tasks = self._registry.values()
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


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    manager = _BgManager(
        api=api,
        timeout=float(config.get("timeout", _DEFAULT_TIMEOUT)),
        heartbeat_interval=float(config.get("heartbeat_interval", _DEFAULT_HEARTBEAT)),
        silence_warning=float(config.get("silence_warning", _DEFAULT_SILENCE_WARNING)),
        denylist={str(name) for name in config.get("denylist", [])},
    )

    def on_agent_start(_: AgentStartEvent) -> None:
        manager.wrap_tools()

    api.on(AgentStartEvent.CHANNEL, on_agent_start)
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
