"""Builtin ``background_exec`` atom: auto-backgrounding + ticker.

Design: ``.claude/designs/session-inbox.md`` (the ``background_exec`` section
and producer wiring decisions).

What it does (opt-in; a scenario lists it):

- At ``agent_start`` it wraps **every** registered tool (minus a ``denylist``
  and its own companion tools) in a transparent auto-bg shim. The shim runs the
  inner tool in an ``asyncio.Task`` and waits for whichever happens first:
  completion, ``timeout`` seconds, or a new core-inbox item:
  - finished in time  → return the inner result **unchanged** (fast tools are
    byte-for-byte unaffected; existing tool tests must stay green);
  - overran / inbox input arrived → register the still-running task in a
    :class:`BackgroundTaskRegistry`, spin up a per-task *ticker*, and return an
    immediate ticket ``ToolResult`` so the turn's "every tool_call gets a
    result" protocol is satisfied. The real result arrives later as a
    ``source="background"`` inbox item.
- Companion tools ``check_background`` / ``cancel_background``
  expose direct controls for backgrounded tool calls. ``cancel_background`` is
  the first caller of :meth:`BackgroundTaskRegistry.cancel`.
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

from loguru import logger

from agentm.core.abi import (
    AgentStartEvent,
    BackgroundActivityEvent,
    ExtensionAPI,
    ExtensionStaleError,
    FunctionTool,
    SessionShutdownEvent,
    TOOL_EXECUTION_DOMAIN_EVENT_LOOP,
    TOOL_EXECUTION_DOMAIN_METADATA_KEY,
    TextContent,
    Tool,
    ToolContinue,
    ToolOutcome,
    ToolResult,
    ToolTerminate,
    execute_tool_call,
)
from agentm.core.lib import (
    BackgroundTask,
    BackgroundTaskRegistry,
    DEFAULT_SHUTDOWN_GRACE_SECONDS,
    to_jsonable,
)
from pydantic import BaseModel, Field

from agentm.core.lib import pydantic_to_tool_schema
from agentm.extensions import ExtensionManifest

_RUNNING: Literal["running"] = "running"
_COMPLETED: Literal["completed"] = "completed"
_ERROR: Literal["error"] = "error"
_CANCELLED: Literal["cancelled"] = "cancelled"
_Status = Literal["running", "completed", "error", "cancelled"]
_ACTIVITY_TERMINAL_STATUSES: frozenset[_Status] = frozenset(
    {_COMPLETED, _ERROR, _CANCELLED}
)

# The companion tools never auto-background themselves — they are pure registry
# pokes that return promptly, and backgrounding a poll would be nonsensical.
_COMPANION_TOOLS = frozenset(
    {"check_background", "cancel_background"}
)

_DEFAULT_TIMEOUT = 60.0
_DEFAULT_HEARTBEAT = 480.0
_DEFAULT_SILENCE_WARNING = 900.0
_MAX_ACTIVITY_LABEL_CHARS = 96
# Inbox completion notes bypass tool_result_cap — keep only a tail preview
# there; the full output stays reachable via check_background.
_COMPLETION_PREVIEW_CHARS = 2000
_INBOX_GRACE_SECONDS = 0.5

# Which inbox sources justify soft-preempting a running foreground tool. Must
# exclude this atom's OWN ``source="background"`` ticker/completion posts: if
# those counted, each detach would post a completion that keeps the inbox
# non-empty and forces the next tool to detach too — a self-sustaining loop.
# Design intent is user input only (session-inbox.md: "soft-preempt on pending
# user input").
_PREEMPT_SOURCES = frozenset({"user"})


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
        "to the agent via the session inbox (ticker). Opt-in per scenario."
    ),
    registers=(
        "tool:check_background",
        "tool:cancel_background",
        "event:agent_start",
        "event:background_activity",
        "event:session_shutdown",
    ),
    config_schema=BackgroundExecConfig,
    requires=(),  # Defers wrapping to agent_start so tool atoms may load in any order.
    api_version=1,
    tier=1,
)


@dataclass(slots=True, kw_only=True)
class _BgTask(BackgroundTask):
    """A backgrounded tool call carried as a :class:`BackgroundTask`.

    The generic asyncio bits (``task_id`` / ``task`` / ``abort_signal`` /
    ``status`` / ``read``) live on the base; everything below is specific to a
    detached tool coroutine and managed by this atom.
    """

    tool_name: str
    label: str
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
    # Key into the ``bash_output_tails`` service (tool_bash) for this call's
    # live-output tail buffer; ``None``-tail for non-streaming tools.
    output_key: str | None = None
    # Workspace-relative live log file the backend tees this call's output
    # into at the execution site (see BashOperations.exec log_path).
    log_path: str | None = None


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
    chunks = [block.text for block in result.content if isinstance(block, TextContent)]
    return "\n".join(chunks).strip()


def _single_line(value: str) -> str:
    return " ".join(value.split())


def _truncate_label(value: str) -> str:
    if len(value) <= _MAX_ACTIVITY_LABEL_CHARS:
        return value
    return value[: _MAX_ACTIVITY_LABEL_CHARS - 3].rstrip() + "..."


def _activity_label(tool_name: str, args: dict[str, Any]) -> str:
    """Presenter-facing label for a detached tool task.

    Keep the generic tool name as the stable fallback, but make shell work
    identifiable in terminal chrome. The full arguments still live in the
    regular tool call transcript; this label is intentionally compact.
    """

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
    """Human-readable completion / error note for the background inbox item.

    Inbox payloads bypass the tool-result pipeline, so ``tool_result_cap``
    never sees them — bound the body to a tail preview here and point at
    ``check_background`` (whose result DOES go through the cap) for the
    full output.
    """

    if state.status == _COMPLETED:
        result = _outcome_result(state.outcome) if state.outcome is not None else None
        body = _result_text(result) if result is not None else ""
        head = f"Background task {state.task_id} ({state.label}) finished."
        if not body:
            return head
        if len(body) > _COMPLETION_PREVIEW_CHARS:
            tail = body[-_COMPLETION_PREVIEW_CHARS:]
            where = (
                f"check_background or read {state.log_path}"
                if state.log_path is not None
                else "check_background"
            )
            return (
                f"{head}\n\nResult (last {_COMPLETION_PREVIEW_CHARS} chars of "
                f"{len(body)} — full output via {where}):\n...{tail}"
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


def _task_payload(state: _BgTask, *, tails: Any | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "task_id": state.task_id,
        "tool_name": state.tool_name,
        "label": state.label,
        "status": state.status,
        "elapsed_s": round(time.monotonic() - state.started_at, 1),
    }
    if state.error is not None:
        payload["error"] = state.error
    if state.log_path is not None:
        payload["log_path"] = state.log_path
    if state.status == _COMPLETED and state.outcome is not None:
        full = _result_text(_outcome_result(state.outcome))
        payload["result"] = _last_n_lines(full, _CHECK_BG_TAIL_LINES)
    elif (
        state.status == _RUNNING
        and tails is not None
        and state.output_key is not None
    ):
        tail = tails.tail(state.output_key)
        if tail:
            payload["latest_output"] = _last_n_lines(tail, _CHECK_BG_TAIL_LINES)
    return payload


def _activity_id(task_id: str) -> str:
    return f"background:{task_id}"


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

    Presents the same ``name`` / ``parameters`` surface as the wrapped tool;
    the wrapped ``description`` gets an auto-background note appended so the
    model knows a long call returns a ticket instead of its normal result. On
    :meth:`execute` it runs the inner tool in a task and waits for completion,
    timeout, or pending core inbox input: a fast call returns its real result
    unchanged; timeout or input arrival hands the live task to the manager and
    returns a ticket.
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
        # Keep the wrapper itself on the session event loop because it touches
        # the session inbox/registry. The wrapped tool's own execution domain
        # is still honored below through execute_tool_call().
        self.metadata[TOOL_EXECUTION_DOMAIN_METADATA_KEY] = (
            TOOL_EXECUTION_DOMAIN_EVENT_LOOP
        )

    @staticmethod
    def _bg_note(tool_name: str, timeout: float) -> str:
        note = (
            f"\n\nNote: if this call runs longer than {timeout:g}s (or new "
            "user input arrives first), it is moved to the background and "
            'returns a {task_id, status: "running"} ticket instead of its '
            "normal result; the real result arrives later as an automatic "
            "inbox notification. Use check_background / cancel_background "
            "to inspect or stop it."
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
        """Rewrite the shell ``timeout`` param description to its real
        semantics under auto-backgrounding: a foreground-handoff window, not
        a kill deadline (see ``_BgManager.prepare_foreground_call``)."""

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
                        "new message when the command finishes. While "
                        "waiting, continue with other work — review your "
                        "code, check for issues, or read related files. "
                        "Use for long-running commands (compilation, test "
                        "suites, large builds)."
                    ),
                },
            },
        }

    async def execute(
        self,
        args: dict[str, Any],
        *,
        signal: asyncio.Event | None = None,
    ) -> ToolResult | ToolOutcome:
        run_in_background = bool(args.get("background"))
        clean_args = (
            {k: v for k, v in args.items() if k != "background"}
            if "background" in args
            else args
        )
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
        inner_args, foreground_timeout = self._manager.prepare_foreground_call(
            self.name, clean_args
        )
        if run_in_background:
            foreground_timeout = 0
        # Bind a live-output tail key around task creation so the bash tool
        # (if this wraps it) streams a bounded tail into a readable buffer
        # AND tees its full output into a source-side log file (written
        # locally or inside the remote sandbox — log bytes never round-trip
        # through the host). The contextvar snapshot propagates into the
        # inner task.
        output_key = uuid.uuid4().hex
        log_path: str | None = None
        tails = self._manager.output_tails()
        tail_token = None
        if tails is not None:
            if self.name in {"bash", "shell"}:
                log_path = self._manager.make_log_path(output_key)
            tail_token = tails.bind(output_key, log_path=log_path)
        try:
            task: asyncio.Task[ToolResult | ToolOutcome] = asyncio.create_task(
                execute_tool_call(self._wrapped, inner_args, signal=abort),
                name=f"agentm-bg-inner-{self.name}",
            )
        finally:
            if tails is not None and tail_token is not None:
                tails.unbind(tail_token)
        try:
            foreground_done, reason = await self._wait_foreground(
                task, timeout=foreground_timeout
            )
        except asyncio.CancelledError:
            if forwarder is not None:
                forwarder.cancel()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            self._manager.discard_call_artifacts(output_key, log_path)
            raise
        if foreground_done:
            # Finished within the foreground window → byte-for-byte unchanged.
            # A foreground ToolTerminate (sub-timeout) passes through verbatim.
            # The source-side log was only insurance for a detach that never
            # happened — drop it along with the tail buffer.
            if forwarder is not None:
                forwarder.cancel()
            self._manager.discard_call_artifacts(output_key, log_path)
            return task.result()
        # Overran or user input arrived: leave it running in the background,
        # return a ticket. The manager takes ownership of the forwarder and
        # tears it down on the task's terminal transition.
        return await self._manager.background(
            tool_name=self.name,
            args=args,
            task=task,
            abort_signal=abort,
            forwarder=forwarder,
            note=reason,
            output_key=output_key,
            log_path=log_path,
        )

    async def _wait_foreground(
        self,
        task: asyncio.Task[ToolResult | ToolOutcome],
        *,
        timeout: float,
    ) -> tuple[bool, str]:
        """Wait until the tool finishes, times out, or core inbox gets input."""

        if timeout <= 0:
            done, _pending = await asyncio.wait({task}, timeout=0)
            return task in done, (
                "Running in background — still running, NOT terminated. "
                "The result will be delivered to you automatically as a new "
                "message when it finishes. While waiting, continue with "
                "other work — review your code, check for potential issues, "
                "or read related files. Do not cancel or poll"
            )

        timeout_task = asyncio.create_task(asyncio.sleep(timeout))
        inbox_task = asyncio.create_task(self._manager.wait_inbox_nonempty())
        try:
            waiters: set[asyncio.Task[Any]] = {task, timeout_task, inbox_task}
            while True:
                done, _pending = await asyncio.wait(
                    waiters, return_when=asyncio.FIRST_COMPLETED
                )
                if task in done:
                    return True, ""
                if timeout_task in done:
                    return (
                        False,
                        f"moved to background after {timeout:g}s — still "
                        "running, NOT terminated. The result will be "
                        "delivered to you automatically as a new message "
                        "when it finishes. While waiting, continue with "
                        "other work — review your code, check for potential "
                        "issues, or read related files. Do not cancel or poll",
                    )
                if inbox_task in done:
                    if inbox_task.result():
                        return (
                            False,
                            "moved to background because new user input is "
                            "pending — still running, NOT terminated. The "
                            "result will be delivered automatically as a "
                            "new message when it finishes",
                        )
                    await asyncio.sleep(0.05)
                    waiters.remove(inbox_task)
                    inbox_task = asyncio.create_task(
                        self._manager.wait_inbox_nonempty()
                    )
                    waiters.add(inbox_task)
        finally:
            timeout_task.cancel()
            inbox_task.cancel()


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
        # Fire-and-forget cleanup tasks (undetached-call log deletion) held
        # here so they are not garbage-collected mid-flight.
        self._cleanup_tasks: set[asyncio.Task[Any]] = set()

    def output_tails(self) -> Any | None:
        """The ``bash_output_tails`` service (tool_bash), if mounted.

        Resolved lazily on every use — install order between the two atoms is
        unconstrained and the service can appear or vanish across reloads.
        """

        try:
            return self._api.get_service("bash_output_tails")
        except ExtensionStaleError:
            return None

    def _emit_activity(
        self,
        state: _BgTask,
        *,
        note: str | None = None,
        terminal: bool = False,
    ) -> None:
        events = getattr(self._api, "events", None)
        if events is None:
            return
        try:
            events.emit_sync(
                BackgroundActivityEvent.CHANNEL,
                BackgroundActivityEvent(
                    source="background",
                    activity_id=_activity_id(state.task_id),
                    label=state.label,
                    status=state.status,
                    session_id=self._api.session_id,
                    note=note,
                    terminal=terminal,
                ),
            )
        except ExtensionStaleError:
            return

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

    def prepare_foreground_call(
        self, tool_name: str, args: dict[str, Any]
    ) -> tuple[dict[str, Any], float]:
        """Return inner args plus the foreground handoff timeout.

        Shell-oriented models often pass a tiny ``timeout`` when they want a
        long-running command to leave the foreground quickly. If that value is
        handed straight to ``tool_bash`` the shell is killed before this atom can
        detach it. Treat short shell timeouts as the foreground handoff window
        while extending the inner shell timeout to this atom's background cap.
        """

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
        abort_signal: asyncio.Event,
        forwarder: asyncio.Task[None] | None = None,
        note: str | None = None,
        output_key: str | None = None,
        log_path: str | None = None,
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
            if output_key is not None:
                self.discard_call_artifacts(output_key, log_path)
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
            output_key=output_key,
            log_path=log_path,
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
        ticket_note = note or f"moved to background after {self.timeout:g}s"
        self._emit_activity(state, note=ticket_note)
        payload: dict[str, Any] = {
            "task_id": task_id,
            "status": _RUNNING,
            "note": ticket_note,
        }
        if state.log_path is not None:
            payload["log_path"] = state.log_path
        return _tool_result(payload)

    def _session_dir_name(self) -> str:
        session_id = getattr(self._api, "session_id", None)
        if isinstance(session_id, str) and session_id:
            return session_id
        return "unknown-session"

    def make_log_path(self, output_key: str) -> str:
        """Workspace-relative source-side log path for one wrapped call."""

        return f".agentm/tool_outputs/{self._session_dir_name()}/bg_{output_key}.log"

    def discard_call_artifacts(self, output_key: str, log_path: str | None) -> None:
        """Drop the tail buffer and (fire-and-forget) the source-side log of a
        call that never detached — the log was only insurance for a detach."""

        tails = self.output_tails()
        if tails is not None:
            tails.discard(output_key)
        if log_path is None:
            return
        try:
            writer = self._api.get_resource_writer()
        except ExtensionStaleError:
            return
        cleanup = asyncio.create_task(
            writer.delete(log_path, rationale="background_exec: undetached call log")
        )
        self._cleanup_tasks.add(cleanup)
        cleanup.add_done_callback(self._cleanup_tasks.discard)

    async def wait_inbox_nonempty(self) -> bool:
        try:
            return await self._api.wait_inbox_nonempty(sources=_PREEMPT_SOURCES)
        except ExtensionStaleError:
            return False

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
        if state.output_key is not None:
            tails = self.output_tails()
            if tails is not None:
                tails.discard(state.output_key)
            state.output_key = None
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
        note = _completion_note(state)
        # #177: a backgrounded ToolTerminate posts terminal=True to the inbox so
        # the runtime stops the loop after delivering this completion. Activity
        # terminal means "remove or retain the chrome row" and is intentionally
        # broader: any final task state closes the presenter activity.
        loop_terminal = state.status == _COMPLETED and isinstance(
            state.outcome, ToolTerminate
        )
        self._emit_activity(
            state,
            note=note,
            terminal=state.status in _ACTIVITY_TERMINAL_STATUSES,
        )
        try:
            self._api.post_inbox(
                source="background",
                payload=note,
                dedup_key=f"bg-complete-{state.task_id}",
                terminal=loop_terminal,
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
                last_signal = state.last_milestone_at
                if state.output_key is not None:
                    tails = self.output_tails()
                    last_data = (
                        tails.last_data_at(state.output_key)
                        if tails is not None
                        else None
                    )
                    if last_data is not None:
                        last_signal = max(last_signal, last_data)
                silent_for = time.monotonic() - last_signal
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
                self._emit_activity(state, note=note)
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
        """List task states without waiting for running work to complete.

        Terminal tasks surfaced here are marked ``read`` so the completion
        reported in this tool result is NOT also re-injected into the inbox by
        ``_watch`` — the agent would otherwise see the same completion twice.
        """

        async with self._registry.lock:
            tasks = self._registry.values()
            for state in tasks:
                if state.status != _RUNNING:
                    state.read = True
        tails = self.output_tails()
        return _tool_result(
            {"tasks": [_task_payload(state, tails=tails) for state in tasks]}
        )

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
                {"error": (f"cannot cancel {task_id}: unknown or already terminal")},
                is_error=True,
            )
        async with self._registry.lock:
            state = self._registry.get(task_id)
            if state is not None:
                self._emit_activity(state, note="cancellation requested")
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
    def __init__(self, api: ExtensionAPI, config: BackgroundExecConfig) -> None:
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
        self._api.on(AgentStartEvent.CHANNEL, self.on_agent_start)
        self._api.on(SessionShutdownEvent.CHANNEL, self._manager.on_session_shutdown)
        self._register_tools()

    def on_agent_start(self, _: AgentStartEvent) -> None:
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
                    "elapsed_s, latest_output?, log_path?, result?, "
                    "error?}]}: a running shell task shows the tail of its "
                    "live output in `latest_output`, and `log_path` is a "
                    "workspace file its full output streams into (read it "
                    "for history beyond the tail). A finished task's full "
                    "output is inlined in `result` (oversized results are "
                    "spilled to a file whose path replaces the overflow). "
                    "Seeing a finished task here consumes its pending "
                    "completion notification."
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
                    "and may complete successfully. Do NOT cancel a task "
                    "just because it was moved to background; wait for it "
                    "to finish. Cancellation is cooperative: this returns "
                    '{task_id, status: "cancelling"} immediately, and the '
                    "actual stop is confirmed later as a new message. "
                    "Unknown or already-finished task ids return an error."
                ),
                parameters=pydantic_to_tool_schema(_CancelBackgroundParams),
                fn=self._manager.cancel_background,
            )
        )


def install(api: ExtensionAPI, config: BackgroundExecConfig) -> None:
    _BackgroundExecRuntime(api, config).install()


__all__ = (
    "BackgroundExecConfig",
    "MANIFEST",
    "install",
)
