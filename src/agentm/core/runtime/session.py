"""AgentSession: thin façade over the event bus, session manager, and agent loop."""

from __future__ import annotations

import asyncio
from loguru import logger
import os
import shutil
from pathlib import Path
from typing import Any

from agentm.core.abi import (
    AgentEndEvent,
    AgentMessage,
    DecideTurnActionEvent,
    DiagnosticEvent,
    EventBus,
    ImageContent,
    Inject,
    Model,
    NoPendingInput,
    Step,
    Stop,
    TextContent,
    Tool,
    ToolTerminated,
    TurnObservation,
)
from agentm.core.abi.events import (
    BeforeAgentStartEvent,
    ChildSessionEndEvent,
    InputEvent,
    MessagePersistedEvent,
    SessionShutdownEvent,
)
from agentm.core.abi.events import ContextEvent
from agentm.core.abi.session import ENTRY_TYPE_TURN_COMMITTED
from agentm.core.abi.loop import resolve_loop_action
from agentm.core.lib import DEFAULT_SHUTDOWN_GRACE_SECONDS, bind_read_state_session
from agentm.core.runtime.resource_loader import ResourceLoader
from agentm.core.runtime.session_helpers import (
    collect_start_veto,
    collect_system_replacement,
)
from agentm.core.runtime.session_inbox import (
    InboxItem,
    SessionInbox,
    render_item,
)
from agentm.core.runtime.session_factory import SessionRuntime
from agentm.core.runtime.session_manager import SessionManager


# Grace window for ``AgentSession.shutdown`` to await the persistent driver
# task before forcing cancellation. Pulled from the shared substrate default
# so atoms (``background_exec`` / ``sub_agent`` / ``monitor``) and the
# session driver use the same number — see ``core.lib.shutdown``. This is
# the substrate-private callsite and intentionally has no config knob; atoms
# expose ``shutdown_grace_seconds`` themselves.
_DRIVER_SHUTDOWN_GRACE_SECONDS = DEFAULT_SHUTDOWN_GRACE_SECONDS

_DEFAULT_EVENT_LOOP_LAG_INTERVAL_SECONDS = 1.0
_DEFAULT_EVENT_LOOP_LAG_WARNING_SECONDS = 5.0
_EVENT_LOOP_LAG_RATE_LIMIT_SECONDS = 60.0
_EVENT_LOOP_LAG_INTERVAL_ENV = "AGENTM_EVENT_LOOP_LAG_INTERVAL_SECONDS"
_EVENT_LOOP_LAG_WARNING_ENV = "AGENTM_EVENT_LOOP_LAG_WARNING_SECONDS"


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning(
            "invalid {}={!r}; using default {}",
            name,
            raw,
            default,
        )
        return default


# --- Config -----------------------------------------------------------------
# ``AgentSessionConfig`` lives in ``core.abi.session_config`` so extensions
# (which cannot import this module per §11.4.5) can still construct one for
# ``api.spawn_child_session``. Re-exported here for ergonomic access.

from agentm.core.abi.session_config import (  # noqa: E402
    AgentSessionConfig,
)


# --- AgentSession -----------------------------------------------------------


class AgentSession:

    def __init__(
        self,
        *,
        cwd: str,
        runtime: SessionRuntime,
        session_id: str,
        parent_bus: EventBus | None,
        parent_session_id: str | None,
        eval_sandbox: Path | None = None,
    ) -> None:
        self._cwd = cwd
        self._bus = runtime.bus
        self._session_manager = runtime.session_manager
        self._resources = runtime.resource_loader
        self._loop = runtime.loop
        self._active_provider_ref = runtime.active_provider_ref
        self._tools = runtime.tools
        self._commands = runtime.commands
        self._providers = runtime.providers
        self._renderers = runtime.renderers
        self._apis = runtime.apis
        self._services = runtime.services
        self._reloader = runtime.reloader
        self._extension_api = (
            next(iter(runtime.apis.values())) if runtime.apis else None
        )
        self._inbox: SessionInbox = runtime.inbox
        self._session_id = session_id
        self._parent_bus = parent_bus
        self._parent_session_id = parent_session_id
        # OTel trace_id shared with every child session. Stamped by the
        # session factory (see ``create_agent_session``) onto the API
        # scope; we mirror it here so embedders can pluck it off the
        # ``AgentSession`` object directly (e.g. to forward into a
        # downstream eval row, an upstream OTel exporter, etc.).
        self._root_session_id: str = (
            self._extension_api.root_session_id if self._extension_api else session_id
        )
        # Set by the cost_budget extension via the cost_budget_exceeded
        # channel; checked at the top of ``prompt`` so the next turn
        # short-circuits cleanly with stop_reason="budget".
        self._budget_exceeded: bool = False
        # Per-session sandbox for ``atom_source_overrides`` (per-task-evolution
        # loop §6.3). Populated by ``AgentSession.create`` when the config
        # supplies overrides; cleaned up on ``shutdown``. ``None`` for
        # ordinary sessions — no filesystem cost.
        self._eval_sandbox: Path | None = eval_sandbox

        # --- step-5 driver state ------------------------------------------
        # The persistent driver task — always-on owner of ``_loop.run``. It
        # blocks on ``inbox.wait_nonempty`` while idle and runs one round per
        # wake. ``_closed`` flips at shutdown so the driver loop exits cleanly;
        # ``_signal`` is the kernel abort event ``interrupt()`` sets to
        # preempt an in-flight ``run``. ``_in_run`` enforces single ownership:
        # any other caller invoking ``_loop.run`` while the driver owns it
        # asserts immediately rather than silently racing.
        self._closed: bool = False
        self._signal: asyncio.Event = asyncio.Event()
        self._in_run: bool = False
        # SET while the driver is parked; CLEARED when it wakes to run.
        self._parked: asyncio.Event = asyncio.Event()
        self._end_waiters: list[asyncio.Future[None]] = []
        # FIFO-serializes concurrent prompt() callers.
        self._prompt_lock: asyncio.Lock = asyncio.Lock()
        # Latched by a terminal=True inbox item; consumed at turn boundary.
        self._pending_terminate: ToolTerminated | None = None
        self._last_round_turn: tuple[int, int] | None = None

        self._bus.on(
            MessagePersistedEvent.CHANNEL, self._on_message_persisted
        )

        self._bus.on(ContextEvent.CHANNEL, self._on_context_drain_inbox)
        self._bus.on(
            DecideTurnActionEvent.CHANNEL, self._on_decide_inbox_keep_alive
        )
        # Registered BEFORE the wake handler so the durable boundary marker is
        # persisted before ``prompt``/``tick`` waiters resolve and the caller
        # may inspect or shut the session down.
        self._bus.on(AgentEndEvent.CHANNEL, self._on_agent_end_commit_boundary)
        self._bus.on(AgentEndEvent.CHANNEL, self._on_agent_end_wake_waiters)
        try:
            asyncio.get_running_loop()
        except RuntimeError as exc:
            raise RuntimeError(
                "AgentSession must be constructed inside a running asyncio "
                "event loop (use ``AgentSession.create``; the persistent "
                "driver task cannot be scheduled without a loop)"
            ) from exc
        self._driver_task: asyncio.Task[None] = asyncio.create_task(
            self._driver(), name=f"agentm-session-driver-{session_id}"
        )
        self._watchdog_task: asyncio.Task[None] | None = (
            self._spawn_event_loop_lag_watchdog()
        )

    # --- SessionInbox runtime handlers ------------------------------------

    def _on_context_drain_inbox(self, event: ContextEvent) -> None:
        """Drain the inbox into the turn's message list (in place) + persist.

        The ``context`` channel lets handlers mutate ``event.messages`` in
        place (loop.py:472); we append the drained, rendered messages so the
        prefix stays stable and the KV/prefix cache survives (append-only, per
        the design's cache-discipline note).
        """

        self._drain_inbox_and_persist(messages=event.messages)

    def _on_decide_inbox_keep_alive(
        self, event: DecideTurnActionEvent
    ) -> Step | Stop | None:
        """Keep the loop alive while the inbox holds undrained items.

        Returning ``Step()`` defers to the next turn, whose ``context``
        handler drains the inbox. A ``final=True`` default (budget / signal /
        max_turns) overrides this via ``resolve_loop_action`` (loop.py:301),
        so a hard ceiling stays hard.

        #177: if a ``terminal=True`` item was drained (recorded on
        ``_pending_terminate``) and the inbox is now empty, return
        ``Stop(ToolTerminated)`` so a backgrounded ``ToolTerminate`` actually
        ends the loop instead of being swallowed as an ordinary completion.
        """

        del event
        if not self._inbox.is_empty():
            return Step()
        # #177: a terminal item was drained — honour the terminate intent now
        # that its message has been delivered. Non-final ``Stop`` (matching a
        # foreground ``ToolTerminate``), so an extension handler on the same
        # channel may still ``Inject`` over it.
        #
        # We do NOT clear ``_pending_terminate`` here. The resolution lattice
        # (loop.py:298) lets a co-loaded floor's ``Inject`` win over this
        # ``Stop`` — e.g. ``sub_agent`` injecting while a child is still
        # running. A foreground terminate survives that because it is the
        # kernel default, recomputed every turn; a one-shot consumed flag would
        # be DESTROYED by the override. So we re-assert the same cause on every
        # boundary and let ``_on_agent_end`` clear it only once the loop has
        # actually stopped on it — the terminate survives across N turns of a
        # running child's injects.
        if self._pending_terminate is not None:
            return Stop(self._pending_terminate)
        return None

    def _drain_inbox_and_persist(
        self, *, messages: list[AgentMessage]
    ) -> None:
        """Drain the inbox into the live turn message list and persist.

        Called from the ``context`` handler at the top of each kernel turn
        (loop.py:466) — the ONLY caller, so ``messages`` is always the
        turn's live list. For each drained item we render it, persist it
        via the session manager (so a mid-run kill still leaves every
        message on disk — same contract as loop.py:609), and append it to
        ``messages`` so the LLM sees it on this very turn.
        """

        for item in self._inbox.drain():
            rendered = render_item(item)
            self._session_manager.append_message(rendered)
            messages.append(rendered)
            # #177: a terminal item carries a terminate intent. Record it so
            # the keep-alive floor (this turn's ``decide_turn_action``) stops
            # the loop after the message above has been delivered to the model.
            # ``source`` is recorded in the cause for trace attribution; a
            # background-produced terminate has no foreground tool_name.
            if item.terminal:
                self._pending_terminate = ToolTerminated(
                    tool_name=f"background:{item.source}",
                    reason="terminate-from-background",
                )

    def _drain_inbox_on_early_return(self, reason: str) -> None:
        """Drain the inbox after a pre-context-event early return to prevent driver spin."""

        drained = self._inbox.drain()
        if not drained:
            return
        logger.warning(f"agentm session driver: discarded {len(drained)} undrained inbox item(s) after a pre-context-event early return ({reason}; sources={[item.source for item in drained]!r}) to avoid driver spin")
        # ``payload`` may be a JSON-native shape (str / dict / list of
        # TextContent/ImageContent dataclasses). The session manager runs the
        # whole payload through to_jsonable before persisting, so attaching
        # the original payload object is safe; if a producer wedges in a
        # non-serialisable type the persistence path will fail loudly there
        # rather than silently here.
        try:
            self._session_manager.append_custom_entry(
                "inbox.discarded",
                {
                    "reason": reason,
                    "sources": [item.source for item in drained],
                    "items": [
                        {
                            "source": item.source,
                            "payload": item.payload,
                            "dedup_key": item.dedup_key,
                        }
                        for item in drained
                    ],
                },
            )
        except Exception:
            # The warning log already captured the discard; a persistence
            # hiccup here must not crash the driver or hide the original
            # early-return path.
            logger.exception(
                "agentm session driver: failed to persist inbox.discarded entry"
            )

    def _on_message_persisted(self, event: MessagePersistedEvent) -> None:
        leaf = self._session_manager.get_leaf_id()
        if leaf is None:
            self._session_manager.reset_leaf()
        else:
            self._session_manager.branch(leaf)
        self._session_manager.append_message(event.message)
        self._last_round_turn = (event.turn_index, event.turn_id)

    # --- Construction -----------------------------------------------------

    @classmethod
    async def create(cls, config: AgentSessionConfig) -> "AgentSession":
        from agentm.core.runtime.session_factory import create_agent_session

        return await create_agent_session(cls, config)

    # --- Public surface ---------------------------------------------------

    @property
    def bus(self) -> EventBus:
        return self._bus

    @property
    def session_manager(self) -> SessionManager:
        return self._session_manager

    @property
    def inbox(self) -> SessionInbox:
        return self._inbox

    @property
    def resources(self) -> ResourceLoader:
        return self._resources

    @property
    def tools(self) -> list[Tool]:
        return list(self._tools)

    @property
    def model(self) -> Model | None:
        active = self._active_provider_ref.value
        return active.model if active is not None else None

    @property
    def cwd(self) -> str:
        return self._cwd

    @property
    def extension_api(self) -> Any:
        return self._extension_api

    def get_service(self, name: str) -> Any | None:
        return self._services.get(name)

    def set_service(self, name: str, obj: Any) -> None:
        if name in self._services:
            raise KeyError(f"service {name!r} is already registered")
        self._services[name] = obj

    def install_atom(self, name: str, config: dict[str, Any] | None = None) -> None:
        """Mount a builtin atom into this already-created session (sync only)."""
        import inspect

        from agentm.core.runtime.extension import load_extension

        if self._extension_api is None:
            raise RuntimeError(
                f"cannot install_atom({name!r}): session has no extension scope"
            )
        module_path = f"agentm.extensions.builtin.{name}"
        result = load_extension(module_path, self._extension_api, config or {})
        if inspect.isawaitable(result):
            # Avoid leaking an un-awaited coroutine; the host call site (the
            # gateway SessionManager) invokes this synchronously.
            if hasattr(result, "close"):
                result.close()
            raise RuntimeError(
                f"install_atom({name!r}) requires an async install, which the "
                "synchronous host path does not support"
            )

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def root_session_id(self) -> str:
        return self._root_session_id

    # --- prompt -----------------------------------------------------------

    async def prompt(
        self,
        text: str,
        *,
        images: list[ImageContent] | None = None,
        signal: asyncio.Event | None = None,
    ) -> list[AgentMessage]:
        """Push one user turn and await the driver's reply. FIFO-serialized."""

        async with self._prompt_lock:
            text, slash_handled = await self._preprocess_input(text)
            if slash_handled is not None:
                return slash_handled

            waiter = self._subscribe_end_waiter()
            forwarder = self._spawn_signal_forwarder(signal)
            self._inbox.push(
                InboxItem(
                    source="user",
                    payload=self._user_payload(text=text, images=images),
                )
            )
            try:
                await waiter
            finally:
                if forwarder is not None:
                    forwarder.cancel()
            return self._session_manager.get_messages()

    async def resume(
        self,
        *,
        signal: asyncio.Event | None = None,
    ) -> list[AgentMessage]:
        """Run one round on the current context with no new input.

        Resumes a parked session — typically one forked mid-trajectory whose cut
        ends in tool results — by kicking the driver without enqueuing a user
        message. ``loop.run`` then continues naturally from
        ``build_session_context`` (model called on the existing context) until it
        terminates as it otherwise would. This is the no-injection counterpart of
        :meth:`prompt`: the same single-round, FIFO-serialized contract, minus the
        user turn. Used to measure a genuine CONTINUE branch (fork + continue)
        that is apples-to-apples with injected interventions.
        """

        async with self._prompt_lock:
            waiter = self._subscribe_end_waiter()
            forwarder = self._spawn_signal_forwarder(signal)
            # kick (not push): wake the driver to run one round on the current
            # context. The kernel ``context`` handler drains the (empty) inbox
            # and clears the event, so the driver re-parks after this round —
            # honouring SessionInbox.kick's drain contract.
            self._inbox.kick()
            try:
                await waiter
            finally:
                if forwarder is not None:
                    forwarder.cancel()
            return self._session_manager.get_messages()

    async def idle(self, timeout: float | None = None) -> bool:
        """Block until parked + inbox empty + no tracked background work."""

        clock = asyncio.get_running_loop().time
        deadline = None if timeout is None else clock() + timeout

        while not self._closed:
            remaining = None if deadline is None else deadline - clock()
            if remaining is not None and remaining <= 0:
                return False
            if not await self._inbox.wait_no_pending_work(remaining):
                # Bound tripped while a tracked unit was still live.
                return False
            remaining = None if deadline is None else deadline - clock()
            if remaining is not None and remaining <= 0:
                return False
            try:
                if remaining is None:
                    await self._parked.wait()
                else:
                    await asyncio.wait_for(self._parked.wait(), remaining)
            except TimeoutError:
                return False
            if self._closed:
                return True
            # Re-read under the (single-threaded) event loop: if a unit posted
            # between the two waits the inbox is non-empty / work is back, so we
            # loop and let the driver run that round before re-parking.
            if (
                self._inbox.is_empty()
                and not self._inbox.has_pending_work
                and self._parked.is_set()
            ):
                return True
            # Yield so the driver can pick up the pending item / a finishing
            # unit can flip its accounting before we re-evaluate.
            await asyncio.sleep(0)
        return True

    def _spawn_signal_forwarder(
        self, external: asyncio.Event | None
    ) -> asyncio.Task[None] | None:

        if external is None:
            return None

        async def _forward() -> None:
            try:
                await external.wait()
            except asyncio.CancelledError:
                return
            self._signal.set()

        return asyncio.create_task(_forward())

    def _spawn_event_loop_lag_watchdog(self) -> asyncio.Task[None] | None:
        threshold = _env_float(
            _EVENT_LOOP_LAG_WARNING_ENV,
            _DEFAULT_EVENT_LOOP_LAG_WARNING_SECONDS,
        )
        if threshold <= 0:
            return None
        interval = max(
            0.1,
            _env_float(
                _EVENT_LOOP_LAG_INTERVAL_ENV,
                _DEFAULT_EVENT_LOOP_LAG_INTERVAL_SECONDS,
            ),
        )
        return asyncio.create_task(
            self._event_loop_lag_watchdog(
                interval_seconds=interval,
                threshold_seconds=threshold,
            ),
            name=f"agentm-session-watchdog-{self._session_id}",
        )

    async def _event_loop_lag_watchdog(
        self,
        *,
        interval_seconds: float,
        threshold_seconds: float,
    ) -> None:
        loop = asyncio.get_running_loop()
        expected = loop.time() + interval_seconds
        last_warning_at = 0.0
        while not self._closed:
            await asyncio.sleep(interval_seconds)
            now = loop.time()
            lag = max(0.0, now - expected)
            expected = now + interval_seconds
            if lag < threshold_seconds:
                continue
            if now - last_warning_at < _EVENT_LOOP_LAG_RATE_LIMIT_SECONDS:
                continue
            last_warning_at = now
            message = (
                "session event loop lag detected: expected watchdog wake was "
                f"delayed by {lag:.1f}s (threshold {threshold_seconds:.1f}s); "
                "a blocking tool or handler may be starving the core session"
            )
            logger.warning(message)
            try:
                await self._bus.emit(
                    DiagnosticEvent.CHANNEL,
                    DiagnosticEvent(
                        level="warning",
                        source="event_loop_watchdog",
                        message=message,
                    ),
                )
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                logger.exception("failed to emit event loop lag diagnostic")

    # --- tick (resume-without-prompt) -------------------------------------

    async def tick(
        self,
        *,
        signal: asyncio.Event | None = None,
    ) -> list[AgentMessage]:
        """Advance by one decide-cycle without new user input."""

        forwarder = self._spawn_signal_forwarder(signal)
        launched_run = False
        try:
            if not self._inbox.is_empty():
                waiter = self._subscribe_end_waiter()
                launched_run = True
                await waiter
                return self._session_manager.get_messages()

            # Empty inbox: synthetic decide cycle for resume-atoms.
            default_action = Stop(NoPendingInput())
            observation = TurnObservation(
                turn_index=0,
                assistant_message=None,
                tool_outcomes=[],
                default_action=default_action,
            )
            returns = await self._bus.emit(
                DecideTurnActionEvent.CHANNEL,
                DecideTurnActionEvent(observation=observation),
            )
            action = resolve_loop_action(default_action, returns)

            if isinstance(action, Inject):
                # Persist injected messages so the driver's next ``loop.run``
                # picks them up via ``build_session_context``, then kick.
                for injected_msg in action.messages:
                    self._session_manager.append_message(injected_msg)
                waiter = self._subscribe_end_waiter()
                # Asymmetry NB: ``kick()`` here wakes the driver even though
                # the inbox is empty — the injected messages went into the
                # session log directly above, not via inbox push. ``kick`` is
                # purely the wake-up signal in this path, not a "drain me"
                # prompt. See ``SessionInbox.kick`` docstring. (The other
                # tick callsite, ``self._inbox.is_empty() == False`` branch
                # above, kicks for genuine inbox content.)
                self._inbox.kick()
                launched_run = True
                await waiter
                return self._session_manager.get_messages()

            # No injector → no work this tick. Synthesize the matching agent_end
            # so observers see the unchanged-list "nothing happened" outcome.
            messages = self._session_manager.get_messages()
            cause = action.cause if isinstance(action, Stop) else default_action.cause
            self._last_round_turn = None
            await self._bus.emit(
                AgentEndEvent.CHANNEL,
                AgentEndEvent(messages=messages, cause=cause),
            )
            return messages
        finally:
            if forwarder is not None:
                forwarder.cancel()
            # Major-1 review fix (tick leak path): if the synthetic decide-
            # cycle propagated an external signal (the forwarder fired
            # ``_signal.set()``) but no run was launched, ``_signal`` would
            # otherwise carry over and abort the next ``prompt``. Clear it
            # here so the no-run paths leave the session clean. When a run
            # WAS launched, the driver's bottom-of-loop clear runs after
            # the round so we don't fight it.
            if not launched_run and self._signal.is_set():
                self._signal.clear()

    # --- interrupt --------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """Snapshot of session state for gateway/client consumption."""
        phase: str
        if self._closed:
            phase = "closed"
        elif self._in_run:
            phase = "running"
        elif not self._inbox.is_empty():
            phase = "draining"
        else:
            phase = "idle"
        return {
            "phase": phase,
            "session_id": self._session_id,
            "tool_names": [t.name for t in self._tools],
        }

    def send_user_message(self, text: str) -> None:
        """Push a user message into the inbox (mid-turn safe)."""
        self._inbox.push(InboxItem(source="user", payload=text))

    def interrupt(self) -> None:
        """No-op when idle; sets the abort signal when a round is in flight."""

        if self._in_run:
            self._signal.set()

    # --- driver -----------------------------------------------------------

    async def _driver(self) -> None:
        """Block on inbox; run one round; loop. Catches per-round exceptions."""

        # Bind this session's read-before-edit scope on the driver task's
        # context, so concurrent sessions in one process (batch eval) can't
        # clobber each other's read_state. Child tasks copy this context.
        bind_read_state_session(self._session_id)

        while not self._closed:
            # #179: parked = idle, blocked on the next item. SET before the
            # wait so a concurrent ``idle()`` sees rest; CLEARED the instant we
            # wake so ``idle()`` never mistakes an about-to-run round for rest.
            self._parked.set()
            try:
                await self._inbox.wait_nonempty()
            except asyncio.CancelledError:
                self._parked.clear()
                self._fail_end_waiters(asyncio.CancelledError())
                return
            self._parked.clear()
            if self._closed:
                return
            try:
                await self._run_one_round()
            except asyncio.CancelledError as exc:
                self._fail_end_waiters(exc)
                raise
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "agentm session driver: round raised; continuing"
                )
                self._fail_end_waiters(exc)
                # Pre-context-event failure: drain the originating push so
                # the driver parks instead of tight-looping on the same
                # failure. Post-first-turn failures (stream raises mid-turn)
                # have already drained the inbox via the kernel ``context``
                # handler, so this call is a no-op for them. See the helper
                # docstring; sibling site is the veto branch in
                # ``_run_one_round``.
                self._drain_inbox_on_early_return(
                    f"pre-first-turn exception: {type(exc).__name__}"
                )
            # An ``interrupt()`` that fired during the round leaves the
            # signal set; clear it so the next round starts clean. We do
            # NOT clear at the top of ``_run_one_round`` because the kernel
            # signal is also written by the per-call signal forwarder
            # (``_spawn_signal_forwarder``) — a top-of-round clear would
            # silently swallow a forwarder-set abort that fired BEFORE the
            # round started (which legitimately happens when ``sub_agent``'s
            # parent calls ``abort`` before the child's driver got scheduled).
            # The corresponding Major-1 fix for the two leak paths
            # (idle-time ``interrupt`` and ``tick``'s synthetic decide
            # leaking through the forwarder) lives in ``interrupt`` itself
            # (no-op when not in a run) and in ``tick``'s ``finally`` block
            # (clears ``_signal`` after the no-run paths if the forwarder
            # set it during the synthetic decide).
            if self._signal.is_set():
                self._signal.clear()

    def _fail_end_waiters(self, exc: BaseException) -> None:
        """Reject every pending ``prompt``/``tick`` waiter with ``exc``.

        Used when a driver round raises before ``agent_end`` fires — the
        waiters would otherwise hang forever. Drops the waiter list so a
        re-fire (e.g. a subsequent ``agent_end``) does not double-resolve.
        """

        waiters = self._end_waiters
        self._end_waiters = []
        for fut in waiters:
            if not fut.done():
                fut.set_exception(exc)

    async def _run_one_round(self) -> list[AgentMessage]:
        """One driver round: build context, run before_agent_start, run loop.

        Single-ownership: asserts ``_in_run`` is False before flipping it;
        any concurrent ``_loop.run`` caller trips the assertion.
        """

        assert not self._in_run, (
            "concurrent _loop.run detected — the AgentSession driver is the "
            "sole owner of the agent loop; do not call _loop.run from anywhere "
            "else"
        )
        self._in_run = True
        try:
            self._last_round_turn = None
            messages = self._session_manager.build_session_context().messages
            system_prompt = ""
            before_event = BeforeAgentStartEvent(
                messages=messages, system=system_prompt,
            )
            before_returns = await self._bus.emit(
                BeforeAgentStartEvent.CHANNEL, before_event,
            )
            # --- Veto: prefer typed event field, fall back to return-dict ---
            veto_cause = (
                before_event.veto
                if before_event.veto is not None
                else collect_start_veto(before_returns)
            )
            if veto_cause is not None:
                await self._bus.emit(
                    AgentEndEvent.CHANNEL,
                    AgentEndEvent(messages=messages, cause=veto_cause),
                )
                self._drain_inbox_on_early_return(
                    f"before_agent_start veto: {type(veto_cause).__name__}"
                )
                return messages
            # --- System prompt: prefer return-dict (back-compat), fall back
            #     to event mutation ---
            replacement_system = collect_system_replacement(before_returns)
            if replacement_system is not None:
                system_prompt = replacement_system
            elif before_event.system:
                system_prompt = before_event.system

            # NB: the kernel ``run`` itself emits ``agent_start`` at entry
            # (loop.py:412); we don't duplicate it here. The kernel also fires
            # ``context`` at the top of each turn — that drains the inbox
            # into the live message list. ``agent_end`` fires from inside the
            # kernel on every termination path, waking our prompt/tick
            # waiters.
            return await self._loop.run(
                messages=messages,
                model=self._require_model(),
                tools=self._tools,
                system=system_prompt,
                signal=self._signal,
            )
        finally:
            self._in_run = False

    # --- waiter plumbing --------------------------------------------------

    def _subscribe_end_waiter(self) -> asyncio.Future[None]:
        """Register a one-shot future that fires on the next ``agent_end``.

        The handler set in ``__init__`` fulfills + drops every waiter on each
        ``agent_end`` event, so a fresh subscription is required per round.
        """

        loop = asyncio.get_running_loop()
        fut: asyncio.Future[None] = loop.create_future()
        self._end_waiters.append(fut)
        return fut

    def _on_agent_end_commit_boundary(self, event: AgentEndEvent) -> None:
        """Persist a ``turn_committed`` marker delimiting a consistent point.

        Entries persist incrementally mid-turn, but a hard crash (process
        kill, OOM) never reaches this handler — only a clean termination does.
        So every marker sits at a consistent boundary, and a crash leaves its
        half-turn unmarked for ``_truncate_to_last_boundary`` to shed on the
        next cold load. Skipped when the leaf is already a marker (repeated
        no-op ticks / vetoes add no new content), so markers never stack.
        """

        leaf = self._session_manager.get_leaf_entry()
        if leaf is not None and leaf.type == ENTRY_TYPE_TURN_COMMITTED:
            return
        cause = type(event.cause).__name__ if event.cause is not None else "unknown"
        payload: dict[str, Any] = {"cause": cause}
        if self._last_round_turn is not None:
            turn_index, turn_id = self._last_round_turn
            payload["turn_index"] = turn_index
            payload["turn_id"] = turn_id
        try:
            self._session_manager.append_custom_entry(
                ENTRY_TYPE_TURN_COMMITTED, payload
            )
            self._last_round_turn = None
        except Exception:
            # A persistence hiccup here must never crash the driver round or
            # block the prompt waiter the next handler resolves.
            logger.exception("failed to append turn_committed boundary marker")

    def _on_agent_end_wake_waiters(self, event: AgentEndEvent) -> None:
        # #177: clear the pending backgrounded-terminate ONLY when the loop
        # actually stopped on that exact cause (identity match against the
        # event's cause). If the round instead ended on something else — a
        # ModelEndTurn while ``sub_agent``'s floor kept the loop alive with an
        # Inject that overrode our Stop — the flag MUST survive so the next
        # boundary re-asserts it. This is the invariant the keep-alive floor
        # relies on: a backgrounded terminate is not lost across turns where a
        # running child keeps injecting.
        if (
            self._pending_terminate is not None
            and event.cause is self._pending_terminate
        ):
            self._pending_terminate = None
        waiters = self._end_waiters
        self._end_waiters = []
        for fut in waiters:
            if not fut.done():
                fut.set_result(None)

    # --- prompt helpers ---------------------------------------------------

    async def _preprocess_input(
        self, text: str
    ) -> tuple[str, list[AgentMessage] | None]:
        event = InputEvent(text=text)
        returns = await self._bus.emit(InputEvent.CHANNEL, event)
        # --- Typed event field path (preferred) ---
        if event.handled and event.handled_messages is not None:
            return text, event.handled_messages
        # --- Return-dict fallback (deprecated) ---
        for value in returns:
            if isinstance(value, dict) and value.get("handled") is True:
                messages = value.get("messages")
                if isinstance(messages, list):
                    return text, messages
        # --- Text rewrite (mutation path, always via event field) ---
        return (event.text if isinstance(event.text, str) else text), None

    def _user_payload(
        self, *, text: str, images: list[ImageContent] | None
    ) -> list[TextContent | ImageContent]:
        """Build the content-block list a ``source="user"`` inbox item carries.

        ``render_item`` turns this into a :class:`UserMessage`. Mirrors the old
        ``_build_user_message`` shape: text block (if any) then any images, so
        an image-only prompt still produces a valid user message and an empty
        text yields an empty-content message exactly as before.
        """

        content: list[TextContent | ImageContent] = []
        if text:
            content.append(TextContent(type="text", text=text))
        if images:
            content.extend(images)
        return content

    def _append_message(self, msg: AgentMessage) -> Any:
        return self._session_manager.append_message(msg)

    def _require_model(self) -> Model:
        model = self.model
        if model is None:
            raise RuntimeError("no active provider model is available")
        return model

    # --- Lifecycle --------------------------------------------------------

    async def shutdown(self, *, strict: bool = False) -> None:
        """Idempotent teardown: stop driver, emit session_shutdown, clear bus."""

        if self._closed:
            return
        self._closed = True
        self._signal.set()
        self._inbox.kick()
        driver_exc: BaseException | None = None
        try:
            await asyncio.wait_for(
                self._driver_task, timeout=_DRIVER_SHUTDOWN_GRACE_SECONDS
            )
        except asyncio.TimeoutError:
            self._driver_task.cancel()
            try:
                await self._driver_task
            except (asyncio.CancelledError, Exception) as exc:  # noqa: BLE001
                # We cancelled the driver after a shutdown-grace timeout, so a
                # CancelledError is expected; any other error here is during
                # teardown and non-fatal.
                logger.debug(
                    "agentm session driver: post-cancel await raised "
                    "{} during shutdown; continuing",
                    type(exc).__name__,
                )
        except asyncio.CancelledError:
            # Shutdown await itself was cancelled — expected during teardown.
            logger.debug("agentm session driver: shutdown await cancelled; continuing")
        except Exception as exc:  # noqa: BLE001
            logger.exception(f"agentm session driver: shutdown await raised ({type(exc).__name__}); continuing")
            driver_exc = exc

        if self._watchdog_task is not None:
            self._watchdog_task.cancel()
            await asyncio.gather(self._watchdog_task, return_exceptions=True)

        for fut in self._end_waiters:
            if not fut.done():
                fut.cancel()
        self._end_waiters = []

        await self._bus.emit(
            SessionShutdownEvent.CHANNEL, SessionShutdownEvent(cwd=self._cwd)
        )

        if self._parent_bus is not None:
            await self._parent_bus.emit(
                ChildSessionEndEvent.CHANNEL,
                ChildSessionEndEvent(
                    child_session_id=self._session_id,
                    parent_session_id=self._parent_session_id or "unknown",
                    final_message_count=len(self._session_manager.get_messages()),
                    error=None,
                ),
            )

        self._reloader.shutdown()
        self._bus.clear()

        try:
            trace_file = self._session_manager.session_file
            if trace_file is not None and trace_file.is_file():
                from agentm.core.runtime.catalog.indexer import index_trace

                index_trace(trace_file)
        except Exception as exc:
            logger.warning(f"agentm catalog indexer post-shutdown failed: {exc!r}")

        # Tear down the eval sandbox dir if this session created one for
        # ``atom_source_overrides``. Best-effort: any failure is logged but
        # never raised (shutdown must not error on stale FS state).
        if self._eval_sandbox is not None:
            try:
                shutil.rmtree(self._eval_sandbox, ignore_errors=True)
            except Exception as exc:  # noqa: BLE001 - shutdown best-effort
                logger.warning(f"failed to clean eval sandbox {self._eval_sandbox}: {exc!r}")

        # B6 boundary-review fix: in strict mode, propagate the captured
        # driver exception AFTER cleanup finished. The bus is drained, the
        # parent is notified, the catalog is indexed — the host gets a
        # clean half-shutdown PLUS the failure to act on.
        if strict and driver_exc is not None:
            raise driver_exc

    # --- Helpers ----------------------------------------------------------



__all__ = [
    "AgentSession",
    "AgentSessionConfig",
]
