"""AgentSession orchestrator: the fat-but-thin v2 façade.

Implements §4 (AgentSession) of ``.claude/designs/extension-as-scenario.md``.
The session holds references to every subsystem (event bus, session manager,
resource loader, registries) and wires events. It runs no business logic:
each "feature" is an extension that registers handlers on the bus.

Lifecycle (``AgentSession.create`` → ``prompt`` → ``shutdown``):

1. Build :class:`EventBus`, :class:`SessionManager`, :class:`ResourceLoader`,
   internal ``_ExtensionAPIImpl``.
2. Load every extension in order; await coroutine returns. The provider
   extension is loaded last so the picked-up provider reflects any earlier
   replacement attempts (last registration wins).
3. Append ``initial_messages`` (if any) into the session manager.

On ``prompt(text)``:

1. Build a :class:`UserMessage`, append it as a session entry.
2. Assemble the system prompt from context files + skill descriptions
   (placeholder; full skill-body expansion comes in a later phase).
3. Emit ``before_agent_start``; handlers may return ``{"system": "..."}``
   replacing the system prompt (last non-None wins, mirroring the kernel
   ``_collect_replacement`` convention).
4. Run ``AgentLoop.run``.
5. Append every new assistant + tool_result message as session entries.
6. Return the full updated message list.

Hard rule: this module imports only stdlib + ``agentm.core.abi`` + the
three sibling v2 modules.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path
from typing import Any

from agentm.core.abi import (
    AgentEndEvent,
    AgentMessage,
    DecideTurnActionEvent,
    EventBus,
    ImageContent,
    Inject,
    Model,
    NoPendingInput,
    Step,
    Stop,
    TextContent,
    Tool,
    TurnObservation,
)
from agentm.core.abi.events import (
    BeforeAgentStartEvent,
    ChildSessionEndEvent,
    MessagePersistedEvent,
    SessionShutdownEvent,
)
from agentm.core.abi.events import ContextEvent
from agentm.core.abi.loop import resolve_loop_action
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
from agentm.core.runtime.session_runtime import SessionRuntime
from agentm.core.runtime.session_manager import SessionManager

logger = logging.getLogger(__name__)

# Grace window for ``AgentSession.shutdown`` to await the persistent driver
# task before forcing cancellation. Matches the cooperative-shutdown shape
# the ``background_exec`` / ``sub_agent`` atoms use for their own task
# registries: ask cooperatively (``_closed`` + ``interrupt``), wait briefly,
# cancel if the task doesn't observe.
_DRIVER_SHUTDOWN_GRACE_SECONDS = 5.0


# --- Config -----------------------------------------------------------------
# ``AgentSessionConfig`` lives in ``core.abi.session_config`` so extensions
# (which cannot import this module per §11.4.5) can still construct one for
# ``api.spawn_child_session``. Re-exported here for ergonomic access.

from agentm.core.abi.session_config import (  # noqa: E402
    AgentSessionConfig,
)


# --- AgentSession -----------------------------------------------------------


class AgentSession:
    """Top-level v2 session façade. Construct via :meth:`create`."""

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
        self._active_provider_box = runtime.active_provider_box
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
        # One-shot waiters subscribed by ``prompt``/``tick`` for the next
        # ``agent_end``. The handler set below fulfills each future once and
        # drops them, so a follow-up prompt subscribes a fresh waiter.
        self._end_waiters: list[asyncio.Future[None]] = []

        # Real-time persistence: the loop emits MessagePersistedEvent for
        # every durable addition (assistant turn / tool_result / injected
        # message). Routing each event through the SessionManager here —
        # rather than diffing the loop's return value at the end — means a
        # mid-loop kill still leaves every completed turn on disk.
        self._bus.on(
            MessagePersistedEvent.CHANNEL, self._on_message_persisted
        )

        # SessionInbox spine (see .claude/designs/session-inbox.md, step 1).
        # Two runtime-owned handlers, registered as closures over the inbox +
        # session_manager — NOT atoms (the inbox is substrate). They reuse the
        # existing loop seams without modifying the kernel ``AgentLoop``:
        #
        #  * ``context`` (turn start, loop.py:466): the single message-entry
        #    point. Drains the inbox and appends every rendered message to the
        #    turn's message list in place, so the LLM sees pending input on the
        #    very next turn.
        #  * ``decide_turn_action`` (turn end): keep-alive floor. If the inbox
        #    is non-empty, return ``Step()`` so the loop runs another turn
        #    whose ``context`` will drain it. This generalizes sub_agent's
        #    lifecycle floor; ``final=True`` causes still hard-win via the
        #    resolve lattice (loop.py:301).
        self._bus.on(ContextEvent.CHANNEL, self._on_context_drain_inbox)
        self._bus.on(
            DecideTurnActionEvent.CHANNEL, self._on_decide_inbox_keep_alive
        )
        # ``prompt``/``tick`` wait on the next ``agent_end`` to know the
        # driver finished their round. Hooked before the driver starts so a
        # racy push-then-await never misses the wake.
        self._bus.on(AgentEndEvent.CHANNEL, self._on_agent_end_wake_waiters)

        # Step 5a: start the persistent driver task. We are constructed inside
        # the running event loop (the async factory awaits ``create``), so
        # ``create_task`` is safe here. The driver runs forever until
        # ``shutdown`` flips ``_closed`` and kicks the inbox.
        #
        # Footgun guard: anyone bypassing the ``AgentSession.create`` factory
        # to construct synchronously will hit ``RuntimeError: no running
        # event loop`` from ``create_task`` below. Surface that contract
        # explicitly so the failure says WHY rather than dumping a stdlib
        # backtrace at the caller.
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
    ) -> Step | None:
        """Keep the loop alive while the inbox holds undrained items.

        Returning ``Step()`` defers to the next turn, whose ``context``
        handler drains the inbox. A ``final=True`` default (budget / signal /
        max_turns) overrides this via ``resolve_loop_action`` (loop.py:301),
        so a hard ceiling stays hard.
        """

        del event
        if not self._inbox.is_empty():
            return Step()
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

    def _drain_inbox_on_early_return(self, reason: str) -> None:
        """Drain (discard) the inbox after a pre-context-event early return.

        The kernel's ``context`` event only fires from inside ``_loop.run``;
        any path that leaves ``_run_one_round`` BEFORE the loop is entered
        (a sticky veto from ``before_agent_start``, an exception during
        context build or ``before_agent_start`` itself, etc.) leaves the
        originating push queued — ``_nonempty`` stays set, ``wait_nonempty``
        returns immediately on the next driver iteration, and the driver
        tight-loops on the same failure (burning CPU even though the
        prompt waiter already saw an ``agent_end``). Every such site
        MUST call this helper after emitting ``agent_end`` / failing the
        waiter so the next iteration parks until a fresh push arrives.

        Discard semantics match the existing exception path: the user's
        prompt waiter has already been resolved (cleanly via ``agent_end``
        or with the exception via ``_fail_end_waiters``), so silently
        re-running the vetoed/failed work on the queued message would be
        wrong. If the user wants to try again they re-push.
        """

        drained = self._inbox.drain()
        if drained:
            logger.warning(
                "agentm session driver: discarded %d undrained inbox "
                "item(s) after a pre-context-event early return (%s; "
                "sources=%r) to avoid driver spin",
                len(drained),
                reason,
                [item.source for item in drained],
            )

    def _on_message_persisted(self, event: MessagePersistedEvent) -> None:
        leaf = self._session_manager.get_leaf_id()
        if leaf is None:
            self._session_manager.reset_leaf()
        else:
            self._session_manager.branch(leaf)
        self._session_manager.append_message(event.message)

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
        """Read-only handle to the session's :class:`SessionInbox`.

        Exposed so a test or embedding host can push items without reaching
        into the private ``_inbox`` attribute. Atoms should keep using
        :meth:`ExtensionAPI.post_inbox` — same backing inbox, with the
        scope's ``ExtensionStaleError`` guard.
        """
        return self._inbox

    @property
    def resources(self) -> ResourceLoader:
        return self._resources

    @property
    def tools(self) -> list[Tool]:
        return list(self._tools)

    @property
    def model(self) -> Model | None:
        active = self._active_provider_box["value"]
        return active.model if active is not None else None

    @property
    def cwd(self) -> str:
        return self._cwd

    def get_service(self, name: str) -> Any | None:
        return self._services.get(name)

    def set_service(self, name: str, obj: Any) -> None:
        """Publish a host-supplied service into the session's registry.

        Mirrors :meth:`ExtensionAPI.set_service` but is callable from the
        embedding process (the runtime host), not just from inside an
        atom. Used by ``agentm-worker`` to inject a ``peer_messaging``
        handle so the optional ``tool_peer_send`` atom can call out to
        the gateway. Refuses to clobber an existing entry — keep service
        ownership unambiguous (matches the atom-side contract).
        """
        if name in self._services:
            raise KeyError(f"service {name!r} is already registered")
        self._services[name] = obj

    @property
    def tool_renderers(self) -> dict[str, Any]:
        return {
            name.removeprefix("tool:"): renderer
            for name, renderer in self._renderers.items()
            if name.startswith("tool:")
        }

    def find_tool(self, name: str) -> Tool | None:
        for tool in self._tools:
            if tool.name == name:
                return tool
        return None

    @property
    def session_id(self) -> str:
        """Stable random id assigned at ``create``. Appears in
        :class:`ChildSessionStartEvent` / :class:`ChildSessionEndEvent`
        payloads when this session is a child of another. Equals the
        session-root OTel ``span_id`` — see :attr:`root_session_id` for
        the OTel ``trace_id`` shared across the whole agent tree."""
        return self._session_id

    @property
    def root_session_id(self) -> str:
        """OTel ``trace_id`` shared by this session and every transitive
        child. For a root session the substrate generates a fresh 32-hex
        uuid; spawned children inherit it verbatim so the entire agent
        tree shows up as a single trace in any OTel-compatible store."""
        return self._root_session_id

    # --- prompt -----------------------------------------------------------

    async def prompt(
        self,
        text: str,
        *,
        images: list[ImageContent] | None = None,
        signal: asyncio.Event | None = None,
    ) -> list[AgentMessage]:
        """Push one user turn onto the inbox and await the driver's reply.

        Sugar over the session inbox + persistent driver (see
        session-inbox.md, step-5 §"Persistent driver + prompt-as-sugar"):
        slash-commands fast-path first; otherwise push a ``source="user"``
        inbox item, subscribe a one-shot waiter for the next ``agent_end``,
        block until the driver finishes that round, and return the live
        message list. The driver is the SOLE caller of ``_loop.run`` — this
        method does no LLM work itself.

        Optional ``signal``: if the caller passes an :class:`asyncio.Event`,
        it's one-directionally forwarded into the session's own ``_signal``
        (preempting the in-flight run with :class:`SignalAborted`, same shape
        :meth:`interrupt` uses). This keeps the legacy per-call abort path
        (e.g. ``sub_agent``'s child-task ``abort_signal``) working unchanged
        without giving callers the ability to clobber the driver-owned
        kernel signal directly.
        """

        # 0. Slash-command dispatch / input preprocessing. Code commands win;
        # otherwise ``input`` handlers may rewrite slash-prefixed text before
        # it falls through to the agent loop.
        text, slash_handled = await self._preprocess_input(text)
        if slash_handled is not None:
            return slash_handled

        # 1. Subscribe a waiter BEFORE the push: if the driver was already
        # awake from a prior wake the run could finish between push and
        # subscribe, and we'd hang on a fulfilled event we never registered
        # for.
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

    def _spawn_signal_forwarder(
        self, external: asyncio.Event | None
    ) -> asyncio.Task[None] | None:
        """One-way bridge from a caller-supplied abort event into ``_signal``.

        Mirrors the ``background_exec`` forwarder pattern: setting the
        external event sets the session's ``_signal`` (the driver-owned
        kernel signal), but setting ``_signal`` directly never touches the
        external one. The bridge is cancelled when ``prompt`` returns so it
        never outlives its prompt.
        """

        if external is None:
            return None

        async def _forward() -> None:
            try:
                await external.wait()
            except asyncio.CancelledError:
                return
            self._signal.set()

        return asyncio.create_task(_forward())

    # --- tick (resume-without-prompt) -------------------------------------

    async def tick(
        self,
        *,
        signal: asyncio.Event | None = None,
    ) -> list[AgentMessage]:
        """Advance the session by one decide-cycle without new user input.

        Two paths preserve today's tick contract:

        * **Inbox non-empty** (e.g. an atom pre-pushed a ``send_user_message``
          /``post_inbox`` item before tick) — wake the driver and wait for
          the next ``agent_end``. Same path ``prompt`` takes.
        * **Inbox empty** — fire a synthetic :class:`DecideTurnActionEvent`
          so resume-atoms (e.g. ``llmharness.replay.reminder_seed``) get
          their one chance to ``Inject``. If a handler injects, persist the
          messages into the session log and kick the driver — the next
          ``_loop.run`` rebuilds context from the log and sees them. If no
          handler injects, emit a synthetic ``agent_end(NoPendingInput)``
          and return — no LLM call, unchanged message list.

        Optional ``signal``: same one-way bridge :meth:`prompt` honours.
        """

        forwarder = self._spawn_signal_forwarder(signal)
        launched_run = False
        try:
            if not self._inbox.is_empty():
                waiter = self._subscribe_end_waiter()
                self._inbox.kick()  # no new item, but wake the driver to drain.
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

    def interrupt(self) -> None:
        """Preempt the in-flight ``_loop.run`` round (Claude-Code style).

        Sets the kernel abort event the driver passed into ``_loop.run`` —
        the loop terminates with ``SignalAborted`` (``final=True``, so no
        floor / no Inject can override). The driver's bottom-of-loop clear
        wipes ``_signal`` after the round returns; the next push drives a
        fresh run with full conversation context preserved (the session
        log is untouched).

        Major-1 review fix: this is a NO-OP when no round is in flight
        (``_in_run`` is False). Previously an idle-time ``interrupt()``
        latched ``_signal``; the next ``prompt`` push woke the driver, the
        kernel's per-turn signal check (``loop.py:440``) fired
        ``SignalAborted`` before any real work happened. With the no-op
        guard, idle interrupts are silently dropped — which is the right
        semantic ("interrupt what's running" — nothing is running, so
        nothing to interrupt). Within one round multiple calls are
        idempotent (``asyncio.Event.set`` on an already-set event is a
        no-op).
        """

        if self._in_run:
            self._signal.set()

    # --- driver -----------------------------------------------------------

    async def _driver(self) -> None:
        """Persistent always-on owner of ``_loop.run``.

        Body: block on ``inbox.wait_nonempty`` until either an item is
        pushed (or ``kick``-ed); run one round; loop. Catches every
        exception per-round so a transient ``run`` failure (provider
        glitch, atom bug) doesn't kill the driver — the next push still
        drives a new run. The only clean exit is ``shutdown`` flipping
        ``_closed`` and kicking the inbox.

        Any in-flight ``prompt`` / ``tick`` waiters that were attached
        BEFORE the round's exception are resolved with that exception, so
        the caller sees the failure rather than hanging on a waiter no
        ``agent_end`` will ever fulfill. A hard task cancellation
        (CancelledError) propagates to those waiters AND aborts the
        driver — the rest of the session is being torn down anyway.
        """

        while not self._closed:
            try:
                await self._inbox.wait_nonempty()
            except asyncio.CancelledError:
                self._fail_end_waiters(asyncio.CancelledError())
                return
            if self._closed:
                return
            try:
                await self._run_one_round()
            except asyncio.CancelledError as exc:
                # Hard cancellation: surface it to any in-flight waiter so
                # ``prompt`` raises rather than hangs, then propagate so the
                # driver task itself enters CANCELLED.
                self._fail_end_waiters(exc)
                raise
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "agentm session driver: round raised; continuing"
                )
                # Wake any in-flight waiter with the exception — they're
                # never going to see ``agent_end`` (the run died before
                # emitting it).
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
            messages = self._session_manager.build_session_context().messages
            system_prompt = self._build_system_prompt()
            before_returns = await self._bus.emit(
                BeforeAgentStartEvent.CHANNEL,
                BeforeAgentStartEvent(messages=messages, system=system_prompt),
            )
            veto_cause = collect_start_veto(before_returns)
            if veto_cause is not None:
                await self._bus.emit(
                    AgentEndEvent.CHANNEL,
                    AgentEndEvent(messages=messages, cause=veto_cause),
                )
                # Pre-context-event early return: drain the originating
                # push so the driver parks instead of tight-looping on the
                # still-queued item (sibling of the exception-path drain
                # in ``_driver``; both call ``_drain_inbox_on_early_return``).
                self._drain_inbox_on_early_return(
                    f"before_agent_start veto: {type(veto_cause).__name__}"
                )
                return messages
            replacement_system = collect_system_replacement(before_returns)
            if replacement_system is not None:
                system_prompt = replacement_system

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

    def _on_agent_end_wake_waiters(self, _event: AgentEndEvent) -> None:
        waiters = self._end_waiters
        self._end_waiters = []
        for fut in waiters:
            if not fut.done():
                fut.set_result(None)

    # --- prompt helpers ---------------------------------------------------

    async def _preprocess_input(
        self, text: str
    ) -> tuple[str, list[AgentMessage] | None]:
        event: dict[str, Any] = {"text": text}
        returns = await self._bus.emit("input", event)
        for value in returns:
            if isinstance(value, dict) and value.get("handled") is True:
                messages = value.get("messages")
                if isinstance(messages, list):
                    return text, messages
        handled_messages = event.get("handled_messages")
        if isinstance(handled_messages, list):
            return text, handled_messages
        new_text = event.get("text")
        return (new_text if isinstance(new_text, str) else text), None

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

    async def shutdown(self) -> None:
        """Signal extensions and clear handlers.

        Order matters:
        1. Flip ``_closed`` and kick the inbox so the driver's
           ``wait_nonempty`` returns and the loop exits cleanly.
        2. ``interrupt()`` any in-flight run so a long-running tool doesn't
           hold the driver hostage past the grace window.
        3. ``await`` the driver task under a bounded grace; cancel + gather
           if it overruns (same shape as the atom-level shutdown drains in
           ``background_exec`` / ``sub_agent``).
        4. Emit ``session_shutdown`` (atoms see the bus alive) → notify
           parent → clear bus → catalog index.

        Idempotent: a second ``shutdown`` call is a no-op (the driver task is
        already terminal and ``_closed`` is sticky).
        """

        if self._closed:
            return
        self._closed = True
        self._signal.set()
        self._inbox.kick()
        try:
            await asyncio.wait_for(
                self._driver_task, timeout=_DRIVER_SHUTDOWN_GRACE_SECONDS
            )
        except asyncio.TimeoutError:
            self._driver_task.cancel()
            try:
                await self._driver_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        except asyncio.CancelledError:
            pass
        except Exception as exc:  # noqa: BLE001
            # Log the exception class explicitly so an embedding host can
            # tell "clean shutdown" from "driver crashed during shutdown"
            # from the log line alone (``logger.exception`` still attaches
            # the traceback). A future ``shutdown(strict=True)`` could
            # re-raise after the bus is drained for hosts that want to
            # propagate the failure; tracked as a follow-up.
            logger.exception(
                "agentm session driver: shutdown await raised (%s); continuing",
                type(exc).__name__,
            )

        # Any prompt/tick still parked on a waiter will never see its
        # agent_end (the driver is gone). Cancel them so the caller surfaces
        # a clean CancelledError rather than hanging forever.
        for fut in self._end_waiters:
            if not fut.done():
                fut.cancel()
        self._end_waiters = []

        await self._bus.emit(
            SessionShutdownEvent.CHANNEL, SessionShutdownEvent(cwd=self._cwd)
        )

        # Notify the parent (if any) BEFORE clearing handlers so that an
        # extension subscribed on the parent bus can still observe the end
        # event with an accurate message count.
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
            from agentm.core.runtime.catalog.indexer import index_trace

            trace_path = (
                Path(self._cwd)
                / ".agentm"
                / "observability"
                / f"{self._session_id}.jsonl"
            )
            if trace_path.is_file():
                index_trace(trace_path)
        except Exception as exc:
            logger.warning("agentm catalog indexer post-shutdown failed: %r", exc)

        # Tear down the eval sandbox dir if this session created one for
        # ``atom_source_overrides``. Best-effort: any failure is logged but
        # never raised (shutdown must not error on stale FS state).
        if self._eval_sandbox is not None:
            try:
                shutil.rmtree(self._eval_sandbox, ignore_errors=True)
            except Exception as exc:  # noqa: BLE001 - shutdown best-effort
                logger.warning(
                    "failed to clean eval sandbox %s: %r",
                    self._eval_sandbox,
                    exc,
                )

    # --- Helpers ----------------------------------------------------------

    def _build_system_prompt(self) -> str:
        """Concatenate context files + skill names/descriptions.

        Placeholder implementation (per design §4): full skill-body expansion
        (lazy injection on invocation) is deferred to a later phase. For v0
        we surface skill descriptions in the system prompt so the model knows
        they exist.
        """

        parts: list[str] = []
        for cf in self._resources.get_context_files():
            parts.append(cf.body.rstrip())

        skills = self._resources.get_skills()
        if skills:
            parts.append("# Available skills")
            for skill in skills:
                parts.append(f"- {skill.name}: {skill.description}")

        return "\n\n".join(parts)


__all__ = [
    "AgentSession",
    "AgentSessionConfig",
]
