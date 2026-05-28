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
    AgentStartEvent,
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
        self, *, messages: list[AgentMessage] | None
    ) -> int:
        """Shared drain helper: pop every inbox item, render it, persist it via
        the session manager, and (when ``messages`` is given) splice it into
        that live list. Returns the number of items drained.

        Persistence mirrors the loop's injected-message contract (loop.py:609,
        session.py:396): every message that lands in the live context must also
        reach the session log, or a mid-run kill would lose it. Used by both
        the per-turn ``context`` handler (with the turn's ``messages`` list) and
        the ``_drive`` entry (``messages=None`` — the live context list does not
        exist yet; it is built from the log right after).
        """

        count = 0
        for item in self._inbox.drain():
            rendered = render_item(item)
            self._session_manager.append_message(rendered)
            if messages is not None:
                messages.append(rendered)
            count += 1
        return count

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
        """Run one user-prompt → assistant-final-answer turn cycle.

        Sugar over the session inbox (see session-inbox.md, decision §"one
        entry, one driver"): the caller's text is pushed as a
        ``source="user"`` inbox item, then :meth:`_drive` runs the loop.
        Dispatches slash-commands first; drives the kernel loop; appends every
        new assistant + tool_result message; returns the full active-branch
        message list. Stays a mechanical dispatcher per design §4.
        """

        # 0. Slash-command dispatch / input preprocessing. Code commands win;
        # otherwise ``input`` handlers may rewrite slash-prefixed text before
        # it falls through to the agent loop.
        text, slash_handled = await self._preprocess_input(text)
        if slash_handled is not None:
            return slash_handled

        # 1. Push the caller's message onto the inbox. ``_drive`` drains it at
        # entry (before ``build_session_context``) so the originating message
        # lands before the first LLM call — no first-turn delay. Image-only
        # prompts still produce a UserMessage (matching the old behaviour
        # where an empty text yielded an empty-content user message).
        self._inbox.push(
            InboxItem(
                source="user",
                payload=self._user_payload(text=text, images=images),
            )
        )
        return await self._drive(signal=signal)

    # --- tick (resume-without-prompt) -------------------------------------

    async def tick(
        self,
        *,
        signal: asyncio.Event | None = None,
    ) -> list[AgentMessage]:
        """Advance the session by one decide-cycle without new user input.

        Sugar over :meth:`_drive` with no inbox push. Used by the CLI when
        ``agentm --resume <sid>`` is invoked with no positional prompt:
        harness-injected messages (e.g. the prefix-replay ``reminder_seed``
        atom) are the source of the first message, and the kernel must give
        extensions a chance to ``Inject`` BEFORE any LLM call.

        ``_drive`` drains whatever the inbox already holds (e.g. items pushed
        via :meth:`ExtensionAPI.send_user_message` before the tick) and fires a
        synthetic :class:`DecideTurnActionEvent` so resume-atoms can ``Inject``.
        Empty inbox + no injector ⇒ ``AgentEndEvent(NoPendingInput())`` and the
        unchanged message list (today's tick contract). A pre-tick inbox item is
        drained and persisted even on the no-injector exit path — by design (the
        inbox is the documented out-of-band entry point), matching ``prompt``.
        """

        return await self._drive(signal=signal)

    # --- shared driver ----------------------------------------------------

    async def _drive(
        self,
        *,
        signal: asyncio.Event | None = None,
    ) -> list[AgentMessage]:
        """Shared entry for ``prompt``/``tick``: run the loop to idle.

        1. Drain the inbox once at entry (before ``build_session_context``) so
           any originating / queued message lands before the first LLM call.
        2. Build the live context + system prompt; run ``before_agent_start``
           (veto / system replacement) symmetric with the pre-inbox path.
        3. Fire a synthetic :class:`DecideTurnActionEvent` (default
           ``Stop(NoPendingInput())``) so resume-atoms may ``Inject`` before
           any LLM call; the runtime keep-alive floor is a no-op here (the
           inbox was just drained).
        4. Run :meth:`AgentLoop.run` if the entry-drain produced any message OR
           a handler injected/stepped; otherwise emit ``AgentEndEvent`` with the
           resolved no-work cause and return the unchanged list.

        Persistence of drained / injected messages happens in
        :meth:`_drain_inbox_and_persist` and the inline inject splice; the
        kernel loop persists its own assistant / tool_result messages in real
        time via ``_on_message_persisted``.
        """

        # 1. Drain the inbox into the session log up front so the originating
        # message is part of the context the LLM first sees (no live-context
        # list yet — it is built from the log at step 2).
        drained_any = self._drain_inbox_and_persist(messages=None) > 0

        # 2. Build the live context + system prompt and run before_agent_start.
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
            return messages
        replacement_system = collect_system_replacement(before_returns)
        if replacement_system is not None:
            system_prompt = replacement_system

        await self._bus.emit(
            AgentStartEvent.CHANNEL, AgentStartEvent(messages=messages)
        )

        # 3. Synthetic decide-cycle: resume-atoms may ``Inject`` before any
        # LLM call. The inbox keep-alive floor is a no-op (drained at step 1).
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
            # Persist + splice handler-injected messages, then run.
            for injected_msg in action.messages:
                self._session_manager.append_message(injected_msg)
            messages.extend(action.messages)
        elif not drained_any:
            # No entry-drained content and no injector → no work this drive.
            # The default ``NoPendingInput`` cause is the literal observers
            # match on (a non-Inject Stop override wins via the lattice).
            cause = action.cause if isinstance(action, Stop) else default_action.cause
            await self._bus.emit(
                AgentEndEvent.CHANNEL,
                AgentEndEvent(messages=messages, cause=cause),
            )
            return messages

        # 4. Run the loop. Every new assistant / tool_result / injected message
        # is persisted in real time by ``_on_message_persisted``; the loop's
        # return value is the final live list.
        return await self._loop.run(
            messages=messages,
            model=self._require_model(),
            tools=self._tools,
            system=system_prompt,
            signal=signal,
        )

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

        Phase 1 emits a single ``session_shutdown`` event then drops every
        subscription. Extensions that need cleanup hook ``on('session_shutdown')``.
        """

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
