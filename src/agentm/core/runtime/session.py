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
    Stop,
    TextContent,
    Tool,
    TurnObservation,
    UserMessage,
)
from agentm.core.abi.events import (
    BeforeAgentStartEvent,
    ChildSessionEndEvent,
    SessionShutdownEvent,
)
from agentm.core.abi.loop import resolve_loop_action
from agentm.core.runtime.resource_loader import ResourceLoader
from agentm.core.runtime.session_helpers import (
    collect_start_veto,
    collect_system_replacement,
    now,
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
        self._pending_user_messages = runtime.pending_user_messages
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

        Drains queued ``send_user_message`` content, dispatches slash-commands,
        budget-gate-checks, drives the kernel loop, appends every new
        assistant + tool_result message, and returns the full active-branch
        message list. Stays a mechanical dispatcher per design §4.
        """

        # 0. Slash-command dispatch / input preprocessing. Code commands win;
        # otherwise ``input`` handlers may rewrite slash-prefixed text before
        # it falls through to the agent loop.
        text, slash_handled = await self._preprocess_input(text)
        if slash_handled is not None:
            return slash_handled

        # 1. Drain ``send_user_message`` queue (FIFO) into user-message
        # entries in the session. This is how ``sub_agent.inject_instruction``
        # and similar extensions push content into the next turn.
        await self._drain_pending_user_messages()

        # 3. Build the caller's user message, after any drained queue items
        # so they appear as turn-prefix context.
        user_msg = self._build_user_message(text=text, images=images)
        entry = self._append_message(user_msg)

        # 4. Gather active-branch messages and run before_agent_start.
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

        # Snapshot object identities of pre-run messages. We can't slice by
        # index because per-turn extensions (e.g. micro_compact) may mutate
        # the list in place via ``messages[:] = compacted`` from a
        # ``before_send_to_llm`` handler — design ``extension-as-scenario``
        # §10b.2: SessionManager owns durable history; context is per-turn
        # ephemeral. Identity-based diff stays correct under any such
        # rewrite.
        pre_run_ids: set[int] = {id(m) for m in messages}

        # 5. Run the loop.
        final_messages = await self._loop.run(
            messages=messages,
            model=self._require_model(),
            tools=self._tools,
            system=system_prompt,
            signal=signal,
        )

        # 6. Append every new assistant / tool_result message — those whose
        # identities did not exist in the pre-run snapshot, in the order
        # they appear in the returned list.
        persisted_context = self._session_manager.build_session_context().messages
        cursor: str | None = self._session_manager.get_leaf_id() or entry.id
        for msg in final_messages:
            if id(msg) in pre_run_ids:
                continue
            if msg in persisted_context:
                continue
            if cursor is None:
                self._session_manager.reset_leaf()
            else:
                self._session_manager.branch(cursor)
            child = self._session_manager.append_message(msg)
            cursor = child.id

        return final_messages

    # --- tick (resume-without-prompt) -------------------------------------

    async def tick(
        self,
        *,
        signal: asyncio.Event | None = None,
    ) -> list[AgentMessage]:
        """Advance the session by one decide-cycle without new user input.

        Used by the CLI when ``agentm --resume <sid>`` is invoked with no
        positional prompt: harness-injected messages (e.g. the prefix-replay
        ``reminder_seed`` atom) are the source of the first message, and the
        kernel must give extensions a chance to ``Inject`` BEFORE any LLM
        call. Fires :class:`AgentStartEvent` then a synthetic
        :class:`DecideTurnActionEvent` whose default is
        ``Stop(NoPendingInput())``; resolves the decision via the same
        lattice the kernel uses post-turn.

        * On ``Inject(messages=[...])`` — append the injected messages to
          the live context and hand off to :meth:`AgentLoop.run` for one
          (or more) normal turns. The next assistant message is persisted
          identically to the ``prompt`` path.
        * Otherwise — emit :class:`AgentEndEvent` with
          ``NoPendingInput()`` and return the unchanged message list. No
          user / assistant message is appended to the session log.
        """

        # Drain queued send_user_messages so atoms that pushed content via
        # the standard queue still get to drive this tick. Harmless when
        # the queue is empty.
        await self._drain_pending_user_messages()

        # Build the live context + system prompt — symmetric with ``prompt``
        # so before_agent_start handlers see the same state they always do.
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

        if not isinstance(action, Inject):
            # No extension provided input → terminate cleanly. The default
            # ``NoPendingInput`` cause is the literal that observers should
            # match on. A non-Inject override (Stop / Step) is also treated
            # as "no work" — Step has no meaning here because there is no
            # assistant turn to chain off of.
            cause = action.cause if isinstance(action, Stop) else default_action.cause
            await self._bus.emit(
                AgentEndEvent.CHANNEL,
                AgentEndEvent(messages=messages, cause=cause),
            )
            return messages

        # Inject — splice the injected messages into the live context and
        # run the kernel loop. Snapshot identities BEFORE the run so the
        # post-loop diff is consistent with the ``prompt`` path. The
        # injected messages themselves were materialised here in
        # ``tick`` (not by the kernel loop), so they must be persisted as
        # session entries — include them in the diff by NOT capturing
        # their ids in the pre-run snapshot.
        pre_run_ids: set[int] = {id(m) for m in messages}
        messages.extend(action.messages)

        final_messages = await self._loop.run(
            messages=messages,
            model=self._require_model(),
            tools=self._tools,
            system=system_prompt,
            signal=signal,
        )

        persisted_context = self._session_manager.build_session_context().messages
        cursor: str | None = self._session_manager.get_leaf_id()
        for msg in final_messages:
            if id(msg) in pre_run_ids:
                continue
            if msg in persisted_context:
                continue
            if cursor is None:
                self._session_manager.reset_leaf()
            else:
                self._session_manager.branch(cursor)
            child = self._session_manager.append_message(msg)
            cursor = child.id

        return final_messages

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

    async def _drain_pending_user_messages(self) -> None:
        """Pop every queued ``send_user_message`` payload and append it as a
        user-message entry. Called once per ``prompt`` before the caller's
        text is appended, so queued items act as turn-prefix context.
        """

        while self._pending_user_messages:
            queued = self._pending_user_messages.pop(0)
            content: list[TextContent | ImageContent]
            if isinstance(queued, str):
                content = [TextContent(type="text", text=queued)]
            else:
                content = list(queued)
            queued_msg = UserMessage(role="user", content=content, timestamp=now())
            self._append_message(queued_msg)

    def _build_user_message(
        self, *, text: str, images: list[ImageContent] | None
    ) -> UserMessage:
        content: list[TextContent | ImageContent] = []
        if text:
            content.append(TextContent(type="text", text=text))
        if images:
            content.extend(images)
        return UserMessage(role="user", content=content, timestamp=now())

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
