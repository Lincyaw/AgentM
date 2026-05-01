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

Hard rule: this module imports only stdlib + ``agentm.core.kernel`` + the
three sibling v2 modules.
"""

from __future__ import annotations

import asyncio
import inspect
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentm.core.kernel import (
    AgentEndEvent,
    AgentLoop,
    AgentMessage,
    EventBus,
    ImageContent,
    LoopConfig,
    Model,
    TextContent,
    Tool,
    UserMessage,
)

from agentm.harness.events import (
    BeforeAgentStartEvent,
    ChildSessionEndEvent,
    ChildSessionStartEvent,
    SessionReadyEvent,
    SessionShutdownEvent,
)
from agentm.harness.extension import (
    CommandSpec,
    ExtensionLoadError,
    ProviderConfig,
    ReadonlySession,
    Renderer,
    _ExtensionAPIImpl,
    load_extension,
)
from agentm.harness.resource_loader import (
    DefaultResourceLoader,
    ResourceLoader,
)
from agentm.harness.session_manager import (
    InMemorySessionManager,
    SessionEntry,
    SessionManager,
    message_entry,
)


# --- Config -----------------------------------------------------------------


@dataclass
class AgentSessionConfig:
    """Knobs handed to :func:`AgentSession.create`. Only ``cwd``, ``provider``
    are required; everything else has a sane default for embedded use."""

    cwd: str
    extensions: list[tuple[str, dict[str, Any]]]
    provider: tuple[str, dict[str, Any]]
    initial_messages: list[AgentMessage] = field(default_factory=list)
    session_manager: SessionManager | None = None
    resource_loader: ResourceLoader | None = None
    loop_config: LoopConfig | None = None
    # --- Child-session lifecycle (used by sub-agent extensions) ----------
    parent_bus: EventBus | None = None
    """If set, ``child_session_start`` / ``child_session_end`` are emitted on
    this bus when the session is created and shut down. Used by the
    ``sub_agent`` extension to roll up nested sessions on the parent."""

    parent_session_id: str | None = None
    """Caller-supplied id of the parent session. Surfaces verbatim in the
    child-lifecycle events. ``None`` becomes ``"unknown"`` in the payload."""

    purpose: str = "root"
    """Caller-defined purpose label, e.g. ``"subagent:rca_worker"``;
    surfaces verbatim in :class:`ChildSessionStartEvent`."""


# --- Helpers ----------------------------------------------------------------


class _SessionView:
    """``ReadonlySession`` adapter over a ``SessionManager``.

    Exposes message reads plus the one mutation extensions are allowed:
    ``append_entry`` for persisting structured payloads (compaction summaries,
    hypothesis snapshots, plan submissions) into the entry tree. Everything
    else (fork / navigate) stays inside the harness.
    """

    def __init__(self, sm: SessionManager) -> None:
        self._sm = sm

    def get_messages(self) -> list[AgentMessage]:
        return self._sm.get_messages()

    def append_entry(
        self,
        type: str,
        payload: Any,
        parent_id: str | None = None,
    ) -> str:
        if parent_id is None:
            branch = self._sm.get_active_branch()
            parent_id = branch[-1].id if branch else None
        entry = SessionEntry(
            type=type,
            id=uuid.uuid4().hex,
            parent_id=parent_id,
            timestamp=_now(),
            payload=payload,
        )
        self._sm.append(entry)
        return entry.id


def _now() -> float:
    return time.time()


def _collect_system_replacement(returns: list[Any]) -> str | None:
    """Pick the last non-None ``system`` replacement from handler returns.

    Mirrors ``loop._collect_replacement`` semantics: handlers may return a
    dict ``{"system": "..."}`` to override the assembled system prompt; the
    most recently registered authoritative voice wins.
    """

    chosen: str | None = None
    for value in returns:
        if isinstance(value, dict) and value.get("system") is not None:
            candidate = value["system"]
            if isinstance(candidate, str):
                chosen = candidate
    return chosen


# --- AgentSession -----------------------------------------------------------


class AgentSession:
    """Top-level v2 session façade. Construct via :meth:`create`."""

    def __init__(
        self,
        *,
        cwd: str,
        bus: EventBus,
        session_manager: SessionManager,
        resource_loader: ResourceLoader,
        loop: AgentLoop,
        active_provider: ProviderConfig,
        tools: list[Tool],
        commands: dict[str, CommandSpec],
        providers: dict[str, ProviderConfig],
        renderers: dict[str, Renderer],
        api: _ExtensionAPIImpl,
        pending_user_messages: list[str | list[Any]],
        session_id: str,
        parent_bus: EventBus | None,
        parent_session_id: str | None,
        purpose: str,
    ) -> None:
        self._cwd = cwd
        self._bus = bus
        self._session_manager = session_manager
        self._resources = resource_loader
        self._loop = loop
        self._active_provider = active_provider
        self._tools = tools
        self._commands = commands
        self._providers = providers
        self._renderers = renderers
        self._extension_api = api
        self._pending_user_messages = pending_user_messages
        self._session_id = session_id
        self._parent_bus = parent_bus
        self._parent_session_id = parent_session_id
        self._purpose = purpose
        # Set by the cost_budget extension via the cost_budget_exceeded
        # channel; checked at the top of ``prompt`` so the next turn
        # short-circuits cleanly with stop_reason="budget".
        self._budget_exceeded: bool = False

    # --- Construction -----------------------------------------------------

    @classmethod
    async def create(cls, config: AgentSessionConfig) -> "AgentSession":
        """Build a session: assemble subsystems, load extensions, return."""

        bus = EventBus()
        session_manager: SessionManager = (
            config.session_manager
            if config.session_manager is not None
            else InMemorySessionManager()
        )
        resource_loader: ResourceLoader = (
            config.resource_loader
            if config.resource_loader is not None
            else DefaultResourceLoader(cwd=Path(config.cwd))
        )

        tools: list[Tool] = []
        commands: dict[str, CommandSpec] = {}
        providers: dict[str, ProviderConfig] = {}
        renderers: dict[str, Renderer] = {}
        pending_user_messages: list[str | list[Any]] = []

        # We need a forward reference to the picked-up active provider so the
        # api.model property reflects it once the provider extension runs.
        active_provider_box: dict[str, ProviderConfig | None] = {"value": None}

        def _model_getter() -> Model | None:
            cur = active_provider_box["value"]
            return cur.model if cur is not None else None

        def _provider_getter() -> ProviderConfig | None:
            return active_provider_box["value"]

        session_view: ReadonlySession = _SessionView(session_manager)

        api = _ExtensionAPIImpl(
            bus=bus,
            cwd=config.cwd,
            session=session_view,
            tools=tools,
            commands=commands,
            providers=providers,
            renderers=renderers,
            pending_user_messages=pending_user_messages,
            model_getter=_model_getter,
            provider_getter=_provider_getter,
        )

        # Load auxiliary extensions first.
        for module_path, ext_cfg in config.extensions:
            result = load_extension(module_path, api, ext_cfg)
            if inspect.isawaitable(result):
                await result

        # Load the provider extension. After it returns, we expect it to have
        # registered a ProviderConfig.
        provider_path, provider_cfg = config.provider
        result = load_extension(provider_path, api, provider_cfg)
        if inspect.isawaitable(result):
            await result

        if not providers:
            raise ExtensionLoadError(
                provider_path,
                RuntimeError(
                    "provider extension did not call api.register_provider"
                ),
            )

        # Pick the most recently registered provider as active. dict insertion
        # order is preserved on Python 3.7+; the last-inserted entry is the
        # authoritative one.
        active_name = next(reversed(providers))
        active_provider = providers[active_name]
        active_provider_box["value"] = active_provider

        # Build the kernel loop now that we have a stream_fn.
        loop = AgentLoop(
            stream_fn=active_provider.stream_fn,
            bus=bus,
            config=config.loop_config or LoopConfig(),
        )

        # Seed initial messages (if any) into the session manager.
        parent_id: str | None = (
            session_manager.get_active_branch()[-1].id
            if session_manager.get_active_branch()
            else None
        )
        for msg in config.initial_messages:
            entry = message_entry(msg, parent_id=parent_id)
            session_manager.append(entry)
            parent_id = entry.id

        session_id = uuid.uuid4().hex
        instance = cls(
            cwd=config.cwd,
            bus=bus,
            session_manager=session_manager,
            resource_loader=resource_loader,
            loop=loop,
            active_provider=active_provider,
            tools=tools,
            commands=commands,
            providers=providers,
            renderers=renderers,
            api=api,
            pending_user_messages=pending_user_messages,
            session_id=session_id,
            parent_bus=config.parent_bus,
            parent_session_id=config.parent_session_id,
            purpose=config.purpose,
        )

        # Latch budget-exceeded once subscribed extensions emit it. The flag
        # is checked at the top of every ``prompt`` so the next turn
        # short-circuits with ``stop_reason='budget'``. Pure event-bus
        # signalling per §10b.8 — no exceptions cross handler boundaries.
        def _on_budget_exceeded(_: Any) -> None:
            instance._budget_exceeded = True

        bus.on("cost_budget_exceeded", _on_budget_exceeded)

        # Emit ``session_ready`` after every extension is loaded and the
        # active provider is picked. This is the only point where extensions
        # are guaranteed to see the final tool/command/model set; ``tool_filter``
        # and similar post-install scrubbers hook here.
        await bus.emit(
            "session_ready",
            SessionReadyEvent(
                cwd=config.cwd,
                session_id=session_id,
                tool_names=tuple(t.name for t in tools),
                command_names=tuple(commands.keys()),
                model=active_provider.model,
            ),
        )

        # Emit child-session lifecycle on the parent's bus (if any) so
        # ``sub_agent`` / ``trajectory`` extensions can roll up nested work.
        if config.parent_bus is not None:
            await config.parent_bus.emit(
                "child_session_start",
                ChildSessionStartEvent(
                    child_session_id=session_id,
                    parent_session_id=config.parent_session_id or "unknown",
                    purpose=config.purpose,
                ),
            )

        return instance

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
        # v0: no allowlist filtering yet; that lands when tool_filter is
        # ported as a builtin extension in Phase 2.
        return list(self._tools)

    @property
    def model(self) -> Model | None:
        return self._active_provider.model

    @property
    def cwd(self) -> str:
        return self._cwd

    @property
    def session_id(self) -> str:
        """Stable random id assigned at ``create``. Appears in
        :class:`ChildSessionStartEvent` / :class:`ChildSessionEndEvent`
        payloads when this session is a child of another."""
        return self._session_id

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

        # 1. Budget gate: if a previous turn tripped cost_budget_exceeded,
        # short-circuit with a stop_reason='budget' agent_end and persist
        # nothing. The flag stays latched until reset by an extension.
        if self._budget_exceeded:
            await self._bus.emit(
                "agent_end",
                AgentEndEvent(
                    messages=self._session_manager.get_messages(),
                    stop_reason="budget",
                ),
            )
            return self._session_manager.get_messages()

        # 2. Drain ``send_user_message`` queue (FIFO) into user-message
        # entries in the session. This is how ``sub_agent.inject_instruction``
        # and similar extensions push content into the next turn.
        await self._drain_pending_user_messages()

        # 3. Build the caller's user message, after any drained queue items
        # so they appear as turn-prefix context.
        text = text.replace("//", "/", 1) if text.lstrip().startswith("//") else text
        user_msg = self._build_user_message(text=text, images=images)
        entry = self._append_message(user_msg)

        # 4. Gather active-branch messages and run before_agent_start.
        messages = self._session_manager.get_messages()
        system_prompt = self._build_system_prompt()
        before_returns = await self._bus.emit(
            "before_agent_start",
            BeforeAgentStartEvent(messages=messages, system=system_prompt),
        )
        replacement_system = _collect_system_replacement(before_returns)
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
        budget_before_run = self._budget_exceeded

        # 5. Run the loop.
        final_messages = await self._loop.run(
            messages=messages,
            model=self._active_provider.model,
            tools=self._tools,
            system=system_prompt,
            signal=signal,
        )

        # 6. Append every new assistant / tool_result message — those whose
        # identities did not exist in the pre-run snapshot, in the order
        # they appear in the returned list.
        active_branch = self._session_manager.get_active_branch()
        cursor: str | None = active_branch[-1].id if active_branch else entry.id
        for msg in final_messages:
            if id(msg) in pre_run_ids:
                continue
            child = message_entry(msg, parent_id=cursor)
            self._session_manager.append(child)
            cursor = child.id

        if self._budget_exceeded and not budget_before_run:
            await self._bus.emit(
                "agent_end",
                AgentEndEvent(
                    messages=final_messages,
                    stop_reason="budget",
                ),
            )

        return final_messages

    # --- prompt helpers ---------------------------------------------------

    async def _preprocess_input(
        self, text: str
    ) -> tuple[str, list[AgentMessage] | None]:
        """Dispatch code commands, then let ``input`` handlers rewrite text.

        Unknown slash-prefixed text is left alone unless an ``input`` handler
        mutates it, so templates can expand and unmatched commands still reach
        the model verbatim.
        """

        stripped = text.lstrip()
        if not stripped.startswith("/") or stripped.startswith("//"):
            return text, None
        head, _, rest = stripped[1:].partition(" ")
        if not head:
            return text, None
        cmd = self._commands.get(head)
        if cmd is not None:
            result = cmd.handler(rest.strip(), self._extension_api)
            if inspect.isawaitable(result):
                await result
            return text, self._session_manager.get_messages()

        event = {"text": text}
        await self._bus.emit("input", event)
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
            queued_msg = UserMessage(
                role="user", content=content, timestamp=_now()
            )
            self._append_message(queued_msg)

    def _build_user_message(
        self, *, text: str, images: list[ImageContent] | None
    ) -> UserMessage:
        content: list[TextContent | ImageContent] = []
        if text:
            content.append(TextContent(type="text", text=text))
        if images:
            content.extend(images)
        return UserMessage(role="user", content=content, timestamp=_now())

    def _append_message(self, msg: AgentMessage) -> Any:
        active_branch = self._session_manager.get_active_branch()
        parent_id: str | None = active_branch[-1].id if active_branch else None
        entry = message_entry(msg, parent_id=parent_id)
        self._session_manager.append(entry)
        return entry

    # --- Lifecycle --------------------------------------------------------

    async def shutdown(self) -> None:
        """Signal extensions and clear handlers.

        Phase 1 emits a single ``session_shutdown`` event then drops every
        subscription. Extensions that need cleanup hook ``on('session_shutdown')``.
        """

        await self._bus.emit("session_shutdown", SessionShutdownEvent(cwd=self._cwd))

        # Notify the parent (if any) BEFORE clearing handlers so that an
        # extension subscribed on the parent bus can still observe the end
        # event with an accurate message count.
        if self._parent_bus is not None:
            await self._parent_bus.emit(
                "child_session_end",
                ChildSessionEndEvent(
                    child_session_id=self._session_id,
                    parent_session_id=self._parent_session_id or "unknown",
                    final_message_count=len(
                        self._session_manager.get_messages()
                    ),
                    error=None,
                ),
            )

        self._bus.clear()

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
