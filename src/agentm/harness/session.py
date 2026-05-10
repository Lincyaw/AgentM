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
    EventBus,
    ImageContent,
    Model,
    TextContent,
    Tool,
    UserMessage,
)
from agentm.harness.atom_reloader import LoadedAtom
from agentm.harness.events import (
    BeforeAgentStartEvent,
    ChildSessionEndEvent,
    SessionShutdownEvent,
)
from agentm.harness.resource_loader import ResourceLoader
from agentm.harness.session_helpers import (
    collect_start_veto,
    collect_system_replacement,
    now,
)
from agentm.harness.session_runtime import SessionRuntime
from agentm.harness.session_manager import SessionManager

logger = logging.getLogger(__name__)

_LoadedAtom = LoadedAtom


# --- Config -----------------------------------------------------------------
# ``AgentSessionConfig`` lives in ``harness.session_config`` so extensions
# (which cannot import this module per §11.4.5) can still construct one for
# ``api.spawn_child_session``. Re-exported below for backward-compatible
# imports from ``agentm.harness.session``.

from agentm.harness.session_config import (  # noqa: E402
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
        purpose: str,
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
        self._purpose = purpose
        # Set by the cost_budget extension via the cost_budget_exceeded
        # channel; checked at the top of ``prompt`` so the next turn
        # short-circuits cleanly with stop_reason="budget".
        self._budget_exceeded: bool = False
        # Per-session sandbox for ``atom_source_overrides`` (per-task-evolution
        # loop §6.3). Populated by ``AgentSession.create`` when the config
        # supplies overrides; cleaned up on ``shutdown``. ``None`` for
        # ordinary sessions — no filesystem cost.
        self._eval_sandbox: Path | None = eval_sandbox

    # --- Compatibility aliases (legacy attribute access) ------------------
    # Tests and a few internal callers used to read these dicts off the
    # session itself; they now live on the reloader. Keeping the attribute
    # surface stable avoids a fan-out test churn.

    @property
    def _loaded_atoms_by_name(self) -> dict[str, LoadedAtom]:
        return self._reloader.loaded_by_name

    @property
    def _owners_by_kind(self) -> dict[str, dict[str, str]]:
        return self._reloader.owners_by_kind

    # --- Construction -----------------------------------------------------

    @classmethod
    async def create(cls, config: AgentSessionConfig) -> "AgentSession":
        from agentm.harness.session_factory import create_agent_session

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
        # v0: no allowlist filtering yet; that lands when tool_filter is
        # ported as a builtin extension in Phase 2.
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
            from agentm.harness.catalog.indexer import index_trace

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
