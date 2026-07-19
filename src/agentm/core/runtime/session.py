"""Session — top-level lifecycle object.

Owns driver task, trajectory, trigger queue, bus, tools, services,
context policies, and shutdown logic.  Also the fork/resume/spawn
entry point.
"""

from __future__ import annotations

import asyncio
import copy
import uuid
from contextlib import contextmanager
from typing import Iterator

from loguru import logger

from agentm.core.abi.messages import AgentMessage, ImageContent, TextContent
from agentm.core.abi.stream import Model, StreamFn
from agentm.core.abi.tool import Tool
from agentm.core.abi.bus import EventBus
from agentm.core.abi.context import (
    ContextPolicy,
    PolicyContext,
    build_context_sync,
)
from agentm.core.abi.events import (
    ChildSessionStartEvent,
    SessionReadyEvent,
    SessionShutdownEvent,
)
from agentm.core.abi.services import ServiceRegistry
from agentm.core.abi.session_api import AgentSessionConfig, SessionContext
from agentm.core.abi.store import SessionMeta, TrajectoryStore
from agentm.core.abi.trajectory import Turn, TurnRef
from agentm.core.abi.tree import SessionGraphProtocol
from agentm.core.abi.trigger import Trigger, TriggerRenderer, UserInput
from agentm.core.abi.lifecycle import (
    ForkEvent,
    LifecycleHook,
    LifecycleHookRegistry,
    ResumeEvent,
)
from agentm.core.runtime.driver import ThinkingLevel, drive
from agentm.core.runtime.trajectory import Trajectory
from agentm.core.runtime.trigger_queue import TriggerQueue


class Session:
    """Top-level session lifecycle object."""

    def __init__(
        self,
        *,
        ctx: SessionContext | None = None,
        session_id: str | None = None,
        trajectory: Trajectory | None = None,
        bus: EventBus | None = None,
        store: TrajectoryStore | None = None,
        graph: SessionGraphProtocol | None = None,
        stream_fn: StreamFn | None = None,
        model: Model | None = None,
        tools: list[Tool] | None = None,
        system: str | None = None,
        context_policies: list[ContextPolicy] | None = None,
        trigger_renderers: dict[str, TriggerRenderer] | None = None,
        max_turns: int | None = None,
        thinking: ThinkingLevel = "off",
        services: ServiceRegistry | None = None,
        cwd: str = "",
        purpose: str = "root",
    ) -> None:
        sid = session_id or uuid.uuid4().hex[:16]
        self.ctx = ctx or SessionContext(
            session_id=sid,
            root_session_id=sid,
            cwd=cwd,
            purpose=purpose,
        )
        self.id = self.ctx.session_id
        self.trajectory = trajectory or Trajectory()
        self.bus = bus or EventBus()
        self.store = store
        self.graph = graph
        self.triggers = TriggerQueue()
        self.tools: list[Tool] = list(tools or [])
        self.system = system
        self.context_policies: list[ContextPolicy] = list(context_policies or [])
        self.trigger_renderers: dict[str, TriggerRenderer] = dict(trigger_renderers or {})
        self.services = services or ServiceRegistry()

        self._stream_fn = stream_fn
        self._model = model
        self._max_turns = max_turns
        self._thinking = thinking
        self._interrupt = asyncio.Event()
        self._shutdown = asyncio.Event()
        self._closed = False
        self._driver_task: asyncio.Task[None] | None = None
        self._pending_installs: list[asyncio.Task[object]] = []
        self.lifecycle = LifecycleHookRegistry()

        if self.graph is not None:
            self.graph.register(
                self.id,
                parent_id=self.ctx.parent_session_id,
                purpose=purpose,
            )

    # --- Lifecycle ---

    def start(self) -> None:
        if self._driver_task is not None:
            return
        if self._stream_fn is None:
            raise RuntimeError("cannot start: no stream_fn")
        if self._model is None:
            raise RuntimeError("cannot start: no model")

        policy_ctx = PolicyContext(
            session_id=self.id,
            parent_session_id=self.ctx.parent_session_id,
            services={n: self.services.get(n) for n in self.services.names()},
            store=self.store,
            model=self._model,
            stream_fn=self._stream_fn,
        )
        for policy in self.context_policies:
            if hasattr(policy, "bind"):
                try:
                    policy.bind(policy_ctx)
                except Exception:
                    logger.exception("policy bind failed")

        self.bus.freeze_clear()
        self._driver_task = asyncio.create_task(
            self._run_driver(),
            name=f"v2-driver-{self.id}",
        )
        self.bus.emit_sync(SessionReadyEvent.CHANNEL, SessionReadyEvent(
            session_id=self.id,
            tool_names=tuple(t.name for t in self.tools),
        ))

    async def _run_driver(self) -> None:
        try:
            assert self._stream_fn is not None
            assert self._model is not None
            await drive(
                trajectory=self.trajectory,
                triggers=self.triggers,
                bus=self.bus,
                stream_fn=self._stream_fn,
                model=self._model,
                tools=self.tools,
                store=self.store,
                session_id=self.id,
                system=self.system,
                context_policies=self.context_policies,
                trigger_renderers=self.trigger_renderers,
                interrupt=self._interrupt,
                shutdown=self._shutdown,
                max_turns=self._max_turns,
                thinking=self._thinking,
                lifecycle=self.lifecycle,
            )
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("session driver crashed")

    async def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._shutdown.set()
        self.triggers.close()
        if self._driver_task is not None:
            try:
                await asyncio.wait_for(self._driver_task, timeout=30.0)
            except TimeoutError:
                self._driver_task.cancel()
                try:
                    await self._driver_task
                except (asyncio.CancelledError, Exception):
                    logger.debug("driver post-cancel cleanup")
            except asyncio.CancelledError:
                pass
        await self.bus.emit(SessionShutdownEvent.CHANNEL, SessionShutdownEvent())
        self.bus._force_clear()

    # --- Input ---

    async def prompt(self, text: str, *, images: list[ImageContent] | None = None) -> None:
        content: list[TextContent | ImageContent] = []
        if text:
            content.append(TextContent(type="text", text=text))
        if images:
            content.extend(images)
        self.triggers.push(UserInput(content=tuple(content)))

    def push_trigger(self, trigger: Trigger) -> None:
        self.triggers.push(trigger)

    def interrupt(self) -> None:
        self._interrupt.set()

    async def idle(self, timeout: float | None = None) -> bool:
        return await self.triggers.wait_quiescent(timeout)

    async def run(self, text: str) -> list[AgentMessage]:
        """Start driver (if needed), prompt, wait for completion, return messages.

        Blocking convenience for child sessions — the "give it a prompt
        and get the answer" pattern used by sub_agent, workflow, and goal.
        """
        if self._driver_task is None:
            self.start()
        await self.prompt(text)
        await self.idle()
        return self.get_messages()

    @contextmanager
    def track_background(self) -> Iterator[None]:
        self.triggers.note_work_started()
        try:
            yield
        finally:
            self.triggers.note_work_finished()

    # --- Query ---

    @property
    def model(self) -> Model | None:
        return self._model

    @property
    def session_id(self) -> str:
        return self.id

    def get_messages(self) -> list[AgentMessage]:
        return build_context_sync(self.trajectory.turns, self.trigger_renderers)

    def get_turns(self) -> list[Turn]:
        return list(self.trajectory.turns)

    def status(self) -> dict[str, str | int | list[str]]:
        phase: str
        if self._closed:
            phase = "closed"
        elif self.trajectory.is_executing:
            phase = "running"
        elif not self.triggers.is_empty():
            phase = "draining"
        else:
            phase = "idle"
        return {
            "phase": phase,
            "session_id": self.id,
            "turns": len(self.trajectory),
            "tool_names": [t.name for t in self.tools],
        }

    # --- Bus delegation ---

    def on(
        self,
        channel: str,
        handler: object,
        *,
        priority: int = 500,
    ) -> object:
        return self.bus.on(channel, handler, priority=priority)  # type: ignore[arg-type]

    # --- Registration ---

    def register_tool(self, tool: Tool) -> None:
        existing = {t.name for t in self.tools}
        if tool.name in existing:
            raise ValueError(f"duplicate tool: {tool.name}")
        self.tools.append(tool)

    def register_context_policy(self, policy: ContextPolicy, *, priority: int = 500) -> None:
        policy._priority = priority  # type: ignore[attr-defined]
        self.context_policies.append(policy)
        self.context_policies.sort(key=lambda p: getattr(p, "_priority", 500))

    def register_trigger_renderer(self, source: str, renderer: TriggerRenderer) -> None:
        self.trigger_renderers[source] = renderer

    def register_trigger_codec(self, source: str, codec: object) -> None:
        from agentm.core.abi.codec import DEFAULT_CODEC
        DEFAULT_CODEC.register_trigger_codec(source, codec)  # type: ignore[arg-type]

    def register_lifecycle_hook(self, hook: LifecycleHook) -> None:
        self.lifecycle.register(hook)

    def _v2_push_trigger_stub(self, *, source: str, payload: str, **kwargs: object) -> None:
        """Bridge for atoms that construct triggers from raw source+payload."""
        from agentm.core.abi.trigger import (
            BackgroundCompletion, MonitorFire, SubagentResult, UserInput,
        )
        if source == "monitor":
            monitor_id = str(kwargs.get("dedup_key", ""))
            self.push_trigger(MonitorFire(monitor_id=monitor_id, payload=payload))
        elif source == "background":
            task_id = str(kwargs.get("task_id", ""))
            terminal = bool(kwargs.get("terminal", False))
            self.push_trigger(BackgroundCompletion(task_id=task_id, payload=payload, terminal=terminal))
        elif source == "subagent":
            child_id = str(kwargs.get("child_session_id", ""))
            terminal = bool(kwargs.get("terminal", False))
            self.push_trigger(SubagentResult(child_session_id=child_id, payload=payload, terminal=terminal))
        else:
            content = (TextContent(type="text", text=payload),)
            self.push_trigger(UserInput(content=content))

    def _v2_send_user_stub(self, text: str) -> None:
        """Bridge for atoms that want to emit a message to the user."""
        from agentm.core.abi.events import DiagnosticEvent
        self.bus.emit_sync(DiagnosticEvent.CHANNEL, DiagnosticEvent(
            level="info", source="session", message=text,
        ))

    def register_provider(self, name: str, config: object) -> None:
        """Register an LLM provider — sets model and stream_fn from config."""
        stream_fn = getattr(config, "stream_fn", None)
        model = getattr(config, "model", None)
        if stream_fn is not None:
            self._stream_fn = stream_fn
        if model is not None:
            self._model = model
        self.services.register(f"provider:{name}", config)

    def get_service(self, key: str) -> object | None:
        """Bridge — atoms that called api.get_service() now call session.get_service()."""
        return self.services.get(key)

    def set_service(self, key: str, value: object) -> None:
        """Bridge — atoms that called api.set_service() now call session.set_service()."""
        self.services.register(key, value)

    def register_operations(self, **kwargs: object) -> None:
        """Bridge — register bash/sandbox operations as services."""
        for key, value in kwargs.items():
            self.services.register(f"operations:{key}", value)

    def register_resource_writer(self, writer: object) -> None:
        """Bridge — register the resource writer as a service."""
        self.services.register("resource_writer", writer)

    def add_observer(self, observer: object) -> None:
        """Bridge — bus observer registration (observers not yet wired into bus dispatch)."""
        self.services.register("bus_observer", observer)

    def install_atom(self, module_path: str, config: dict[str, object] | None = None) -> None:
        """Load and install an atom at runtime."""
        from agentm.core.runtime.extension import load_extension
        result = load_extension(module_path, self, config or {})
        if result is not None:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                task = asyncio.ensure_future(result)
                self._pending_installs.append(task)

    def unload_atom(self, name: str, **kwargs: object) -> None:
        """Placeholder — atom unloading not yet supported in v2."""
        logger.warning("unload_atom({!r}) called but not implemented in v2", name)

    def reload_atom(self, name: str, **kwargs: object) -> None:
        """Placeholder — atom reloading not yet supported in v2."""
        logger.warning("reload_atom({!r}) called but not implemented in v2", name)

    def list_atoms(self) -> list[str]:
        """Placeholder — atom listing not yet supported in v2."""
        return []

    @property
    def cwd(self) -> str:
        return self.ctx.cwd

    @property
    def root_session_id(self) -> str:
        return self.ctx.root_session_id

    @property
    def scenario(self) -> str | None:
        return self.ctx.scenario

    @property
    def lineage(self) -> dict[str, str]:
        return {
            "session_id": self.id,
            "root_session_id": self.ctx.root_session_id,
            "parent_session_id": self.ctx.parent_session_id or "",
            "purpose": self.ctx.purpose,
        }

    @property
    def provider(self) -> object | None:
        """Active ProviderConfig, if any provider atom registered one."""
        for name in self.services.names():
            if name.startswith("provider:"):
                return self.services.get(name)
        return None

    @property
    def experiment(self) -> dict[str, object] | None:
        return self.services.get("experiment")  # type: ignore[return-value]

    # --- Spawn (child session with inheritance) ---

    async def spawn(
        self,
        *,
        purpose: str = "subagent",
        tools: list[Tool] | None = None,
        system: str | None = None,
        model: Model | None = None,
        stream_fn: StreamFn | None = None,
        scenario: str | None = None,
        cwd: str | None = None,
        max_turns: int | None = None,
        extra_services: ServiceRegistry | None = None,
    ) -> "Session":
        """Spawn a lightweight child inheriting parent config.

        Only specify what you want to override.
        """

        child_ctx = self.ctx.child(
            session_id=uuid.uuid4().hex[:16],
            purpose=purpose,
            cwd=cwd,
            scenario=scenario,
        )
        child_services = ServiceRegistry()
        child_services.update_from(self.services)
        if extra_services is not None:
            child_services.update_from(extra_services)

        child = Session(
            ctx=child_ctx,
            trajectory=Trajectory(),
            bus=EventBus(),
            store=self.store,
            graph=self.graph,
            stream_fn=stream_fn or self._stream_fn,
            model=model or self._model,
            tools=tools if tools is not None else list(self.tools),
            system=system if system is not None else self.system,
            context_policies=[copy.copy(p) for p in self.context_policies],
            trigger_renderers=dict(self.trigger_renderers),
            max_turns=max_turns or self._max_turns,
            thinking=self._thinking,
            services=child_services,
        )

        await self._register_child(child, purpose=purpose)
        return child

    async def spawn_child_session(
        self,
        config: AgentSessionConfig,
    ) -> "Session":
        """Spawn a fully-constructed child via the session factory.

        Goes through the full factory pipeline: resolves scenario,
        loads extensions, installs atoms.  Use this when the child
        needs a different scenario or extension set from the parent.
        """
        from agentm.core.runtime.session_factory import create_child_session

        child = await create_child_session(parent=self, config=config)
        await self._register_child(child, purpose=config.purpose)
        return child

    async def _register_child(self, child: "Session", *, purpose: str) -> None:
        if child.store is not None:
            child.store.create_session(SessionMeta(
                id=child.id,
                parent_id=self.id,
                purpose=purpose,
                cwd=child.ctx.cwd,
            ))

        if child.graph is not None:
            child.graph.register(
                child.id,
                parent_id=self.id,
                purpose=purpose,
                edge_kind="spawned",
            )

        await self.bus.emit(ChildSessionStartEvent.CHANNEL, ChildSessionStartEvent(
            child_session_id=child.id,
            parent_session_id=self.id,
            purpose=purpose,
        ))

    # --- Fork / Resume ---

    @classmethod
    async def fork(cls, source: "Session", at: TurnRef, *, purpose: str = "fork") -> "Session":
        prefix = source.trajectory.prefix(at)
        child_ctx = source.ctx.child(
            session_id=uuid.uuid4().hex[:16],
            purpose=purpose,
        )

        child_services = ServiceRegistry()
        child_services.update_from(source.services)

        forked = cls(
            ctx=child_ctx,
            trajectory=prefix,
            bus=EventBus(),
            store=source.store,
            graph=source.graph,
            stream_fn=source._stream_fn,
            model=source._model,
            system=source.system,
            context_policies=[copy.copy(p) for p in source.context_policies],
            trigger_renderers=dict(source.trigger_renderers),
            max_turns=source._max_turns,
            thinking=source._thinking,
            services=child_services,
        )

        if forked.store is not None:
            forked.store.create_session(SessionMeta(
                id=forked.id, parent_id=source.id,
                fork_point=at, purpose=purpose,
                cwd=source.ctx.cwd,
            ))
            for turn in prefix.turns:
                forked.store.append(forked.id, turn)

        if forked.graph is not None:
            forked.graph.register(
                forked.id, parent_id=source.id,
                fork_point=at, purpose=purpose,
                edge_kind="forked",
            )

        await source.lifecycle.fire_fork(ForkEvent(
            source_session_id=source.id,
            fork_session_id=forked.id,
            fork_point=at,
            source_turns=tuple(prefix.turns),
        ))

        return forked

    @classmethod
    async def resume(cls, session_id: str, store: TrajectoryStore, **kwargs: object) -> "Session":
        meta, turns = store.load(session_id)
        trajectory = Trajectory(turns=turns)
        ctx = SessionContext(
            session_id=session_id,
            root_session_id=session_id,
            parent_session_id=meta.parent_id,
            cwd=meta.cwd,
            purpose=meta.purpose,
        )
        session = cls(ctx=ctx, trajectory=trajectory, store=store, **kwargs)  # type: ignore[arg-type]

        await session.lifecycle.fire_resume(ResumeEvent(
            session_id=session_id,
            committed_turns=tuple(turns),
        ))

        return session


__all__ = ["Session"]
