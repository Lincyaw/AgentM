"""Session — top-level lifecycle object.

Owns driver task, trajectory, trigger queue, bus, tools, services,
context policies, and shutdown logic.  Also the fork/resume/spawn
entry point.
"""

from __future__ import annotations

import asyncio
import copy
import time
import uuid
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import replace
from typing import Iterator, cast

from loguru import logger

from agentm.core.abi.cancel import CancelReason, CancelSignal, EventCancelSource
from agentm.core.abi.codec import CodecRegistry, TriggerCodec
from agentm.core.abi.catalog import (
    AtomCatalog,
    VersionedResourceStore,
)
from agentm.core.abi.lifecycle import EffectScope
from agentm.core.abi.messages import AgentMessage, ImageContent, TextContent
from agentm.core.abi.provider import ProviderConfig, ProviderResolver
from agentm.core.abi.resource import ResourceTxn, ResourceWriter
from agentm.core.abi.stream import Model, StreamFn
from agentm.core.abi.tool import Tool
from agentm.core.abi.tool_executor import ToolExecutor
from agentm.core.abi.bus import EventBus, EventBusObserver, Handler
from agentm.core.abi.context import (
    ContextPolicy,
    PolicyContext,
    build_context_sync,
)
from agentm.core.abi.events import (
    ApiRegisterEvent,
    ChildSessionEndEvent,
    ChildSessionStartEvent,
    SessionReadyEvent,
    SessionShutdownEvent,
)
from agentm.core.abi.services import ServiceRegistry
from agentm.core.abi.roles import (
    ATOM_CATALOG_SERVICE,
    EFFECT_SCOPE_SERVICE,
    PROVIDER_RESOLVER_SERVICE,
    RESOURCE_WRITER_SERVICE,
    RESOURCE_TXN_SERVICE,
    TOOL_EXECUTOR_SERVICE,
    VERSIONED_RESOURCE_STORE_SERVICE,
)
from agentm.core.abi.session_api import AgentSessionConfig, SessionContext
from agentm.core.abi.store import SessionMeta, TrajectoryStore
from agentm.core.abi.trajectory import Turn, TurnRef
from agentm.core.abi.tree import SessionGraphProtocol
from agentm.core.abi.trigger import Trigger, TriggerPriority, TriggerRenderer, UserInput
from agentm.core.runtime.driver import ThinkingLevel, drive
from agentm.core.runtime.session_meta import (
    context_from_session_meta,
    session_meta_config,
)
from agentm.core.runtime.trajectory import Trajectory
from agentm.core.runtime.trigger_queue import TriggerQueue, TriggerReceipt


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
        codec: CodecRegistry | None = None,
        max_turns: int | None = None,
        max_tool_calls: int | None = None,
        tool_allowlist: Sequence[str] | None = None,
        thinking: ThinkingLevel = "off",
        cancel_signal: CancelSignal | None = None,
        provider_resolver: ProviderResolver | None = None,
        services: ServiceRegistry | None = None,
        cwd: str = "",
        purpose: str = "root",
    ) -> None:
        sid = session_id or uuid.uuid4().hex[:16]
        if ctx is None:
            self.ctx = SessionContext(
                session_id=sid,
                root_session_id=sid,
                cwd=cwd,
                purpose=purpose,
            )
        elif not ctx.session_id or not ctx.root_session_id:
            resolved_sid = ctx.session_id or sid
            self.ctx = replace(
                ctx,
                session_id=resolved_sid,
                root_session_id=ctx.root_session_id or resolved_sid,
            )
        else:
            self.ctx = ctx
        self.id = self.ctx.session_id
        if trajectory is not None:
            self.trajectory = trajectory
        else:
            stored_turns: list[Turn] = []
            if store is not None and store.session_exists(self.id):
                _, stored_turns = store.load(self.id)
            self.trajectory = Trajectory(turns=stored_turns)
        self.bus = bus or EventBus()
        self.store = store
        self.graph = graph
        self.triggers = TriggerQueue()
        self.tools: list[Tool] = list(tools or [])
        self.system = system
        self.context_policies: list[ContextPolicy] = list(context_policies or [])
        self.trigger_renderers: dict[str, TriggerRenderer] = dict(trigger_renderers or {})
        store_codec = getattr(store, "codec", None)
        self.codec = codec or (
            store_codec if isinstance(store_codec, CodecRegistry) else CodecRegistry()
        )
        self.services = services or ServiceRegistry()

        self._stream_fn = stream_fn
        self._model = model
        self._max_turns = max_turns
        self._max_tool_calls = max_tool_calls
        self._thinking = thinking
        self._parent_cancel_signal = cancel_signal
        self._interrupt = EventCancelSource()
        self._shutdown = EventCancelSource()
        self._closed = False
        self._driver_error: str | None = None
        self._driver_task: asyncio.Task[None] | None = None
        self.installed_extensions: list[str] = []
        self._active_provider_name: str | None = None
        if provider_resolver is not None:
            self.services.register(
                PROVIDER_RESOLVER_SERVICE,
                provider_resolver,
                scope="host",
            )
        if tool_allowlist is not None:
            self.services.register("tool_allowlist", tuple(tool_allowlist), scope="session")

        if self.graph is not None and self.ctx.parent_session_id is None:
            self.graph.register(
                self.id,
                purpose=self.ctx.purpose,
            )

    # --- Lifecycle ---

    @classmethod
    async def create(cls, config: AgentSessionConfig) -> "Session":
        """Create a fully configured root session from an SDK config."""
        from agentm.core.runtime.session_factory import create_from_config

        return await create_from_config(config)

    def start(self) -> None:
        if self._driver_task is not None:
            return
        self._activate_provider(fallback=self._active_provider_name)
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
                policy.bind(policy_ctx)

        self.bus.freeze_clear()
        self._driver_task = asyncio.create_task(
            self._run_driver(),
            name=f"v2-driver-{self.id}",
        )
        self.bus.emit_sync(SessionReadyEvent.CHANNEL, SessionReadyEvent(
            session_id=self.id,
            root_session_id=self.ctx.root_session_id,
            parent_session_id=self.ctx.parent_session_id,
            cwd=self.ctx.cwd,
            tool_names=tuple(t.name for t in self.tools),
            extension_module_paths=tuple(self.installed_extensions),
            model=self._model,
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
                cancel_signal=self._parent_cancel_signal,
                effect_scope=self.get_effect_scope(),
                resource_writer=self.get_resource_writer(),
                services=self.services,
                tool_executor=self.get_tool_executor(),
                max_turns=self._max_turns,
                max_tool_calls=self._max_tool_calls,
                tool_allowlist=self._tool_allowlist(),
                thinking=self._thinking,
            )
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            self._driver_error = str(exc)
            logger.exception("session driver crashed")

    async def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._shutdown.set("shutdown")
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
        telemetry = self.services.get("session_telemetry")
        shutdown = getattr(telemetry, "shutdown", None)
        if callable(shutdown):
            shutdown()
        self.bus._force_clear()

    # --- Input ---

    async def prompt(
        self,
        text: str,
        *,
        images: list[ImageContent] | None = None,
        priority: TriggerPriority = "next",
        origin: str | None = "human",
        mode: str = "prompt",
    ) -> TriggerReceipt[object]:
        content: list[TextContent | ImageContent] = []
        if text:
            content.append(TextContent(type="text", text=text))
        if images:
            content.extend(images)
        return self.push_trigger(
            UserInput(content=tuple(content)),
            priority=priority,
            origin=origin,
            mode=mode,
        )

    def push_trigger(
        self,
        trigger: Trigger,
        *,
        priority: TriggerPriority = "next",
        target_session_id: str | None = None,
        target_agent_id: str | None = None,
        origin: str | None = None,
        mode: str = "prompt",
        is_meta: bool = False,
        skip_commands: bool = False,
        meta: dict[str, object] | None = None,
    ) -> TriggerReceipt[object]:
        receipt = self.triggers.push(
            trigger,
            priority=priority,
            target_session_id=target_session_id,
            target_agent_id=target_agent_id,
            origin=origin,
            mode=mode,
            is_meta=is_meta,
            skip_commands=skip_commands,
            meta=meta,
        )
        if priority == "now":
            self.interrupt("submit_interrupt")
        return receipt

    def interrupt(self, reason: CancelReason | str = "user_cancel") -> None:
        self._interrupt.set(reason)

    async def idle(self, timeout: float | None = None) -> bool:
        return await self.triggers.wait_quiescent(timeout)

    async def run(self, text: str) -> list[AgentMessage]:
        """Start driver (if needed), prompt, wait for completion, return messages.

        Blocking convenience for child sessions — the "give it a prompt
        and get the answer" pattern used by sub_agent, workflow, and goal.
        """
        if self._driver_task is None:
            self.start()
        receipt = await self.prompt(text)
        await receipt.wait()
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
        handler: Handler,
        *,
        priority: int = 500,
    ) -> Callable[[], None]:
        from agentm.core.runtime.extension import current_installing_extension

        owner = current_installing_extension() or None
        return self.bus.on(channel, handler, priority=priority, owner=owner)

    # --- Registration ---

    def register_tool(self, tool: Tool) -> None:
        existing = {t.name for t in self.tools}
        if tool.name in existing:
            raise ValueError(f"duplicate tool: {tool.name}")
        self.tools.append(tool)
        self._emit_register_event("tool", tool.name, {"tool": tool})

    def register_context_policy(self, policy: ContextPolicy, *, priority: int = 500) -> None:
        policy._priority = priority  # type: ignore[attr-defined]
        self.context_policies.append(policy)
        self.context_policies.sort(key=lambda p: getattr(p, "_priority", 500))
        self._emit_register_event(
            "context_policy",
            type(policy).__name__,
            {"policy": policy, "priority": priority},
        )

    def register_trigger_renderer(self, source: str, renderer: TriggerRenderer) -> None:
        self.trigger_renderers[source] = renderer
        self._emit_register_event(
            "trigger_renderer",
            source,
            {"renderer": renderer},
        )

    def register_trigger_codec(self, source: str, codec: object) -> None:
        if not isinstance(codec, TriggerCodec):
            raise TypeError("trigger codec must implement serialize and deserialize")
        self.codec.register_trigger_codec(source, codec)
        self._emit_register_event(
            "trigger_codec",
            source,
            {"codec": codec},
        )

    def register_provider(
        self,
        name: str,
        config: ProviderConfig,
        *,
        replace: bool = False,
    ) -> None:
        """Register an LLM provider and refresh the active provider."""
        key = f"provider:{name}"
        if self.services.get(key) is not None and not replace:
            raise ValueError(f"provider {name!r} already registered")
        self.services.register(key, config, scope="session")
        self._activate_provider(fallback=name)
        self._emit_register_event("provider", name, {"provider": config})

    def has_provider(self, name: str) -> bool:
        return self.services.get(f"provider:{name}") is not None

    def get_provider(self, name: str | None = None) -> ProviderConfig | None:
        if name is None:
            self._activate_provider(fallback=self._active_provider_name)
        provider_name = name or self._active_provider_name
        if provider_name is None:
            return None
        provider = self.services.get(f"provider:{provider_name}")
        return provider if isinstance(provider, ProviderConfig) else None

    def provider_names(self) -> list[str]:
        prefix = "provider:"
        return sorted(
            name[len(prefix):]
            for name in self.services.names()
            if name.startswith(prefix)
        )

    def _provider_configs(self) -> dict[str, ProviderConfig]:
        prefix = "provider:"
        providers: dict[str, ProviderConfig] = {}
        for service_name in self.services.names():
            if not service_name.startswith(prefix):
                continue
            provider = self.services.get(service_name)
            if isinstance(provider, ProviderConfig):
                providers[service_name[len(prefix):]] = provider
        return providers

    def _provider_resolver(self) -> ProviderResolver | None:
        candidate = self.services.get(PROVIDER_RESOLVER_SERVICE)
        if callable(getattr(candidate, "resolve_provider", None)):
            return cast(ProviderResolver, candidate)
        return None

    def _activate_provider(self, *, fallback: str | None) -> None:
        providers = self._provider_configs()
        if not providers:
            return
        selected = self._resolve_provider_name(providers, fallback=fallback)
        if selected is None:
            return
        provider = providers[selected]
        self._active_provider_name = selected
        self._stream_fn = provider.stream_fn
        self._model = provider.model

    def _resolve_provider_name(
        self,
        providers: dict[str, ProviderConfig],
        *,
        fallback: str | None,
    ) -> str | None:
        resolver = self._provider_resolver()
        if resolver is not None:
            selected = resolver.resolve_provider(providers)
            if selected is not None and selected in providers:
                return selected
        if fallback is not None and fallback in providers:
            return fallback
        if (
            self._active_provider_name is not None
            and self._active_provider_name in providers
        ):
            return self._active_provider_name
        return next(reversed(providers))

    def register_operations(self, **kwargs: object) -> None:
        """Register named operation services."""
        from agentm.core.abi.operations import BashOperations

        protocols = {"bash": BashOperations}
        for key, value in kwargs.items():
            service_name = f"operations:{key}"
            if self.services.has(service_name):
                raise ValueError(f"operation {key!r} already registered")
            self.services.register(
                service_name,
                value,
                protocols.get(key),
                scope="session",
            )
            self._emit_register_event(
                "operations",
                key,
                {"service_name": service_name, "service": value},
            )

    def register_resource_writer(
        self,
        writer: ResourceWriter,
        *,
        replace: bool = False,
    ) -> None:
        service_name = RESOURCE_WRITER_SERVICE
        if self.services.has(service_name) and not replace:
            raise ValueError("resource_writer already registered")
        self.services.register(
            service_name,
            writer,
            ResourceWriter,
            scope="resource",
        )
        self._emit_register_event(
            "resource_writer",
            service_name,
            {"service": writer},
        )

    def get_resource_writer(self) -> ResourceWriter | None:
        return self.services.get(
            RESOURCE_WRITER_SERVICE,
            cast(type[ResourceWriter], ResourceWriter),
        )

    def get_resource_txn(self) -> ResourceTxn | None:
        return self.services.get(
            RESOURCE_TXN_SERVICE,
            cast(type[ResourceTxn], ResourceTxn),
        )

    def register_tool_executor(
        self,
        executor: ToolExecutor,
        *,
        replace: bool = False,
    ) -> None:
        service_name = TOOL_EXECUTOR_SERVICE
        if self.services.has(service_name) and not replace:
            raise ValueError("tool_executor already registered")
        self.services.register(
            service_name,
            executor,
            cast(type[ToolExecutor], ToolExecutor),
            scope="host",
        )
        self._emit_register_event(
            "tool_executor",
            service_name,
            {"service": executor},
        )

    def get_tool_executor(self) -> ToolExecutor | None:
        return self.services.get(
            TOOL_EXECUTOR_SERVICE,
            cast(type[ToolExecutor], ToolExecutor),
        )

    def register_effect_scope(
        self,
        scope: EffectScope,
        *,
        replace: bool = False,
    ) -> None:
        service_name = EFFECT_SCOPE_SERVICE
        if self.services.has(service_name) and not replace:
            raise ValueError("effect_scope already registered")
        self.services.register(
            service_name,
            scope,
            cast(type[EffectScope], EffectScope),
            scope="tree",
        )
        self._emit_register_event(
            "effect_scope",
            service_name,
            {"service": scope},
        )

    def get_effect_scope(self) -> EffectScope | None:
        return self.services.get(
            EFFECT_SCOPE_SERVICE,
            cast(type[EffectScope], EffectScope),
        )

    def register_versioned_resource_store(
        self,
        store: VersionedResourceStore,
        *,
        replace: bool = False,
    ) -> None:
        service_name = VERSIONED_RESOURCE_STORE_SERVICE
        if self.services.has(service_name) and not replace:
            raise ValueError("versioned_resource_store already registered")
        self.services.register(
            service_name,
            store,
            cast(type[VersionedResourceStore], VersionedResourceStore),
            scope="host",
        )
        self._emit_register_event(
            "versioned_resource_store",
            service_name,
            {"service": store},
        )

    def get_versioned_resource_store(self) -> VersionedResourceStore | None:
        return self.services.get(
            VERSIONED_RESOURCE_STORE_SERVICE,
            cast(type[VersionedResourceStore], VersionedResourceStore),
        )

    def register_atom_catalog(
        self,
        catalog: AtomCatalog,
        *,
        replace: bool = False,
    ) -> None:
        service_name = ATOM_CATALOG_SERVICE
        if self.services.has(service_name) and not replace:
            raise ValueError("atom_catalog already registered")
        self.services.register(
            service_name,
            catalog,
            cast(type[AtomCatalog], AtomCatalog),
            scope="host",
        )
        self._emit_register_event(
            "atom_catalog",
            service_name,
            {"service": catalog},
        )

    def get_atom_catalog(self) -> AtomCatalog | None:
        return self.services.get(
            ATOM_CATALOG_SERVICE,
            cast(type[AtomCatalog], AtomCatalog),
        )

    def _emit_register_event(
        self,
        kind: str,
        name: str,
        payload: dict[str, object],
    ) -> None:
        from agentm.core.runtime.extension import current_installing_extension

        self.bus.emit_sync(
            ApiRegisterEvent.CHANNEL,
            ApiRegisterEvent(
                kind=kind,
                name=name,
                extension=current_installing_extension(),
                payload=payload,
            ),
        )

    def _tool_allowlist(self) -> tuple[str, ...] | None:
        raw = self.services.get("tool_allowlist")
        if raw is None:
            return None
        if isinstance(raw, str):
            return (raw,)
        if isinstance(raw, Sequence):
            return tuple(str(item) for item in raw)
        logger.warning("tool_allowlist service has unsupported type {}", type(raw).__name__)
        return None

    def add_observer(self, observer: EventBusObserver) -> Callable[[], None]:
        """Register a bus observer for session-scoped instrumentation."""
        return self.bus.add_observer(observer)

    async def install_extension(
        self,
        module_path: str,
        config: dict[str, object] | None = None,
        *,
        trigger: str = "runtime",
    ) -> None:
        """Install an extension through the standard lifecycle path."""
        from agentm.core.runtime.extension import install_extension

        await install_extension(self, module_path, config or {}, trigger=trigger)

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
    def provider(self) -> ProviderConfig | None:
        """Active ProviderConfig, if any provider atom registered one."""
        return self.get_provider()

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
        cancel_signal: CancelSignal | None = None,
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
        child_services.inherit_from(self.services)
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
            codec=self.codec,
            max_turns=self._max_turns if max_turns is None else max_turns,
            max_tool_calls=self._max_tool_calls,
            tool_allowlist=self._tool_allowlist(),
            thinking=self._thinking,
            cancel_signal=cancel_signal,
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
        async def _on_child_shutdown(_: SessionShutdownEvent) -> None:
            await self.bus.emit(
                ChildSessionEndEvent.CHANNEL,
                ChildSessionEndEvent(
                    child_session_id=child.id,
                    parent_session_id=self.id,
                    final_message_count=len(child.get_messages()),
                    error=child._driver_error,
                ),
            )

        child.bus.on(SessionShutdownEvent.CHANNEL, _on_child_shutdown)

        if child.store is not None:
            await asyncio.to_thread(
                child.store.create_session,
                SessionMeta(
                    id=child.id,
                    parent_id=self.id,
                    purpose=purpose,
                    cwd=child.ctx.cwd,
                    created_at=time.time(),
                    config=session_meta_config(child.ctx),
                ),
            )

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

        if source.store is not None:
            await asyncio.to_thread(
                source.store.create_session_with_turns,
                SessionMeta(
                    id=child_ctx.session_id,
                    parent_id=source.id,
                    fork_point=at,
                    purpose=purpose,
                    cwd=source.ctx.cwd,
                    created_at=time.time(),
                    config=session_meta_config(child_ctx),
                ),
                prefix.turns,
            )

        child_services = ServiceRegistry()
        child_services.inherit_from(source.services)
        effect_scope = source.get_effect_scope()
        if effect_scope is not None:
            child_effect_scope = await effect_scope.fork_at(
                at,
                source_session_id=source.id,
                child_session_id=child_ctx.session_id,
            )
            child_services.register(
                EFFECT_SCOPE_SERVICE,
                child_effect_scope,
                cast(type[EffectScope], EffectScope),
                scope="tree",
            )

        forked = cls(
            ctx=child_ctx,
            trajectory=prefix,
            bus=EventBus(),
            store=source.store,
            graph=source.graph,
            stream_fn=source._stream_fn,
            model=source._model,
            tools=list(source.tools),
            system=source.system,
            context_policies=[copy.copy(p) for p in source.context_policies],
            trigger_renderers=dict(source.trigger_renderers),
            codec=source.codec,
            max_turns=source._max_turns,
            max_tool_calls=source._max_tool_calls,
            tool_allowlist=source._tool_allowlist(),
            thinking=source._thinking,
            services=child_services,
        )

        if forked.graph is not None:
            forked.graph.register(
                forked.id, parent_id=source.id,
                fork_point=at, purpose=purpose,
                edge_kind="forked",
            )

        return forked

    @classmethod
    async def resume(cls, session_id: str, store: TrajectoryStore, **kwargs: object) -> "Session":
        meta, turns = await asyncio.to_thread(store.load, session_id)
        trajectory = Trajectory(turns=turns)
        ctx = context_from_session_meta(session_id, meta)
        session = cls(ctx=ctx, trajectory=trajectory, store=store, **kwargs)  # type: ignore[arg-type]
        effect_scope = session.get_effect_scope()
        if effect_scope is not None:
            await effect_scope.restore(session_id=session.id, turns=tuple(turns))

        return session


AgentSession = Session

__all__ = ["AgentSession", "Session"]
