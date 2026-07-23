# code-health: ignore-file[AM025] -- runtime composes plugin, service, and trajectory boundary values
"""Session runtime — lifecycle, registration, providers, and composition.

Owns driver task, trajectory, trigger queue, bus, tools, services,
context policies, and shutdown logic.  ``Session`` (session.py) extends
this with child, fork, and resume operations.

Runtime boundaries (resource ports, tool execution, permission, effect
scope, catalogs) are plain service-role bindings; there are no
per-boundary register/get methods.
"""

from __future__ import annotations

import asyncio
import copy
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Iterator

from loguru import logger

from agentm.core.abi.cancel import (
    CancelReason,
    CancelSignal,
    CompositeCancelSignal,
    EventCancelSource,
)
from agentm.core.abi.codec import (
    CodecBackedTrajectoryStore,
    CodecRegistry,
    TriggerCodec,
)
from agentm.core.abi.catalog import (
    ActiveSetFingerprint,
)
from agentm.core.abi.lifecycle import (
    EnvironmentRestoreFailureHandler,
    EnvironmentRestoreStatus,
)
from agentm.core.abi.messages import (
    AgentMessage,
    ImageContent,
    JsonValue,
    TextContent,
    freeze_json,
)
from agentm.core.abi.operations import BashOperations, EnvironmentOperations
from agentm.core.abi.permission import PermissionAudience
from agentm.core.abi.provider import (
    ProviderConfig,
    ProviderResolver,
    ProviderSessionIdentity,
)
from agentm.core.abi.stream import Model, StreamFn, ThinkingLevel
from agentm.core.abi.tool import Tool
from agentm.core.abi.bus import EventBus, EventBusObserver, Handler
from agentm.core.abi.context import (
    BindableContextPolicy,
    ContextPolicy,
    PolicyContext,
    build_context_sync,
)
from agentm.core.abi.events import (
    ApiRegisterEvent,
    SessionReadyEvent,
    SessionShutdownEvent,
    TurnCommittedEvent,
)
from agentm.core.abi.services import ServiceRegistry, ServiceScope
from agentm.core.abi.roles import (
    ACTIVE_SET_FINGERPRINT_ROLE,
    BASH_OPERATIONS_ROLE,
    CONTEXT_COMPACTION,
    CONTEXT_COMPACTION_SERVICE,
    EFFECT_SCOPE_ROLE,
    ENVIRONMENT_OPERATIONS,
    ENVIRONMENT_RESTORE_FAILURE_HANDLER,
    ENVIRONMENT_RESTORE_STATUS_ROLE,
    EXPERIMENT_SERVICE,
    PERMISSION_POLICY_ROLE,
    PROVIDER_RESOLVER_SERVICE,
    PROVIDER_SESSION_IDENTITY,
    RESOLVED_SESSION_SPEC_SERVICE,
    RESOURCE_WRITER,
    SESSION_TELEMETRY_ROLE,
    TOOL_ALLOWLIST_SERVICE,
    TOOL_EXECUTOR,
    TOOL_ORCHESTRATOR,
    TRAJECTORY_STORE_ROLE,
)
from agentm.core.abi.session_api import (
    ExtensionSpec,
    ResolvedSessionSpec,
    SessionContext,
)
from agentm.core.abi.store import TrajectoryStore
from agentm.core.abi.trajectory import (
    Turn,
)
from agentm.core.abi.tree import SessionGraphProtocol
from agentm.core.abi.trigger import (
    Trigger,
    TriggerPriority,
    TriggerRenderer,
    UserInput,
)
from agentm.core.lib.async_cancel import await_known_outcome
from agentm.core.runtime.driver import DriverConfig, drive
from agentm.core.runtime.tool_orchestration import default_tool_orchestrator
from agentm.core.runtime.trajectory import Trajectory
from agentm.core.runtime.trigger_queue import TriggerQueue, TriggerReceipt


@dataclass(slots=True)
class SessionRuntimeConfig:
    """Low-level runtime dependencies after factory composition is resolved."""

    ctx: SessionContext | None = None
    session_id: str | None = None
    trajectory: Trajectory | None = None
    bus: EventBus | None = None
    store: TrajectoryStore | None = None
    graph: SessionGraphProtocol | None = None
    stream_fn: StreamFn | None = None
    model: Model | None = None
    tools: list[Tool] = field(default_factory=list)
    system: str | None = None
    context_policies: list[ContextPolicy] = field(default_factory=list)
    trigger_renderers: dict[str, TriggerRenderer] = field(default_factory=dict)
    codec: CodecRegistry | None = None
    max_turns: int | None = None
    max_tool_calls: int | None = None
    tool_allowlist: Sequence[str] | None = None
    thinking: ThinkingLevel = "off"
    cancel_signal: CancelSignal | None = None
    provider_identity: ProviderSessionIdentity | None = None
    services: ServiceRegistry | None = None
    cwd: str = ""
    purpose: str = "root"

    # Capability boundaries (resource ports, tool execution, permission,
    # effect scope, catalogs, provider resolver) are NOT fields here: they are
    # bound into ``services`` by the factory before construction. The registry
    # is the single representation of a session's boundaries.


@dataclass(frozen=True, slots=True)
class CompositionSnapshot:
    """Typed view of a session's rebuildable composition.

    The formal surface for spawn/fork/child construction — factories consume
    this instead of reading session privates.
    """

    extensions: tuple[ExtensionSpec, ...]
    external_tools: tuple[Tool, ...]
    external_context_policies: tuple[ContextPolicy, ...]
    external_trigger_renderers: dict[str, TriggerRenderer]
    codec: CodecRegistry
    stream_fn: StreamFn | None
    model: Model | None
    system: str | None
    max_turns: int | None
    max_tool_calls: int | None
    tool_allowlist: tuple[str, ...] | None
    thinking: ThinkingLevel
    lineage_cancel: CancelSignal


class SessionRuntime:
    """Single-session runtime: identity, driver, registration, providers."""

    def __init__(self, config: SessionRuntimeConfig | None = None) -> None:
        runtime = SessionRuntimeConfig() if config is None else config
        ctx = runtime.ctx
        session_id = runtime.session_id
        trajectory = runtime.trajectory
        bus = runtime.bus
        store = runtime.store
        graph = runtime.graph
        tools = runtime.tools
        context_policies = runtime.context_policies
        trigger_renderers = runtime.trigger_renderers
        codec = runtime.codec
        services = runtime.services
        sid = session_id or uuid.uuid4().hex[:16]
        if ctx is None:
            self.ctx = SessionContext(
                session_id=sid,
                root_session_id=sid,
                cwd=runtime.cwd,
                purpose=runtime.purpose,
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
            self.trajectory = Trajectory()
        self.bus = EventBus() if bus is None else bus
        self.store = store
        self.graph = graph
        self.triggers = TriggerQueue()
        self.tools: list[Tool] = list(tools or [])
        self._tool_owners: dict[int, str | None] = {
            id(tool): None for tool in self.tools
        }
        self.system = runtime.system
        self.context_policies: list[ContextPolicy] = list(context_policies or [])
        self._context_policy_owners: dict[int, str | None] = {
            id(policy): None for policy in self.context_policies
        }
        self._context_policy_priorities: dict[int, int] = {
            id(policy): 500 for policy in self.context_policies
        }
        self.trigger_renderers: dict[str, TriggerRenderer] = dict(
            trigger_renderers or {}
        )
        self._trigger_renderer_owners: dict[str, str | None] = {
            source: None for source in self.trigger_renderers
        }
        self._trigger_codec_owners: dict[str, str | None] = {}
        store_codec = (
            store.codec if isinstance(store, CodecBackedTrajectoryStore) else None
        )
        if codec is not None:
            self.codec = codec
        elif isinstance(store_codec, CodecRegistry):
            self.codec = store_codec
        else:
            self.codec = CodecRegistry()
        self.services = ServiceRegistry() if services is None else services
        self.services.set_bind_observer(self._on_service_bind)
        if store is not None:
            selected_store = self.services.get_role(TRAJECTORY_STORE_ROLE)
            if selected_store is None:
                self.services.bind(TRAJECTORY_STORE_ROLE, store)
            elif selected_store is not store:
                raise ValueError(
                    "session services contain a different trajectory store"
                )

        self._stream_fn = runtime.stream_fn
        self._model = runtime.model
        self._max_turns = runtime.max_turns
        self._max_tool_calls = runtime.max_tool_calls
        self._thinking = runtime.thinking
        self._parent_cancel_signal = runtime.cancel_signal
        self._interrupt = EventCancelSource()
        self._shutdown = EventCancelSource()
        self._closed = False
        self._driver_error: str | None = None
        self._driver_task: asyncio.Task[None] | None = None
        self._shutdown_task: asyncio.Task[None] | None = None
        self._cleanup_callbacks: list[Callable[[], Awaitable[None]]] = []
        self.installed_extensions: list[str] = []
        self._installed_extension_specs: list[ExtensionSpec] = []
        self._active_provider_name: str | None = None
        self._provider_owners: dict[str, str | None] = {}
        inherited_provider_identity = self.services.get_role(PROVIDER_SESSION_IDENTITY)
        self._provider_identity: ProviderSessionIdentity | None = (
            runtime.provider_identity
            if runtime.provider_identity is not None
            else inherited_provider_identity
        )
        if self._provider_identity is not None:
            self.services.bind(
                PROVIDER_SESSION_IDENTITY,
                self._provider_identity,
                replace=True,
            )
        if self.services.get_role(TOOL_ORCHESTRATOR) is None:
            self.services.bind(TOOL_ORCHESTRATOR, default_tool_orchestrator())
        if runtime.tool_allowlist is not None:
            self.services.register(
                TOOL_ALLOWLIST_SERVICE,
                tuple(runtime.tool_allowlist),
                scope="session",
            )

        if self.graph is not None and self.ctx.parent_session_id is None:
            self.graph.register(
                self.id,
                purpose=self.ctx.purpose,
            )

    # --- Lifecycle ---

    def start(self) -> None:
        if self._driver_task is not None:
            return
        self._activate_provider()
        if self._stream_fn is None:
            raise RuntimeError(f"session {self.id}: cannot start without stream_fn")
        if self._model is None:
            raise RuntimeError(f"session {self.id}: cannot start without model")

        policy_ctx = PolicyContext(
            session_id=self.id,
            parent_session_id=self.ctx.parent_session_id,
            services=self.services,
            store=self.store,
            model=self._model,
            stream_fn=self._stream_fn,
            trigger_renderers=dict(self.trigger_renderers),
        )
        for policy in self.context_policies:
            if isinstance(policy, BindableContextPolicy):
                policy.bind(policy_ctx)

        self.bus.on(
            TurnCommittedEvent.CHANNEL,
            self._freeze_provider_on_turn_commit,
            owner="agentm.core.session",
        )
        self.bus.freeze_clear()
        self._driver_task = asyncio.create_task(
            self._run_driver(),
            name=f"v2-driver-{self.id}",
        )
        self.bus.emit_sync(
            SessionReadyEvent.CHANNEL,
            SessionReadyEvent(
                session_id=self.id,
                root_session_id=self.ctx.root_session_id,
                parent_session_id=self.ctx.parent_session_id,
                cwd=self.ctx.cwd,
                tool_names=tuple(t.name for t in self.tools),
                extension_module_paths=tuple(self.installed_extensions),
                model=self._model,
            ),
        )

    def _freeze_provider_on_turn_commit(self, _: TurnCommittedEvent) -> None:
        self._freeze_provider_after_commits()

    async def _run_driver(self) -> None:
        try:
            assert self._stream_fn is not None
            assert self._model is not None
            provider = self.get_provider()
            audience: PermissionAudience = "user" if self.ctx.depth == 0 else "subagent"
            await drive(
                DriverConfig(
                    trajectory=self.trajectory,
                    triggers=self.triggers,
                    bus=self.bus,
                    stream_fn=self._stream_fn,
                    model=self._model,
                    tools=self.tools,
                    store=self.store,
                    session_id=self.id,
                    root_session_id=self.ctx.root_session_id,
                    parent_session_id=self.ctx.parent_session_id,
                    permission_audience=audience,
                    system=self.system,
                    context_policies=self.context_policies,
                    prompt_cache_adapter=(
                        provider.prompt_cache_adapter if provider is not None else None
                    ),
                    trigger_renderers=self.trigger_renderers,
                    interrupt=self._interrupt,
                    shutdown=self._shutdown,
                    cancel_signal=self._parent_cancel_signal,
                    effect_scope=self.services.get_role(EFFECT_SCOPE_ROLE),
                    resource_writer=self.services.get_role(RESOURCE_WRITER),
                    services=self.services,
                    tool_executor=self.services.get_role(TOOL_EXECUTOR),
                    tool_orchestrator=self.services.require_role(TOOL_ORCHESTRATOR),
                    permission_policy=self.services.get_role(PERMISSION_POLICY_ROLE),
                    max_turns=self._max_turns,
                    max_tool_calls=self._max_tool_calls,
                    tool_allowlist=self._tool_allowlist(),
                    thinking=self._thinking,
                )
            )
        except asyncio.CancelledError:
            logger.debug("session {} driver cancelled", self.id)
        except Exception as exc:
            self._driver_error = str(exc)
            logger.exception("session {} driver crashed", self.id)

    async def shutdown(self) -> None:
        shutdown_task = self._shutdown_task
        if shutdown_task is None:
            self._closed = True
            self._shutdown.set("shutdown")
            self.triggers.close()
            shutdown_task = asyncio.create_task(
                self._shutdown_once(),
                name=f"agentm-shutdown-{self.id}",
            )
            self._shutdown_task = shutdown_task
        if shutdown_task is asyncio.current_task():
            return
        await await_known_outcome(shutdown_task)

    async def _shutdown_once(self) -> None:
        if self._driver_task is not None:
            try:
                await asyncio.wait_for(self._driver_task, timeout=30.0)
            except TimeoutError:
                logger.warning(
                    "session {} driver did not stop within 30s, force-cancelling",
                    self.id,
                )
                self._driver_task.cancel()
                try:
                    await self._driver_task
                except (asyncio.CancelledError, Exception) as exc:
                    logger.debug("session {} driver post-cancel: {}", self.id, exc)
            except asyncio.CancelledError:
                logger.debug("session {} shutdown waiter cancelled", self.id)
        cleanup_errors: list[BaseException] = []
        try:
            await self.bus.emit(SessionShutdownEvent.CHANNEL, SessionShutdownEvent())
        except BaseException as exc:
            cleanup_errors.append(exc)
        environment = self.services.get_role(ENVIRONMENT_OPERATIONS)
        if environment is not None:
            try:
                await environment.close()
            except BaseException as exc:
                cleanup_errors.append(exc)
        telemetry = self.services.get_role(SESSION_TELEMETRY_ROLE)
        if telemetry is not None:
            try:
                await asyncio.to_thread(telemetry.shutdown)
            except BaseException as exc:
                cleanup_errors.append(exc)
        for callback in reversed(self._cleanup_callbacks):
            try:
                await callback()
            except BaseException as exc:
                cleanup_errors.append(exc)
        self._cleanup_callbacks.clear()
        self.bus._force_clear()
        if cleanup_errors:
            raise BaseExceptionGroup("session shutdown cleanup failed", cleanup_errors)

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
        meta: dict[str, JsonValue] | None = None,
    ) -> TriggerReceipt[object]:
        for label, target in (
            ("target_session_id", target_session_id),
            ("target_agent_id", target_agent_id),
        ):
            if target is not None and target != self.id:
                raise ValueError(
                    f"{label}={target!r} does not address session {self.id!r}; "
                    "route to the target session before pushing"
                )
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

    def compact(self) -> None:
        """Schedule compaction after the active step without interrupting it."""

        if self._closed:
            raise RuntimeError("cannot compact a closed session")
        compaction = self.services.get_role(CONTEXT_COMPACTION)
        if compaction is None:
            raise RuntimeError(
                "no ContextCompactionService registered "
                f"(service {CONTEXT_COMPACTION_SERVICE!r})"
            )
        if self._driver_task is None:
            self.start()
        compaction.request()

    def register_cleanup(
        self,
        callback: Callable[[], Awaitable[None]],
    ) -> None:
        """Register a presenter-owned async cleanup for session shutdown."""

        if self._closed:
            raise RuntimeError("cannot register cleanup on a closed session")
        self._cleanup_callbacks.append(callback)

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

    def add_observer(self, observer: EventBusObserver) -> Callable[[], None]:
        """Register a bus observer for session-scoped instrumentation."""
        return self.bus.add_observer(observer)

    # --- Registration ---

    def register_tool(self, tool: Tool) -> None:
        from agentm.core.runtime.extension import current_installing_extension

        existing = {t.name for t in self.tools}
        if tool.name in existing:
            raise ValueError(f"duplicate tool: {tool.name}")
        self.tools.append(tool)
        self._tool_owners[id(tool)] = current_installing_extension() or None
        self._emit_register_event("tool", tool.name, {"tool": tool})

    def register_context_policy(
        self, policy: ContextPolicy, *, priority: int = 500
    ) -> None:
        from agentm.core.runtime.extension import current_installing_extension

        self.context_policies.append(policy)
        self._context_policy_priorities[id(policy)] = priority
        self._context_policy_owners[id(policy)] = current_installing_extension() or None
        self.context_policies.sort(
            key=lambda item: self._context_policy_priorities[id(item)]
        )
        self._emit_register_event(
            "context_policy",
            type(policy).__name__,
            {"policy": policy, "priority": priority},
        )

    def register_trigger_renderer(self, source: str, renderer: TriggerRenderer) -> None:
        from agentm.core.runtime.extension import current_installing_extension

        self.trigger_renderers[source] = renderer
        self._trigger_renderer_owners[source] = current_installing_extension() or None
        self._emit_register_event(
            "trigger_renderer",
            source,
            {"renderer": renderer},
        )

    def register_trigger_codec(self, source: str, codec: object) -> None:
        from agentm.core.runtime.extension import current_installing_extension

        if not isinstance(codec, TriggerCodec):
            raise TypeError("trigger codec must implement serialize and deserialize")
        self.codec.register_trigger_codec(source, codec)
        self._trigger_codec_owners[source] = current_installing_extension() or None
        self._emit_register_event(
            "trigger_codec",
            source,
            {"codec": codec},
        )

    def register_operations(
        self,
        *,
        replace: bool = False,
        service_scope: ServiceScope = "session",
        **kwargs: object,
    ) -> None:
        """Register named operation services."""

        protocols: dict[str, type | None] = {
            "bash": BashOperations,
            "environment": EnvironmentOperations,
        }
        service_names = {
            "bash": BASH_OPERATIONS_ROLE.key,
            "environment": ENVIRONMENT_OPERATIONS.key,
        }
        for key, value in kwargs.items():
            service_name = service_names.get(key, f"operations:{key}")
            if self.services.has(service_name):
                if not replace:
                    raise ValueError(f"operation {key!r} already registered")
                self.services.unregister(service_name)
            self.services.register(
                service_name,
                value,
                protocols.get(key),
                scope=service_scope,
            )
            self._emit_register_event(
                "operations",
                key,
                {"service_name": service_name, "service": value},
            )

    def _on_service_bind(
        self,
        key: str,
        service: object,
        scope: ServiceScope,
    ) -> None:
        self._emit_register_event(
            "service",
            key,
            {"service": service, "scope": scope},
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

    # --- Providers ---

    def register_provider(
        self,
        name: str,
        config: ProviderConfig,
        *,
        replace: bool = False,
    ) -> None:
        """Register an LLM provider and refresh the active provider."""
        if not isinstance(name, str) or not name:
            raise ValueError("provider registry name must be a non-empty string")
        if not isinstance(config, ProviderConfig):
            raise TypeError("provider config must be ProviderConfig")
        if config.name != name:
            raise ValueError(
                f"provider registry name {name!r} does not match "
                f"ProviderConfig.name {config.name!r}"
            )
        key = f"provider:{name}"
        previous = self.services.get(key)
        if previous is not None and not replace:
            raise ValueError(f"provider {name!r} already registered")
        prospective = self._provider_configs()
        prospective[name] = config
        if self._provider_identity is None:
            self._resolve_provider_name(prospective)
        elif (
            self._provider_identity.name == name
            and self._provider_identity.model_id is not None
            and config.model.id != self._provider_identity.model_id
        ):
            raise RuntimeError(
                "cannot replace the session-bound provider with model "
                f"{config.model.id!r}; expected "
                f"{self._provider_identity.model_id!r}"
            )
        from agentm.core.runtime.extension import current_installing_extension

        previous_owner = self._provider_owners.get(name)
        self.services.register(key, config, scope="session")
        self._provider_owners[name] = current_installing_extension() or None
        try:
            self._activate_provider()
        except BaseException:
            if previous is None:
                self.services.unregister(key)
                self._provider_owners.pop(name, None)
            else:
                self.services.register(key, previous, scope="session")
                self._provider_owners[name] = previous_owner
            raise
        self._emit_register_event("provider", name, {"provider": config})

    def has_provider(self, name: str) -> bool:
        return self.services.get(f"provider:{name}") is not None

    def get_provider(self, name: str | None = None) -> ProviderConfig | None:
        if name is None:
            self._activate_provider()
        provider_name = name or self._active_provider_name
        if provider_name is None:
            return None
        provider = self.services.get(f"provider:{provider_name}")
        return provider if isinstance(provider, ProviderConfig) else None

    def provider_names(self) -> list[str]:
        prefix = "provider:"
        return sorted(
            name[len(prefix) :]
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
                providers[service_name[len(prefix) :]] = provider
        return providers

    def _provider_resolver(self) -> ProviderResolver | None:
        candidate = self.services.get(PROVIDER_RESOLVER_SERVICE)
        return candidate if isinstance(candidate, ProviderResolver) else None

    def _activate_provider(self) -> None:
        providers = self._provider_configs()
        if not providers:
            if (
                self._provider_identity is not None
                and self._model is not None
                and self._provider_identity.model_id is not None
                and self._model.id != self._provider_identity.model_id
            ):
                raise RuntimeError(
                    "cannot activate provider: session is bound to model "
                    f"{self._provider_identity.model_id!r}, got "
                    f"{self._model.id!r}"
                )
            return
        self._freeze_provider_after_commits()
        if self._provider_identity is not None:
            provider = providers.get(self._provider_identity.name)
            if provider is None:
                if not self.trajectory.turns:
                    return
                raise RuntimeError(
                    "cannot activate provider: session is bound to provider "
                    f"{self._provider_identity.name!r}, but it is not registered"
                )
            model_id = provider.model.id
            if (
                self._provider_identity.model_id is not None
                and model_id != self._provider_identity.model_id
            ):
                raise RuntimeError(
                    "cannot activate provider: session is bound to model "
                    f"{self._provider_identity.model_id!r}, got {model_id!r}"
                )
            self._active_provider_name = self._provider_identity.name
            self._stream_fn = provider.stream_fn
            self._model = provider.model
            return
        selected = self._resolve_provider_name(providers)
        if selected is None:
            return
        provider = providers[selected]
        self._active_provider_name = selected
        self._stream_fn = provider.stream_fn
        self._model = provider.model

    def _freeze_provider_after_commits(self) -> None:
        if self._provider_identity is not None or not self.trajectory.turns:
            return
        providers = self._provider_configs()
        first_model_id = next(
            (
                turn.meta.model_id
                for turn in self.trajectory.turns
                if turn.meta.model_id
            ),
            None,
        )
        if not providers:
            if self._model is None:
                return
            active_set = self._active_set_fingerprint()
            self._set_provider_identity(
                ProviderSessionIdentity(
                    name=self._active_provider_name or "direct",
                    model_id=first_model_id or self._model.id,
                    active_set_digest=(
                        active_set.digest if active_set is not None else None
                    ),
                    frozen_after_turn_index=self.trajectory.turns[0].index,
                )
            )
            return
        provider_name = self._active_provider_name
        if provider_name is None or provider_name not in providers:
            raise RuntimeError(
                "cannot freeze provider identity: committed turns have no "
                "active registered provider"
            )
        provider = providers[provider_name]
        if first_model_id is not None and provider.model.id != first_model_id:
            raise RuntimeError(
                "cannot freeze provider identity: committed model "
                f"{first_model_id!r} does not match selected provider model "
                f"{provider.model.id!r}"
            )
        active_set = self._active_set_fingerprint()
        self._set_provider_identity(
            ProviderSessionIdentity(
                name=provider_name,
                model_id=first_model_id or provider.model.id,
                active_set_digest=active_set.digest if active_set is not None else None,
                frozen_after_turn_index=self.trajectory.turns[0].index,
            )
        )

    def _set_provider_identity(self, identity: ProviderSessionIdentity) -> None:
        self._provider_identity = identity
        self.services.bind(PROVIDER_SESSION_IDENTITY, identity, replace=True)

    def provider_session_identity(self) -> ProviderSessionIdentity | None:
        """Return the provider/model identity bound to this session, if known."""

        self._freeze_provider_after_commits()
        if self._provider_identity is not None:
            active_set = self._active_set_fingerprint()
            if active_set is not None:
                bound_digest = self._provider_identity.active_set_digest
                if bound_digest is None:
                    self._set_provider_identity(
                        replace(
                            self._provider_identity,
                            active_set_digest=active_set.digest,
                        )
                    )
                elif bound_digest != active_set.digest:
                    raise RuntimeError(
                        "provider identity active set does not match the session: "
                        f"{bound_digest} != {active_set.digest}"
                    )
            return self._provider_identity
        if self._active_provider_name is None:
            self._activate_provider()
        if self._model is None:
            return None
        active_set = self._active_set_fingerprint()
        return ProviderSessionIdentity(
            name=self._active_provider_name or "direct",
            model_id=self._model.id,
            active_set_digest=active_set.digest if active_set is not None else None,
        )

    def _environment_restore_failure_handler(
        self,
    ) -> EnvironmentRestoreFailureHandler | None:
        return self.services.get_role(ENVIRONMENT_RESTORE_FAILURE_HANDLER)

    def _record_environment_restore_status(
        self,
        status: EnvironmentRestoreStatus,
    ) -> None:
        self.services.bind(ENVIRONMENT_RESTORE_STATUS_ROLE, status, replace=True)

    def _resolve_provider_name(
        self,
        providers: dict[str, ProviderConfig],
    ) -> str:
        if not providers:
            raise LookupError("cannot resolve an empty provider registry")
        resolver = self._provider_resolver()
        if resolver is not None:
            selected = resolver.resolve_provider(providers)
            if selected is None:
                raise LookupError(
                    "provider resolver returned None for a non-empty registry"
                )
            if selected not in providers:
                raise LookupError(
                    f"provider resolver selected unregistered provider {selected!r}"
                )
            return selected
        if len(providers) == 1:
            return next(iter(providers))
        raise RuntimeError(
            "multiple providers are registered; configure a ProviderResolver "
            "instead of relying on registration order"
        )

    # --- Resolved composition metadata ---

    def _resolved_session_spec(self) -> ResolvedSessionSpec | None:
        spec = self.services.get(RESOLVED_SESSION_SPEC_SERVICE)
        return spec if isinstance(spec, ResolvedSessionSpec) else None

    def _active_set_fingerprint(self) -> ActiveSetFingerprint | None:
        return self.services.get_role(ACTIVE_SET_FINGERPRINT_ROLE)

    def _tool_allowlist(self) -> tuple[str, ...] | None:
        raw = self.services.get(TOOL_ALLOWLIST_SERVICE)
        if raw is None:
            return None
        if isinstance(raw, str):
            if not raw:
                raise ValueError("tool_allowlist entries must be non-empty strings")
            return (raw,)
        if isinstance(raw, Sequence):
            items = tuple(raw)
            if not all(isinstance(item, str) and item for item in items):
                raise TypeError("tool_allowlist service must contain non-empty strings")
            return items
        raise TypeError(
            "tool_allowlist service must be a string or sequence, got "
            f"{type(raw).__name__}"
        )

    # --- Composition ---

    async def install_extension(
        self,
        extension: ExtensionSpec | str,
        config: dict[str, object] | None = None,
        *,
        trigger: str = "runtime",
    ) -> None:
        """Install an extension through the standard lifecycle path."""
        from agentm.core.runtime.extension import install_extension

        await install_extension(
            self,
            extension,
            None if isinstance(extension, ExtensionSpec) else config or {},
            trigger=trigger,
        )

    def record_installed_extension(
        self,
        spec: ExtensionSpec,
    ) -> None:
        """Record one installed extension for composition snapshots."""

        if not isinstance(spec, ExtensionSpec):
            raise TypeError("installed extension record requires ExtensionSpec")
        self.installed_extensions.append(spec.module_path)
        self._installed_extension_specs.append(
            ExtensionSpec(source=spec.source, config=spec.config)
        )

    def _external_tools(self) -> list[Tool]:
        return [tool for tool in self.tools if self._tool_owners.get(id(tool)) is None]

    def _external_context_policies(self) -> list[ContextPolicy]:
        return [
            policy
            for policy in self.context_policies
            if self._context_policy_owners.get(id(policy)) is None
        ]

    def _external_trigger_renderers(self) -> dict[str, TriggerRenderer]:
        return {
            source: renderer
            for source, renderer in self.trigger_renderers.items()
            if self._trigger_renderer_owners.get(source) is None
        }

    def _composition_codec(self) -> CodecRegistry:
        return self.codec.copy_without_trigger_sources(
            {
                source
                for source, owner in self._trigger_codec_owners.items()
                if owner is not None
            }
        )

    def _composition_extensions(
        self,
        *,
        include_provider_atoms: bool,
    ) -> list[ExtensionSpec]:
        excluded = (
            set()
            if include_provider_atoms
            else {
                owner for owner in self._provider_owners.values() if owner is not None
            }
        )
        return [
            ExtensionSpec(source=spec.source, config=spec.config)
            for spec in self._installed_extension_specs
            if spec.module_path not in excluded
        ]

    def composition_snapshot(
        self,
        *,
        include_provider_atoms: bool = True,
    ) -> CompositionSnapshot:
        """Snapshot the rebuildable composition for spawn/fork/child paths."""

        return CompositionSnapshot(
            extensions=tuple(
                self._composition_extensions(
                    include_provider_atoms=include_provider_atoms,
                )
            ),
            external_tools=tuple(self._external_tools()),
            external_context_policies=tuple(
                copy.copy(policy) for policy in self._external_context_policies()
            ),
            external_trigger_renderers=self._external_trigger_renderers(),
            codec=self._composition_codec(),
            stream_fn=self._stream_fn,
            model=self._model,
            system=self.system,
            max_turns=self._max_turns,
            max_tool_calls=self._max_tool_calls,
            tool_allowlist=self._tool_allowlist(),
            thinking=self._thinking,
            lineage_cancel=CompositeCancelSignal(
                self._interrupt,
                self._shutdown,
                self._parent_cancel_signal,
            ),
        )

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
    def experiment(self) -> dict[str, JsonValue] | None:
        value = self.services.get(EXPERIMENT_SERVICE)
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise TypeError("experiment service must be a JSON object")
        frozen = freeze_json(value)
        if not isinstance(frozen, Mapping):
            raise TypeError("experiment service must be a JSON object")
        return dict(frozen)


__all__ = ["CompositionSnapshot", "SessionRuntime", "SessionRuntimeConfig"]
