"""Session core lifecycle, registration, and composition layers.

Owns driver task, trajectory, trigger queue, bus, tools, services,
context policies, and shutdown logic.  Also the fork/resume/spawn
entry point.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Iterator, cast

from loguru import logger

from agentm.core.abi.cancel import (
    CancelReason,
    CancelSignal,
    EventCancelSource,
)
from agentm.core.abi.codec import (
    CodecBackedTrajectoryStore,
    CodecRegistry,
    TriggerCodec,
)
from agentm.core.abi.catalog import (
    ActiveSetFingerprint,
    AtomCatalog,
    AtomCatalogQuery,
    VersionedResourceStore,
)
from agentm.core.abi.lifecycle import (
    EffectScope,
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
from agentm.core.abi.permission import PermissionPolicy
from agentm.core.abi.permission import PermissionAudience
from agentm.core.abi.operations import EnvironmentOperations
from agentm.core.abi.provider import (
    ProviderConfig,
    ProviderResolver,
    ProviderSessionIdentity,
)
from agentm.core.abi.resource import (
    ResourceReader,
    ResourceStore,
    ResourceTxn,
    ResourceWriter,
)
from agentm.core.abi.stream import Model, StreamFn, ThinkingLevel
from agentm.core.abi.telemetry import SessionTelemetry
from agentm.core.abi.tool import Tool
from agentm.core.abi.tool_executor import ToolExecutor
from agentm.core.abi.tool_orchestration import ToolOrchestrator
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
    ATOM_CATALOG_SERVICE,
    ACTIVE_SET_FINGERPRINT_SERVICE,
    CATALOG_QUERY_SERVICE,
    BASH_OPERATIONS_SERVICE,
    EFFECT_SCOPE_SERVICE,
    ENVIRONMENT_RESTORE_FAILURE_HANDLER_SERVICE,
    ENVIRONMENT_RESTORE_STATUS_SERVICE,
    ENVIRONMENT_OPERATIONS_SERVICE,
    PERMISSION_POLICY_SERVICE,
    PROVIDER_RESOLVER_SERVICE,
    PROVIDER_SESSION_IDENTITY_SERVICE,
    RESOLVED_SESSION_SPEC_SERVICE,
    RESOURCE_READER_SERVICE,
    RESOURCE_STORE_SERVICE,
    RESOURCE_TXN_SERVICE,
    RESOURCE_WRITER_SERVICE,
    TOOL_EXECUTOR_SERVICE,
    TOOL_ORCHESTRATOR_SERVICE,
    TRAJECTORY_STORE_SERVICE,
    VERSIONED_RESOURCE_STORE_SERVICE,
)
from agentm.core.abi.session_api import (
    AgentSessionConfig,
    ExtensionSpec,
    ResolvedSessionSpec,
    SessionContext,
)
from agentm.core.abi.store import TrajectoryStore
from agentm.core.abi.trajectory import (
    Turn,
)
from agentm.core.abi.tree import SessionGraphProtocol
from agentm.core.abi.trigger import Trigger, TriggerPriority, TriggerRenderer, UserInput
from agentm.core.runtime.driver import DriverConfig, drive
from agentm.core.runtime.tool_orchestration import default_tool_orchestrator
from agentm.core.runtime.trajectory import Trajectory
from agentm.core.runtime.trigger_queue import TriggerQueue, TriggerReceipt

if TYPE_CHECKING:
    from agentm.core.runtime.session import Session


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
    provider_resolver: ProviderResolver | None = None
    tool_executor: ToolExecutor | None = None
    tool_orchestrator: ToolOrchestrator | None = None
    permission_policy: PermissionPolicy | None = None
    resource_reader: ResourceReader | None = None
    resource_store: ResourceStore | None = None
    versioned_resource_store: VersionedResourceStore | None = None
    environment_restore_failure_handler: EnvironmentRestoreFailureHandler | None = None
    provider_identity: ProviderSessionIdentity | None = None
    services: ServiceRegistry | None = None
    cwd: str = ""
    purpose: str = "root"


class _SessionLifecycle:
    """Session identity, driver lifecycle, triggers, and read-only state."""

    register_tool_executor: Callable[..., None]
    register_tool_orchestrator: Callable[..., None]
    register_permission_policy: Callable[..., None]
    register_resource_reader: Callable[..., None]
    register_resource_store: Callable[..., None]
    register_versioned_resource_store: Callable[..., None]
    _activate_provider: Callable[..., None]
    _freeze_provider_after_commits: Callable[..., None]
    get_provider: Callable[..., ProviderConfig | None]
    get_effect_scope: Callable[..., EffectScope | None]
    get_resource_writer: Callable[..., ResourceWriter | None]
    get_tool_executor: Callable[..., ToolExecutor | None]
    get_tool_orchestrator: Callable[..., ToolOrchestrator]
    get_permission_policy: Callable[..., PermissionPolicy | None]
    _tool_allowlist: Callable[..., tuple[str, ...] | None]

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
        self.codec = (
            codec
            if codec is not None
            else store_codec
            if isinstance(store_codec, CodecRegistry)
            else CodecRegistry()
        )
        self.services = ServiceRegistry() if services is None else services
        if store is not None:
            selected_store = self.services.get(
                TRAJECTORY_STORE_SERVICE,
                cast(type[TrajectoryStore], TrajectoryStore),
            )
            if selected_store is None:
                self.services.register(
                    TRAJECTORY_STORE_SERVICE,
                    store,
                    cast(type[TrajectoryStore], TrajectoryStore),
                    scope="tree",
                )
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
        self._cleanup_callbacks: list[Callable[[], Awaitable[None]]] = []
        self.installed_extensions: list[str] = []
        self._installed_extension_specs: list[ExtensionSpec] = []
        self._active_provider_name: str | None = None
        self._provider_owners: dict[str, str | None] = {}
        inherited_provider_identity = self.services.get(
            PROVIDER_SESSION_IDENTITY_SERVICE,
            ProviderSessionIdentity,
        )
        self._provider_identity: ProviderSessionIdentity | None = (
            runtime.provider_identity
            if runtime.provider_identity is not None
            else inherited_provider_identity
            if isinstance(inherited_provider_identity, ProviderSessionIdentity)
            else None
        )
        if self._provider_identity is not None:
            self.services.register(
                PROVIDER_SESSION_IDENTITY_SERVICE,
                self._provider_identity,
                ProviderSessionIdentity,
                scope="session",
            )
        if runtime.provider_resolver is not None:
            self.services.register(
                PROVIDER_RESOLVER_SERVICE,
                runtime.provider_resolver,
                scope="host",
            )
        if runtime.tool_executor is not None:
            self.register_tool_executor(runtime.tool_executor, replace=True)
        if runtime.tool_orchestrator is not None:
            self.register_tool_orchestrator(runtime.tool_orchestrator, replace=True)
        elif not self.services.has(TOOL_ORCHESTRATOR_SERVICE):
            self.register_tool_orchestrator(default_tool_orchestrator())
        if runtime.permission_policy is not None:
            self.register_permission_policy(runtime.permission_policy, replace=True)
        if runtime.resource_reader is not None:
            self.register_resource_reader(runtime.resource_reader, replace=True)
        if runtime.resource_store is not None:
            self.register_resource_store(runtime.resource_store, replace=True)
        if runtime.versioned_resource_store is not None:
            self.register_versioned_resource_store(
                runtime.versioned_resource_store,
                replace=True,
            )
        if runtime.environment_restore_failure_handler is not None:
            self.services.register(
                ENVIRONMENT_RESTORE_FAILURE_HANDLER_SERVICE,
                runtime.environment_restore_failure_handler,
                EnvironmentRestoreFailureHandler,
                scope="host",
            )
        if runtime.tool_allowlist is not None:
            self.services.register(
                "tool_allowlist",
                tuple(runtime.tool_allowlist),
                scope="session",
            )

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
        self._activate_provider()
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
                    permission_audience=cast(
                        PermissionAudience,
                        "user" if self.ctx.depth == 0 else "subagent",
                    ),
                    system=self.system,
                    context_policies=self.context_policies,
                    prompt_cache_adapter=(
                        provider.prompt_cache_adapter if provider is not None else None
                    ),
                    trigger_renderers=self.trigger_renderers,
                    interrupt=self._interrupt,
                    shutdown=self._shutdown,
                    cancel_signal=self._parent_cancel_signal,
                    effect_scope=self.get_effect_scope(),
                    resource_writer=self.get_resource_writer(),
                    services=self.services,
                    tool_executor=self.get_tool_executor(),
                    tool_orchestrator=self.get_tool_orchestrator(),
                    permission_policy=self.get_permission_policy(),
                    max_turns=self._max_turns,
                    max_tool_calls=self._max_tool_calls,
                    tool_allowlist=self._tool_allowlist(),
                    thinking=self._thinking,
                )
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
        cleanup_errors: list[BaseException] = []
        try:
            await self.bus.emit(SessionShutdownEvent.CHANNEL, SessionShutdownEvent())
        except BaseException as exc:
            cleanup_errors.append(exc)
        environment = self.services.get(
            ENVIRONMENT_OPERATIONS_SERVICE,
            cast(type[EnvironmentOperations], EnvironmentOperations),
        )
        if environment is not None:
            try:
                await environment.close()
            except BaseException as exc:
                cleanup_errors.append(exc)
        telemetry = self.services.get(
            "session_telemetry",
            cast(type[SessionTelemetry], SessionTelemetry),
        )
        if telemetry is not None:
            try:
                telemetry.shutdown()
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


class _SessionRegistration(_SessionLifecycle):
    """Extension-facing event, tool, context, and trigger registration."""

    _emit_register_event: Callable[..., None]

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


class _SessionProviders(_SessionRegistration):
    """Provider selection, activation, and durable identity binding."""

    _active_set_fingerprint: Callable[..., ActiveSetFingerprint | None]

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
        self.services.register(
            PROVIDER_SESSION_IDENTITY_SERVICE,
            identity,
            ProviderSessionIdentity,
            scope="session",
        )

    def provider_session_identity(self) -> ProviderSessionIdentity | None:
        """Return the provider/model identity bound to this session, if known."""

        self._freeze_provider_after_commits()
        if self._provider_identity is not None:
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
        return self.services.get(
            ENVIRONMENT_RESTORE_FAILURE_HANDLER_SERVICE,
            cast(
                type[EnvironmentRestoreFailureHandler],
                EnvironmentRestoreFailureHandler,
            ),
        )

    def _record_environment_restore_status(
        self,
        status: EnvironmentRestoreStatus,
    ) -> None:
        self.services.register(
            ENVIRONMENT_RESTORE_STATUS_SERVICE,
            status,
            EnvironmentRestoreStatus,
            scope="session",
        )

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


class _SessionResourceServices(_SessionProviders):
    """Resource, environment, and resolved-composition services."""

    _emit_register_event: Callable[..., None]

    def _resolved_session_spec(self) -> ResolvedSessionSpec | None:
        spec = self.services.get(RESOLVED_SESSION_SPEC_SERVICE)
        return spec if isinstance(spec, ResolvedSessionSpec) else None

    def _active_set_fingerprint(self) -> ActiveSetFingerprint | None:
        fingerprint = self.services.get(ACTIVE_SET_FINGERPRINT_SERVICE)
        return fingerprint if isinstance(fingerprint, ActiveSetFingerprint) else None

    def register_operations(
        self,
        *,
        replace: bool = False,
        service_scope: ServiceScope = "session",
        **kwargs: object,
    ) -> None:
        """Register named operation services."""
        from agentm.core.abi.operations import BashOperations

        service_names = {
            "bash": BASH_OPERATIONS_SERVICE,
            "environment": ENVIRONMENT_OPERATIONS_SERVICE,
        }
        protocols = {
            "bash": BashOperations,
            "environment": EnvironmentOperations,
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

    def register_resource_reader(
        self,
        reader: ResourceReader,
        *,
        replace: bool = False,
    ) -> None:
        service_name = RESOURCE_READER_SERVICE
        if self.services.has(service_name) and not replace:
            raise ValueError("resource_reader already registered")
        self.services.register(
            service_name,
            reader,
            ResourceReader,
            scope="resource",
        )
        self._emit_register_event(
            "resource_reader",
            service_name,
            {"service": reader},
        )

    def get_resource_reader(self) -> ResourceReader | None:
        return self.services.get(
            RESOURCE_READER_SERVICE,
            cast(type[ResourceReader], ResourceReader),
        )

    def register_resource_store(
        self,
        store: ResourceStore,
        *,
        replace: bool = False,
    ) -> None:
        service_name = RESOURCE_STORE_SERVICE
        previous = self.services.get(service_name)
        if self.services.has(service_name) and not replace:
            raise ValueError("resource_store already registered")
        self.services.register(
            service_name,
            store,
            ResourceStore,
            scope="resource",
        )
        reader = self.services.get(RESOURCE_READER_SERVICE)
        if reader is None:
            self.register_resource_reader(store)
        elif replace and reader is previous:
            self.register_resource_reader(store, replace=True)
        self._emit_register_event(
            "resource_store",
            service_name,
            {"service": store},
        )

    def get_resource_store(self) -> ResourceStore | None:
        return self.services.get(
            RESOURCE_STORE_SERVICE,
            cast(type[ResourceStore], ResourceStore),
        )

    def get_resource_txn(self) -> ResourceTxn | None:
        return self.services.get(
            RESOURCE_TXN_SERVICE,
            cast(type[ResourceTxn], ResourceTxn),
        )


class _SessionRuntimeServices(_SessionResourceServices):
    """Execution, permission, trajectory, effect, and catalog services."""

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

    def register_tool_orchestrator(
        self,
        orchestrator: ToolOrchestrator,
        *,
        replace: bool = False,
    ) -> None:
        service_name = TOOL_ORCHESTRATOR_SERVICE
        if self.services.has(service_name) and not replace:
            raise ValueError("tool_orchestrator already registered")
        self.services.register(
            service_name,
            orchestrator,
            cast(type[ToolOrchestrator], ToolOrchestrator),
            scope="host",
        )
        self._emit_register_event(
            "tool_orchestrator",
            service_name,
            {"service": orchestrator},
        )

    def get_tool_orchestrator(self) -> ToolOrchestrator:
        orchestrator = self.services.get(
            TOOL_ORCHESTRATOR_SERVICE,
            cast(type[ToolOrchestrator], ToolOrchestrator),
        )
        if orchestrator is None:
            raise RuntimeError("session has no tool orchestrator")
        return orchestrator

    def register_permission_policy(
        self,
        policy: PermissionPolicy,
        *,
        replace: bool = False,
    ) -> None:
        service_name = PERMISSION_POLICY_SERVICE
        if self.services.has(service_name) and not replace:
            raise ValueError("permission_policy already registered")
        self.services.register(
            service_name,
            policy,
            cast(type[PermissionPolicy], PermissionPolicy),
            scope="host",
        )
        self._emit_register_event(
            "permission_policy",
            service_name,
            {"service": policy},
        )

    def get_permission_policy(self) -> PermissionPolicy | None:
        return self.services.get(
            PERMISSION_POLICY_SERVICE,
            cast(type[PermissionPolicy], PermissionPolicy),
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
        if isinstance(catalog, AtomCatalogQuery):
            self.services.register(
                CATALOG_QUERY_SERVICE,
                catalog,
                AtomCatalogQuery,
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

    def add_observer(self, observer: EventBusObserver) -> Callable[[], None]:
        """Register a bus observer for session-scoped instrumentation."""
        return self.bus.add_observer(observer)


class _SessionComposition(_SessionRuntimeServices):
    """Installed extension ownership and inherited composition metadata."""

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

    def _record_installed_extension(
        self,
        spec: ExtensionSpec,
    ) -> None:
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
        value = self.services.get("experiment")
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise TypeError("experiment service must be a JSON object")
        frozen = freeze_json(value)
        if not isinstance(frozen, Mapping):
            raise TypeError("experiment service must be a JSON object")
        return dict(frozen)


__all__ = ["SessionRuntimeConfig"]
