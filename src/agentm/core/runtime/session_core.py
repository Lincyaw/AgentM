# code-health: ignore-file[AM025] -- runtime composes plugin, service, and trajectory boundary values
"""Session runtime — lifecycle, registration, providers, and composition.

Owns driver task, trajectory, trigger queue, bus, tools, services,
context policies, and shutdown logic.  ``Session`` (session.py) extends
this with child, fork, and resume operations.

Two collaborators own the state the runtime only routes to: which model
the session talks to lives in ``provider_registry.py``, and which atom
registered what lives in ``extension_install.py``.

Runtime boundaries (resource ports, tool execution, permission, effect
scope, catalogs) are plain service-role bindings; there are no
per-boundary register/get methods.
"""

from __future__ import annotations

import asyncio
import copy
import uuid
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, cast

from loguru import logger

from agentm.core.abi.bus import EventBus, EventBusObserver, Handler
from agentm.core.abi.cancel import (
    CancelReason,
    CancelSignal,
    CompositeCancelSignal,
    EventCancelSource,
)
from agentm.core.abi.catalog import (
    ActiveSetFingerprint,
)
from agentm.core.abi.codec import (
    CodecBackedTrajectoryStore,
    CodecRegistry,
    TriggerCodec,
)
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
from agentm.core.abi.manifest import requirement_key
from agentm.core.abi.operations import BashOperations, EnvironmentOperations
from agentm.core.abi.permission import PermissionAudience
from agentm.core.abi.provider import (
    ProviderConfig,
    ProviderSessionIdentity,
)
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
    RESOLVED_SESSION_SPEC_SERVICE,
    RESOURCE_WRITER,
    SESSION_TELEMETRY_ROLE,
    TOOL_ALLOWLIST_SERVICE,
    TOOL_EXECUTOR,
    TOOL_ORCHESTRATOR,
    TRAJECTORY_STORE_ROLE,
)
from agentm.core.abi.services import ServiceRegistry, ServiceScope
from agentm.core.abi.session_api import (
    ExtensionSpec,
    ResolvedSessionSpec,
    SessionContext,
    SessionResult,
)
from agentm.core.abi.store import TrajectoryStore
from agentm.core.abi.stream import Model, StreamFn, ThinkingLevel
from agentm.core.abi.tool import Tool
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
from agentm.core.lib.session_result import compute_session_result
from agentm.core.runtime.driver import DriverConfig, drive
from agentm.core.runtime.extension_install import InstallLedger, LedgerSnapshot
from agentm.core.runtime.provider_registry import ProviderRegistry, ProviderSnapshot
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


@dataclass(frozen=True, slots=True)
class _ExtensionInstallSnapshot:
    """Mutable session composition captured before one atom installation."""

    bus: EventBus
    services: ServiceRegistry
    codec: CodecRegistry
    tools: tuple[Tool, ...]
    context_policies: tuple[ContextPolicy, ...]
    trigger_renderers: dict[str, TriggerRenderer]
    ledger: LedgerSnapshot
    providers: ProviderSnapshot


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
        self.system = runtime.system
        self.context_policies: list[ContextPolicy] = list(context_policies or [])
        self.trigger_renderers: dict[str, TriggerRenderer] = dict(
            trigger_renderers or {}
        )
        self._extensions = InstallLedger(
            tools=self.tools,
            context_policies=self.context_policies,
            trigger_renderers=self.trigger_renderers,
        )
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
        self._providers = ProviderRegistry(
            services=self.services,
            committed_turns=lambda: self.trajectory.turns,
            active_set=self._active_set_fingerprint,
            emit_register_event=self._emit_register_event,
            stream_fn=runtime.stream_fn,
            model=runtime.model,
            identity=runtime.provider_identity,
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
        self._providers.activate()
        if self._providers.stream_fn is None:
            raise RuntimeError(f"session {self.id}: cannot start without stream_fn")
        if self._providers.model is None:
            raise RuntimeError(f"session {self.id}: cannot start without model")

        for policy in self.context_policies:
            self._bind_context_policy(policy)

        self.bus.on(
            TurnCommittedEvent.CHANNEL,
            self._providers.on_turn_committed,
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
                model=self._providers.model,
            ),
        )

    def _policy_context(self) -> PolicyContext:
        return PolicyContext(
            session_id=self.id,
            parent_session_id=self.ctx.parent_session_id,
            services=self.services,
            store=self.store,
            model=self._providers.model,
            stream_fn=self._providers.stream_fn,
            trigger_renderers=dict(self.trigger_renderers),
        )

    def _bind_context_policy(self, policy: ContextPolicy) -> None:
        if isinstance(policy, BindableContextPolicy):
            policy.bind(self._policy_context())

    async def _run_driver(self) -> None:
        try:
            stream_fn = self._providers.stream_fn
            model = self._providers.model
            assert stream_fn is not None
            assert model is not None
            audience: PermissionAudience = "user" if self.ctx.depth == 0 else "subagent"
            await drive(
                DriverConfig(
                    trajectory=self.trajectory,
                    triggers=self.triggers,
                    bus=self.bus,
                    stream_fn=stream_fn,
                    model=model,
                    tools=self.tools,
                    store=self.store,
                    session_id=self.id,
                    root_session_id=self.ctx.root_session_id,
                    parent_session_id=self.ctx.parent_session_id,
                    permission_audience=audience,
                    system=self.system,
                    context_policies=self.context_policies,
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
        return self._providers.model

    @property
    def installed_extensions(self) -> list[str]:
        """Module paths of the atoms installed into this session, in order."""

        return self._extensions.module_paths

    @property
    def session_id(self) -> str:
        return self.id

    def get_messages(self) -> list[AgentMessage]:
        return build_context_sync(self.trajectory.turns, self.trigger_renderers)

    def get_turns(self) -> list[Turn]:
        return list(self.trajectory.turns)

    def final_result(self) -> SessionResult | None:
        return compute_session_result(self.trajectory.turns)

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
        self._extensions.note_tool(tool, current_installing_extension() or None)
        self._emit_register_event("tool", tool.name, {"tool": tool})

    def register_context_policy(
        self, policy: ContextPolicy, *, priority: int = 500
    ) -> None:
        from agentm.core.runtime.extension import current_installing_extension

        if any(existing is policy for existing in self.context_policies):
            raise ValueError("context policy instance is already registered")
        self.context_policies.append(policy)
        self._extensions.note_context_policy(
            policy,
            current_installing_extension() or None,
            priority=priority,
        )
        self.context_policies.sort(key=self._extensions.priority_of)
        if self._driver_task is not None:
            try:
                self._bind_context_policy(policy)
            except BaseException:
                self.context_policies.remove(policy)
                self._extensions.drop_context_policy(policy)
                raise
        self._emit_register_event(
            "context_policy",
            type(policy).__name__,
            {"policy": policy, "priority": priority},
        )

    def register_trigger_renderer(self, source: str, renderer: TriggerRenderer) -> None:
        from agentm.core.runtime.extension import current_installing_extension

        self.trigger_renderers[source] = renderer
        self._extensions.note_trigger_renderer(
            source, current_installing_extension() or None
        )
        self._emit_register_event(
            "trigger_renderer",
            source,
            {"renderer": renderer},
        )

    def register_trigger_codec(self, source: str, codec: object) -> None:
        from agentm.core.runtime.extension import current_installing_extension

        if not isinstance(codec, TriggerCodec):
            raise TypeError("trigger codec must implement serialize and deserialize")
        owner = current_installing_extension() or None
        # A superseded atom keeps its codec registered so committed turns stay
        # decodable; its replacement takes the source over rather than colliding.
        self.codec.register_trigger_codec(
            source,
            codec,
            replace=self._extensions.trigger_codec_is_superseded(source),
        )
        self._extensions.note_trigger_codec(source, owner)
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
        from agentm.core.runtime.extension import current_installing_extension

        self._extensions.note_service(key, current_installing_extension() or None)
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

        event = ApiRegisterEvent(
            kind=kind,
            name=name,
            extension=current_installing_extension(),
            payload=payload,
        )
        self.bus.emit_sync(ApiRegisterEvent.CHANNEL, event)

    # --- Providers ---

    def register_provider(
        self,
        name: str,
        config: ProviderConfig,
        *,
        replace: bool = False,
    ) -> None:
        """Register an LLM provider and refresh the active provider."""

        self._providers.register(name, config, replace=replace)

    def has_provider(self, name: str) -> bool:
        return self._providers.has(name)

    def get_provider(self, name: str | None = None) -> ProviderConfig | None:
        return self._providers.get(name)

    def provider_names(self) -> list[str]:
        return self._providers.names()

    def provider_session_identity(self) -> ProviderSessionIdentity | None:
        """Return the provider/model identity bound to this session, if known."""

        return self._providers.session_identity()

    # --- Environment restore ---

    def _environment_restore_failure_handler(
        self,
    ) -> EnvironmentRestoreFailureHandler | None:
        return self.services.get_role(ENVIRONMENT_RESTORE_FAILURE_HANDLER)

    def _record_environment_restore_status(
        self,
        status: EnvironmentRestoreStatus,
    ) -> None:
        self.services.bind(ENVIRONMENT_RESTORE_STATUS_ROLE, status, replace=True)

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
        replace: bool = False,
    ) -> None:
        """Install an extension through the standard lifecycle path.

        Installing into a running session is allowed. The driver re-reads the
        session's tool list at every turn boundary, so an atom installed while a
        turn is in flight becomes visible to the model on the next turn and
        cannot change the tool surface the running turn already advertised.

        What the driver captured once at start is not reachable this way: an
        atom installed at runtime cannot replace the tool executor or the
        permission policy the running driver consults.
        """
        from agentm.core.runtime.extension import install_extension

        runtime_install = self._driver_task is not None
        if runtime_install:
            self._verify_runtime_requirements(extension)
        await install_extension(
            cast("Session", self),
            extension,
            None if isinstance(extension, ExtensionSpec) else config or {},
            trigger=trigger,
            runtime=runtime_install,
            replace=replace,
        )

    def _verify_runtime_requirements(
        self,
        extension: ExtensionSpec | str,
    ) -> None:
        """Solve one atom's requirements against the live capability set.

        Composition-time solving orders the whole plan at once. A late install
        has no plan to be ordered within, so its requirements are checked
        against what the session actually provides right now, and the failure
        reads the same either way.
        """
        from agentm.core.runtime.extension import load_manifest_for_spec

        manifest = load_manifest_for_spec(extension)
        if manifest is None or not manifest.requires:
            return
        available = {f"service:{name}" for name in self.services.names()}
        available |= {f"atom:{path}" for path in self._extensions.module_paths}
        missing = [
            requirement
            for requirement in manifest.requires
            if requirement_key(requirement) not in available
        ]
        if missing:
            raise ValueError(
                f"unsatisfied atom dependencies: {manifest.name} requires "
                f"{', '.join(missing)}"
            )

    def remove_atom_registrations(self, module_path: str) -> None:
        """Detach everything one atom registered, so a newer version can land.

        Trigger codecs are left registered on purpose. A committed turn names
        its trigger source, and a session that could not decode that source
        would fail to resume; the superseding version registers over the same
        source instead of the source disappearing between the two.

        Two consequences a caller has to know, because neither is repairable
        from here:

        A replacement does not land where the original sat. Bus subscriptions
        are ordered by ``(priority, seq)`` with a monotonic ``seq``, so
        re-registering puts the atom last within its band. For a channel where
        position decides the outcome -- the system prompt is last-writer-wins,
        the tool list is mapped by each handler in turn -- a reloaded atom
        composes differently from the same atom installed at startup. Reload is
        a development loop, not a way to reproduce a run.

        A service key is unregistered outright, including one this atom bound
        over another atom's value. Executor decorators do that: each reads the
        current binding and binds itself wrapping it, so the ledger's owner for
        that key is whoever bound last. Superseding that atom removes the whole
        composed chain rather than unwrapping one layer, and the replacement
        rebuilds from whatever the bare default is. Atoms that wrap a service
        are not supersede-safe.
        """

        registrations = self._extensions.registrations_of(module_path)
        if registrations.tool_ids:
            self.tools[:] = [
                tool for tool in self.tools if id(tool) not in registrations.tool_ids
            ]
        if registrations.context_policy_ids:
            self.context_policies[:] = [
                policy
                for policy in self.context_policies
                if id(policy) not in registrations.context_policy_ids
            ]
        for source in registrations.trigger_renderer_sources:
            self.trigger_renderers.pop(source, None)
        for key in registrations.service_keys:
            self.services.unregister(key)
        removed_handlers = self.bus.remove_owner(module_path)
        self._extensions.forget(module_path)
        logger.debug(
            "detached atom {}: {} tools, {} policies, {} renderers, "
            "{} services, {} handlers",
            module_path,
            len(registrations.tool_ids),
            len(registrations.context_policy_ids),
            len(registrations.trigger_renderer_sources),
            len(registrations.service_keys),
            removed_handlers,
        )

    def installed_atom_module_path(self, atom_name: str) -> str | None:
        """Module path of an installed atom by its manifest name, if present."""

        return self._extensions.installed_module_path(atom_name)

    def _capture_extension_install_state(self) -> _ExtensionInstallSnapshot:
        return _ExtensionInstallSnapshot(
            bus=self.bus.copy(),
            services=self.services.copy(),
            codec=self.codec.copy(),
            tools=tuple(self.tools),
            context_policies=tuple(self.context_policies),
            trigger_renderers=dict(self.trigger_renderers),
            ledger=self._extensions.capture(),
            providers=self._providers.capture(),
        )

    def _restore_extension_install_state(
        self,
        snapshot: _ExtensionInstallSnapshot,
    ) -> None:
        # Restore in place throughout. The driver and the install ledger hold
        # references to these same container objects, so rebinding the attribute
        # would roll back the session's view while leaving the driver serving
        # whatever the failed install appended.
        self.bus.replace_from(snapshot.bus)
        self.services.replace_from(snapshot.services)
        self.codec.replace_from(snapshot.codec)
        self.tools[:] = snapshot.tools
        self.context_policies[:] = snapshot.context_policies
        self.trigger_renderers.clear()
        self.trigger_renderers.update(snapshot.trigger_renderers)
        self._extensions.restore(snapshot.ledger)
        self._providers.restore(snapshot.providers)

    def record_installed_extension(
        self,
        spec: ExtensionSpec,
        *,
        runtime: bool = False,
        atom_name: str | None = None,
    ) -> None:
        """Record one installed extension for composition snapshots."""

        if not isinstance(spec, ExtensionSpec):
            raise TypeError("installed extension record requires ExtensionSpec")
        self._extensions.record_installed(spec, runtime=runtime, atom_name=atom_name)

    def composition_snapshot(
        self,
        *,
        include_provider_atoms: bool = True,
    ) -> CompositionSnapshot:
        """Snapshot the rebuildable composition for spawn/fork/child paths."""

        provider_atoms: set[str] = (
            set()
            if include_provider_atoms
            else {
                owner
                for owner in self._providers.owners().values()
                if owner is not None
            }
        )
        return CompositionSnapshot(
            extensions=tuple(
                self._extensions.composition_extensions(
                    excluded_module_paths=provider_atoms,
                )
            ),
            external_tools=tuple(self._extensions.external_tools(self.tools)),
            external_context_policies=tuple(
                copy.copy(policy)
                for policy in self._extensions.external_context_policies(
                    self.context_policies
                )
            ),
            external_trigger_renderers=self._extensions.external_trigger_renderers(
                self.trigger_renderers
            ),
            codec=self._extensions.composition_codec(self.codec),
            stream_fn=self._providers.stream_fn,
            model=self._providers.model,
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
