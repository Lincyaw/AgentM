# code-health: ignore-file[AM025] -- runtime composes plugin, service, and trajectory boundary values
"""Session runtime — lifecycle, registration, providers, and composition.

Owns driver task, trajectory, trigger queue, bus, tools, services,
context policies, and shutdown logic.  ``Session`` (session.py) extends
this with child, fork, and resume operations.

Two collaborators own the state the runtime only routes to: which model
the session talks to lives in ``provider_registry.py``, and which atom
registered what lives in ``extension_install.py``.

What an atom registered does not live in this session's tables at all.  Each
installation gets its own ``AtomContext`` (``atom_context.py``) with its own
tools, policies, renderers, services, bus segment and effect log, and this
session's visible state is the union of its *own* tables and every linked
context's.  Detaching an atom unlinks its context; the methods below are
therefore the host's write path, attributed to nobody, and an atom never
reaches them.  The install ledger still records the same ownership from the
context's side, so the two accounts can be checked against each other.

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
from agentm.core.abi.manifest import live_capability_keys, requirement_key
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
    AtomInstall,
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
from agentm.core.runtime.atom_context import (
    AtomContext,
    AtomResidue,
    ChainedPolicies,
    ChainedRenderers,
    ChainedTools,
    ContextOwnership,
    DepartedContexts,
    PolicyRow,
    ownership_of,
    policy_row,
)
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
    """Mutable session composition captured before one atom installation.

    The host's own tables and the *shape* of the context tree, which is all an
    installation can move: an atom writes into its own context, so putting the
    link list back is what takes a failed installation's writes away, and what
    gives a superseded atom's back.
    """

    bus: EventBus
    services: ServiceRegistry
    codec: CodecRegistry
    tools: tuple[Tool, ...]
    context_policies: tuple[PolicyRow, ...]
    trigger_renderers: dict[str, TriggerRenderer]
    linked: tuple[AtomContext, ...]
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
        self.system = runtime.system
        # The host's own tables, and the contexts linked into them. Everything
        # the session reads is the union of the two, through the live views
        # below -- the driver takes those once at start and re-reads them at
        # every turn boundary, so an atom linked mid-run is visible from the
        # next turn without anything being handed to the driver again.
        self._own_tools: list[Tool] = list(tools or [])
        self._own_policies: list[PolicyRow] = [
            policy_row(policy, 500) for policy in context_policies or []
        ]
        self._own_renderers: dict[str, TriggerRenderer] = dict(trigger_renderers or {})
        self._linked: list[AtomContext] = []
        self._departed = DepartedContexts()
        self.tools: Sequence[Tool] = ChainedTools(self._own_tools, self._linked)
        self.context_policies: Sequence[ContextPolicy] = ChainedPolicies(
            self._own_policies, self._linked
        )
        self.trigger_renderers: Mapping[str, TriggerRenderer] = ChainedRenderers(
            self._own_renderers, self._linked
        )
        self._extensions = InstallLedger(
            tools=self._own_tools,
            context_policies=[row.policy for row in self._own_policies],
            trigger_renderers=self._own_renderers,
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
        self.services.set_write_observer(self._on_service_write)
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
        self._pending_atom_installs: list[AtomInstall] = []
        self._shutdown_task: asyncio.Task[None] | None = None
        self._cleanup_callbacks: list[Callable[[], Awaitable[None]]] = []
        self._providers = ProviderRegistry(
            services=self.services,
            committed_turns=lambda: self.trajectory.turns,
            active_set=self._active_set_fingerprint,
            emit_register_event=self._emit_register_event,
            refile_service=self.refile_service,
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

        self._bind_all_context_policies()

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

    def _policy_context(self, services: ServiceRegistry) -> PolicyContext:
        """The context one policy is bound with.

        ``services`` is the registry of whoever registered the policy, so a
        policy an atom installed writes through its own atom's context and a
        policy the host installed writes as the host. A policy that could only
        be handed the session's own registry would be a hole: it outlives no
        atom, but every service it registered would.
        """

        return PolicyContext(
            session_id=self.id,
            parent_session_id=self.ctx.parent_session_id,
            services=services,
            store=self.store,
            model=self._providers.model,
            stream_fn=self._providers.stream_fn,
            trigger_renderers=dict(self.trigger_renderers),
        )

    def bind_context_policy(
        self,
        policy: ContextPolicy,
        *,
        services: ServiceRegistry,
    ) -> None:
        """Give one policy its runtime context, writing through ``services``."""

        if isinstance(policy, BindableContextPolicy):
            policy.bind(self._policy_context(services))

    def _bind_all_context_policies(self) -> None:
        """Bind every policy the session holds, each through its own writer."""

        for row in self._own_policies:
            self.bind_context_policy(row.policy, services=self.services)
        for context in self._linked:
            for row in context.tables.policies:
                self.bind_context_policy(row.policy, services=context.services)

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
                    drain_atom_installs=self.drain_atom_installs,
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
        # The session is going away, so every context is unlinked and its
        # recorded effects are released rather than run -- here rather than by
        # the collector, because a log can still hold a queued body to close. A
        # task that survives writes into an unlinked context afterwards, which
        # nothing reads.
        for context in tuple(self._linked):
            context.unlink_from(self)
            cleanup_errors.extend(context.suspend().dispose())
        # A context this session *removed* is undone rather than released: its
        # removal already decided that what it wrote comes back out, and an
        # effect recorded through it since is under the same decision.
        cleanup_errors.extend(self._departed.undo())
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
        return build_context_sync(self.trajectory.turns, dict(self.trigger_renderers))

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
        """Subscribe on the session's own segment, owned by nobody.

        This is the host's path. An atom does not reach it: it subscribes into
        its own context's segment, which is unlinked when it leaves.
        """

        return self.bus.on(channel, handler, priority=priority)

    def add_observer(self, observer: EventBusObserver) -> Callable[[], None]:
        """Register a bus observer for session-scoped instrumentation."""

        return self.bus.add_observer(observer)

    # --- Registration (the host's own tables) ---

    def register_tool(self, tool: Tool) -> None:
        existing = {t.name for t in self.tools}
        if tool.name in existing:
            raise ValueError(f"duplicate tool: {tool.name}")
        self._own_tools.append(tool)
        self._extensions.note_tool(tool, None)
        self._emit_register_event("tool", tool.name, {"tool": tool})

    def register_context_policy(
        self, policy: ContextPolicy, *, priority: int = 500
    ) -> None:
        if any(existing is policy for existing in self.context_policies):
            raise ValueError("context policy instance is already registered")
        row = policy_row(policy, priority)
        self._own_policies.append(row)
        self._own_policies.sort(key=lambda held: (held.priority, held.order))
        self._extensions.note_context_policy(policy, None, priority=priority)
        if self._driver_task is not None:
            try:
                self.bind_context_policy(policy, services=self.services)
            except BaseException:
                self._own_policies[:] = [
                    held for held in self._own_policies if held is not row
                ]
                self._extensions.drop_context_policy(policy)
                raise
        self._emit_register_event(
            "context_policy",
            type(policy).__name__,
            {"policy": policy, "priority": priority},
        )

    def register_trigger_renderer(self, source: str, renderer: TriggerRenderer) -> None:
        self._own_renderers[source] = renderer
        self._extensions.note_trigger_renderer(source, None)
        self._emit_register_event(
            "trigger_renderer",
            source,
            {"renderer": renderer},
        )

    def register_trigger_codec(self, source: str, codec: object) -> None:
        if not isinstance(codec, TriggerCodec):
            raise TypeError("trigger codec must implement serialize and deserialize")
        # A superseded atom keeps its codec registered so committed turns stay
        # decodable; its replacement takes the source over rather than colliding.
        self.codec.register_trigger_codec(
            source,
            codec,
            replace=self._extensions.trigger_codec_is_superseded(source),
        )
        self._extensions.note_trigger_codec(source, None)
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

    def note_atom_trigger_codec(
        self,
        source: str,
        codec: TriggerCodec,
        owner: str,
    ) -> None:
        """Register one atom's trigger codec on the session, and record it.

        The one registration an atom makes that does not land in its own
        tables, so the rule about it lives here rather than there: a superseded
        atom keeps its codec so committed turns stay decodable, and its
        replacement takes the source over rather than colliding.
        """

        self.codec.register_trigger_codec(
            source,
            codec,
            replace=self._extensions.trigger_codec_is_superseded(source),
        )
        self._extensions.note_trigger_codec(source, owner)

    def refile_service(self, key: str) -> None:
        """Re-file one service key under whoever's write resolves now.

        For a key that changes hands without being written -- a rollback taking
        a shadowing write back out. No write observer fires for a removal, and
        the new owner is a property of the context tree, which the registry
        that removed the write cannot read for itself.
        """

        self._extensions.note_service(key, self.ownership().service(key))

    def _on_service_write(
        self,
        key: str,
        service: object,
        scope: ServiceScope,
        *,
        role_bind: bool,
    ) -> None:
        """A write into the session's own registry — the host's, by definition."""

        self.note_service_write(key, service, scope, role_bind=role_bind, context=None)

    def note_service_write(
        self,
        key: str,
        service: object,
        scope: ServiceScope,
        *,
        role_bind: bool,
        context: AtomContext | None,
    ) -> None:
        """Mirror one service write into the ledger; announce role bindings.

        Which registry the write landed in is what decides the owner: an atom
        holds its own, the host holds the session's, and each registry's write
        observer is the one belonging to whoever holds it. No caller passes a
        name and no ambient state is consulted, so the task the write happens on
        cannot change the answer.

        A write into a context this session does not aggregate is not accounted
        for at all. That is a read of the session's *own* link list to decide
        what goes into the session's *own* record -- the same aggregation every
        other reader performs -- and not a permission check on the writer: the
        write has already happened, into a table it was entitled to, and
        nothing here can refuse it or reach it.

        The ledger is not what removal reads any more -- unlinking the context
        is -- but it is kept up to date so its account and the context tree's
        can be checked against each other. The register event stays role-only:
        plain registrations are frequent and include the driver's per-turn
        resource transaction, which nothing on the bus wants to hear about once
        a turn.
        """

        if context is not None and not self.is_linked(context):
            return
        owner = None if context is None else context.module_path
        self._extensions.note_service(key, owner)
        if role_bind:
            self.emit_register_event(
                "service",
                key,
                {"service": service, "scope": scope},
                owner=owner,
            )

    def emit_register_event(
        self,
        kind: str,
        name: str,
        payload: dict[str, object],
        *,
        owner: str | None,
    ) -> None:
        """Announce one registration, naming the context that made it."""

        event = ApiRegisterEvent(
            kind=kind,
            name=name,
            extension=owner or "",
            payload=payload,
        )
        self.bus.emit_sync(ApiRegisterEvent.CHANNEL, event)

    def _emit_register_event(
        self,
        kind: str,
        name: str,
        payload: dict[str, object],
    ) -> None:
        self.emit_register_event(kind, name, payload, owner=None)

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
        if runtime_install:
            self._note_atom_install(extension)

    def _note_atom_install(self, extension: ExtensionSpec | str) -> None:
        """Queue a durable record of one runtime install for the next commit."""

        from agentm.core.runtime.extension import (
            coerce_extension_spec,
            load_manifest_for_spec,
        )

        spec = coerce_extension_spec(extension, None)
        manifest = load_manifest_for_spec(spec)
        if manifest is None:
            logger.warning(
                "atom {} installed at runtime without a MANIFEST; it cannot be "
                "recorded on the turn and will not survive resume",
                spec.module_path,
            )
            return
        self._pending_atom_installs.append(
            AtomInstall(
                atom_name=manifest.name,
                source_kind=spec.source.kind,
                location=spec.source.location,
                digest=spec.source.digest,
                config=spec.config,
            )
        )

    def drain_atom_installs(self) -> tuple[AtomInstall, ...]:
        """Take the runtime installs awaiting a turn to be recorded on."""

        drained = tuple(self._pending_atom_installs)
        self._pending_atom_installs.clear()
        return drained

    def _live_capability_keys(self) -> set[str]:
        """What this session provides right now, keyed as manifests key it.

        Everything present counts, the embedder's own tools included. This is
        deliberately wider than the set a cold composition solves against (see
        ``session_factory._service_capabilities``): that set is narrow because
        it decides install *order*, and a late install has no order to decide.
        """

        return live_capability_keys(
            services=self.services.names(),
            atoms=self._extensions.installed_atom_names(),
            tools=[tool.name for tool in self.tools],
            providers=self._providers.names(),
            trigger_renderers=self.trigger_renderers.keys(),
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

        The two solvers take deliberately different inputs and only the failure
        message is shared. Ordering is what makes the cold set narrow, and
        ordering does not exist here: a satisfiable requirement is satisfiable
        no matter who provided it, so a host tool counts at runtime where it
        would not count at composition time.
        """
        from agentm.core.runtime.extension import load_manifest_for_spec

        manifest = load_manifest_for_spec(extension)
        if manifest is None or not manifest.requires:
            return
        available = self._live_capability_keys()
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

    def remove_atom_registrations(self, module_path: str) -> AtomResidue | None:
        """Unlink one atom's context and move what it held out of it.

        Returns the residue rather than reverting it, because the two callers
        want opposite things from it. ``uninstall_extension`` reverts it: the
        atom is gone and whatever it recorded through ``api.effect`` has to be
        undone. The supersede path holds it until the replacement has landed,
        and gives it back to the context if it has not — which is why nothing
        here is reverted eagerly. A failed ``replace=True`` has nothing to
        un-revert, and that is a property of this method, not a promise made
        elsewhere.

        Unlinking is what takes the atom's tools, policies, renderers,
        services and bus handlers out of the session: they were never in the
        session's tables. Suspending the context additionally empties its own
        tables, so its *reads* resolve up the chain to whatever replaced it.

        Trigger codecs stay registered: a committed turn names its trigger
        source, and a session that could not decode that source would fail to
        resume. They are the one write recorded as an effect with a retention
        reason rather than held in a context table, so this leaves them alone
        and ``revert`` leaves them alone too.

        One limit remains and is the caller's, because it is not repairable
        from here: a replacement lands last within its bus priority band rather
        than where the original sat, since subscriptions order by
        ``(priority, seq)`` and the replacement's are new. Where position
        decides the outcome, a caller that cares reinstalls everything the
        composition lists after this atom, which is what ``atom_watch`` does.

        One sequencing property this leaves behind is load-bearing outside
        core. On the supersede path ``install_extension`` calls this and then
        ``load_extension`` with no await between them, so the window in which
        a superseded atom's keys are absent is not observable by any other
        task. ``atom_watch`` reads its own follower key to tell "superseded"
        from "detached outright"; after the window it reads the successor's
        registration through the chain, which is the handover.
        """

        context = self.context_for(module_path)
        if context is None:
            self._extensions.forget(module_path)
            return None
        context.unlink_from(self)
        residue = context.suspend()
        self._departed.note(context)
        self._extensions.forget(module_path)
        removed_providers = self._uncover(residue, module_path)
        logger.debug(
            "unlinked atom {}: {} tools, {} policies, {} renderers, "
            "{} services, {} providers, {} bus channels, {} recorded effects",
            module_path,
            len(residue.tables.tools),
            len(residue.tables.policies),
            len(residue.tables.renderers),
            len(residue.services.own_names()),
            len(removed_providers),
            len(residue.segment.handlers),
            len(residue.effects),
        )
        return residue

    def _uncover(self, residue: AtomResidue, module_path: str) -> list[str]:
        """Say who holds the keys an unlinked context was shadowing.

        Unlinking is not the same as removing, and this is where the difference
        shows. A key two contexts wrote resolves to the survivor the moment the
        writer leaves, so the session's other accounts of ownership have to be
        told: the install ledger, whose ``forget`` has just dropped the key
        outright, and the provider registry, whose ownership index and active
        name are its own.

        Returns the providers that really went away, as opposed to the ones
        that were merely uncovered.
        """

        held = self.ownership()
        for key in residue.services.own_names():
            if self.services.has(key):
                self._extensions.note_service(key, held.service(key))
        for source in residue.tables.renderers:
            if source in self.trigger_renderers:
                self._extensions.note_trigger_renderer(source, held.renderer(source))
        departed: list[str] = []
        uncovered: list[str] = []
        for name, owner in self._providers.owners().items():
            if owner != module_path:
                continue
            if self.services.has(f"provider:{name}"):
                uncovered.append(name)
            else:
                departed.append(name)
        for name in uncovered:
            self._providers.note_owner(name, held.service(f"provider:{name}"))
        for name in departed:
            self._providers.unregister(name)
        if uncovered:
            # Re-resolve, so the session's active stream and model are the ones
            # the uncovered registration names rather than the departed atom's.
            self._reactivate_providers()
        return departed

    def _reactivate_providers(self) -> None:
        """Pick the active provider again, tolerating a session with none left.

        ``activate`` raises when the session is bound to a provider that is now
        absent, which is a real error at a registration but not something a
        detach may propagate: the atom is already gone, and the caller asked to
        remove it, not to install one.
        """

        try:
            self._providers.activate()
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            logger.warning(
                "session {} could not re-resolve its provider after a detach: {}",
                self.id,
                exc,
            )

    def uninstall_extension(self, atom: ExtensionSpec | str) -> bool:
        """Detach an atom; report whether one was installed.

        Accepts the spec it was installed from, whose module path is derived
        from its source and needs nothing loaded, or a manifest name for a
        caller that only knows what the atom calls itself.

        Removal is the same operation superseding already performs, so it
        keeps the same limit: an atom's trigger codecs stay registered. A
        committed turn names its trigger source, and a session that could no
        longer decode it would fail to resume, so removing an atom takes away
        what it offers the model and leaves what the record depends on.
        """

        if isinstance(atom, ExtensionSpec):
            module_path: str | None = atom.module_path
            if module_path not in self._extensions.module_paths:
                return False
        else:
            module_path = self._extensions.installed_module_path(atom)
        if module_path is None:
            return False
        residue = self.remove_atom_registrations(module_path)
        if residue is not None:
            residue.effects.revert_or_raise(
                f"undoing the effects of {module_path} failed"
            )
        logger.info("uninstalled atom {}", module_path)
        return True

    def installed_atom_module_path(self, atom_name: str) -> str | None:
        """Module path of an installed atom by its manifest name, if present."""

        return self._extensions.installed_module_path(atom_name)

    # --- Context tree ---

    def link_context(self, context: AtomContext) -> None:
        """Aggregate one atom context into what this session resolves.

        At most one context per module path is ever linked, and this is where
        that holds. Everything else that answers for an atom is keyed on the
        module path and holds exactly one entry for it -- the ledger's
        attribution, its replayable spec, ``forget``, ``context_for`` -- so a
        second live context under one path is state they cannot represent:
        removal would unlink one incarnation while the ledger dropped the path
        outright, leaving the other linked with nothing left to name it.

        Superseding is not that case, because the previous context is unlinked
        before its replacement links. Refused here rather than in the install
        path because this is the list that would hold the second one, and the
        cold path already refuses it as a duplicate atom name.
        """

        if any(existing is context for existing in self._linked):
            return
        module_path = context.module_path
        if any(existing.module_path == module_path for existing in self._linked):
            raise ValueError(
                f"atom {module_path} is already installed in this session; "
                "install it with replace=True to supersede the one that is, or "
                "uninstall that one first"
            )
        self._linked.append(context)
        # A supersede rollback links a removed context again, and a context
        # this session holds has nothing left for shutdown to undo separately.
        self._departed.forget(context)

    def unlink_context(self, context: AtomContext) -> None:
        """Stop aggregating one atom context; safe to repeat."""

        self._linked[:] = [
            existing for existing in self._linked if existing is not context
        ]

    def is_linked(self, context: AtomContext) -> bool:
        """Whether this session aggregates ``context`` right now."""

        return any(existing is context for existing in self._linked)

    def linked_contexts(self) -> tuple[AtomContext, ...]:
        """The atom contexts this session resolves through, in link order."""

        return tuple(self._linked)

    def context_for(self, module_path: str) -> AtomContext | None:
        """The linked context of one atom, if it is still linked.

        There is at most one, and ``link_context`` is what makes that true.  A
        second incarnation of an atom exists only across a supersede, where the
        first is unlinked before the second links: the first's context, if a
        task of it survives, is not returned here and answers for nobody.
        """

        for context in self._linked:
            if context.module_path == module_path:
                return context
        return None

    def policy_rows(self) -> tuple[PolicyRow, ...]:
        """Every context policy with the priority it was registered at.

        The view exposes policies; a reader that has to state their priorities
        -- the composition digest -- needs the rows behind it.
        """

        rows = list(self._own_policies)
        for context in self._linked:
            rows.extend(context.tables.policies)
        rows.sort(key=lambda row: (row.priority, row.order))
        return tuple(rows)

    def ownership(self) -> ContextOwnership:
        """Which context holds each tool, policy, renderer and service.

        The session's own registry takes part: a service key resolves by write
        order across the whole chain, so a host write later than every atom's
        owns it -- as nobody.
        """

        return ownership_of(self.services, self._linked)

    @property
    def driver_running(self) -> bool:
        """Whether the driver loop has been started."""

        return self._driver_task is not None

    def _capture_extension_install_state(self) -> _ExtensionInstallSnapshot:
        """The host's own tables plus the shape of the context tree."""

        return _ExtensionInstallSnapshot(
            bus=self.bus.copy(),
            services=self.services.copy(),
            codec=self.codec.copy(),
            tools=tuple(self._own_tools),
            context_policies=tuple(self._own_policies),
            trigger_renderers=dict(self._own_renderers),
            linked=tuple(self._linked),
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
        #
        # Three link lists say the same thing and all three are put back: the
        # bus's segments, the registry's child registries, and this session's
        # contexts. A failed installation's context is absent from all of them
        # afterwards, and a superseded one is present in all of them again.
        self.bus.replace_from(snapshot.bus)
        self.services.replace_from(snapshot.services)
        self.codec.replace_from(snapshot.codec)
        self._own_tools[:] = snapshot.tools
        self._own_policies[:] = snapshot.context_policies
        self._own_renderers.clear()
        self._own_renderers.update(snapshot.trigger_renderers)
        self._linked[:] = snapshot.linked
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
