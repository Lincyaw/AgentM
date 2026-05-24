"""Extension loader + concrete ``ExtensionAPI`` implementation.

Implements the impl side of §2 (Extension Model) and §3 (ExtensionAPI v0
surface) of ``.claude/designs/extension-as-scenario.md``. The atom-facing
Protocols, dataclasses, exceptions, and ContextVar live in
:mod:`agentm.core.abi.extension` — this module imports them.

Pluggability hard rule (see ``.claude/designs/pluggable-architecture.md`` §1):
atoms never import from this module. The §11 validator forbids
``agentm.core.runtime`` outright.
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from agentm.core.abi import (
    EventBus,
    Model,
    ProviderConfig,
    Tool,
)
from agentm.core.abi.catalog import CatalogService
from agentm.core.abi.extension import (
    AtomInfo,
    ChildSessionFactory,
    CommandDispatcher,
    CommandDispatchResult,
    CommandSpec,
    ExtensionAPI,
    ExtensionFactory,
    ExtensionLoadError,
    ExtensionStaleError,
    Handler,
    InstallAtomResult,
    ReadonlySession,
    ReloadResult,
    Renderer,
    UnknownCommandError,
    UnloadAtomResult,
    Unsubscribe,
    _INSTALLING_EXTENSION,
    _SessionGateway,
    current_installing_extension,
)
from agentm.core.abi.operations import (
    BashOperations,
    FileOperations,
    Operations,
)
from agentm.core.abi.project_layout import ProjectLayout
from agentm.core.abi.resource import ResourceWriter
from agentm.core.runtime.otel_export import (
    SessionTelemetry,
    setup_session_telemetry,
)
from agentm.core.runtime.resource_writer import GitBackedResourceWriter
from agentm.core.runtime.services import (
    default_catalog_service,
    default_project_layout,
)


class _OperationsHolder:
    """Mutable single-slot container for the session's ``Operations`` bundle.

    Operations is the **strict-register** axis: the substrate has no sensible
    "no-op shell" default, so an atom must call
    :meth:`ExtensionAPI.register_operations` exactly once before freeze or
    the session refuses to start. All :class:`_ExtensionAPIImpl` instances
    built for a session share the same holder so that whichever atom
    registers Operations is visible to every subsequent atom regardless of
    api capture order.

    Contrast :class:`_ResourceWriterHolder` below, which is **default-pluggable**.
    See ``.claude/designs/pluggable-architecture.md`` §"Port-default pattern"
    for the rule that picks between the two shapes.
    """

    __slots__ = ("bundle",)

    def __init__(self) -> None:
        self.bundle: Operations | None = None


class _ResourceWriterHolder:
    """Mutable single-slot container for the session's ``ResourceWriter``.

    ResourceWriter is a **default-pluggable** axis: the substrate itself
    depends on a working writer (catalog freeze, atom reload, etc. — all
    substrate-side bookkeeping that runs before any atom-driven write), so
    it pre-populates the slot with :class:`GitBackedResourceWriter` against
    the host filesystem. An atom may overwrite the slot once via
    :meth:`ExtensionAPI.register_resource_writer` to redirect writes (e.g.
    into a sandbox). Catalog and project-layout follow the same shape.

    The ``replaced`` flag exists solely to enforce "register-once": without
    it, ``register_resource_writer`` couldn't distinguish "an atom is
    overwriting the substrate default" (allowed) from "a second atom is
    overwriting an earlier atom's writer" (rejected). The Operations holder
    doesn't need this because it has no substrate default to disambiguate
    from.

    Sharing the holder across every :class:`_ExtensionAPIImpl` instance for
    a session means the override is visible to every downstream consumer,
    including ``AtomReloader``.
    """

    __slots__ = ("writer", "replaced")

    def __init__(self, writer: ResourceWriter) -> None:
        self.writer: ResourceWriter = writer
        self.replaced: bool = False


class _NoopSessionGateway:
    def reload_atom(
        self,
        name: str,
        new_source: str,
        *,
        agent_initiated: bool = True,
        rationale: str | None = None,
    ) -> ReloadResult:
        del name, new_source, agent_initiated, rationale
        raise RuntimeError("reload_atom is unavailable on this ExtensionAPI instance")

    def install_atom(
        self,
        *,
        name: str,
        source: str,
        target_path: str | None,
        config: dict[str, Any] | None,
        rationale: str | None,
        agent_initiated: bool,
    ) -> InstallAtomResult:
        del name, source, target_path, config, rationale, agent_initiated
        raise RuntimeError("install_atom is unavailable on this ExtensionAPI instance")

    def unload_atom(
        self,
        name: str,
        *,
        agent_initiated: bool = True,
    ) -> UnloadAtomResult:
        del name, agent_initiated
        raise RuntimeError("unload_atom is unavailable on this ExtensionAPI instance")

    def freeze_current(self, name: str) -> str:
        del name
        raise RuntimeError(
            "freeze_current is unavailable on this ExtensionAPI instance"
        )

    def list_atoms(self) -> list[AtomInfo]:
        return []

    def is_constitution_path(self, path: str) -> bool:
        del path
        return False


class _NoopChildSessionFactory:
    async def __call__(self, _config: Any) -> Any:
        raise RuntimeError(
            "spawn_child_session is unavailable on this ExtensionAPI instance"
        )


# Callable that builds a ``SessionTelemetry`` for the current session. Lazy
# construction means tests that never touch telemetry pay no SDK setup cost,
# and the file handle / batch processor threads only spin up when something
# actually wants to write spans or logs. See PR-A
# (``agentm.core.runtime.otel_export``) for the underlying primitives.
SessionTelemetryFactory = Callable[[], SessionTelemetry]


class _SessionTelemetryHolder:
    """Mutable single-slot container for the session's :class:`SessionTelemetry`.

    Telemetry is a **default-pluggable** axis like :class:`_ResourceWriterHolder`:
    the substrate has a working default (``setup_session_telemetry`` from
    :mod:`agentm.core.runtime.otel_export`, which attaches a per-session
    SpanProcessor + LogRecordProcessor to the **process-level** OTel
    providers — PR-H), so the slot pre-populates with a *factory* rather
    than a live handle — construction is deferred to the first
    :meth:`_ExtensionAPIImpl.get_session_telemetry` call so tests / sessions
    that never read telemetry pay no cost.

    On first construction the holder also installs a ``SessionShutdownEvent``
    handler on the session bus that calls :meth:`SessionTelemetry.shutdown`,
    so the per-session batch processors drain and are removed from the
    global providers when the session ends. The providers themselves
    outlive the session and are torn down once at process exit by an
    ``atexit`` hook registered inside ``setup_process_telemetry``.
    ``SessionTelemetry.shutdown`` is itself idempotent so an explicit
    shutdown by the atom + the bus-driven teardown is safe.

    Sharing one holder across every :class:`_ExtensionAPIImpl` instance for
    a session means later atoms see the same telemetry handle as the first
    consumer (typically the observability atom).
    """

    __slots__ = ("_factory", "_bus", "_telemetry", "_shutdown_handler_registered")

    def __init__(
        self,
        factory: SessionTelemetryFactory,
        *,
        bus: EventBus,
    ) -> None:
        self._factory: SessionTelemetryFactory = factory
        self._bus = bus
        self._telemetry: SessionTelemetry | None = None
        self._shutdown_handler_registered = False

    def get(self) -> SessionTelemetry:
        if self._telemetry is None:
            self._telemetry = self._factory()
            self._register_shutdown_handler()
        return self._telemetry

    def _register_shutdown_handler(self) -> None:
        if self._shutdown_handler_registered:
            return
        # Lazy import to keep the events module's runtime-level event names
        # off the top-level import path here (matches the existing pattern
        # used in ``_emit_register``).
        from agentm.core.abi.events import BusPriority, SessionShutdownEvent

        def _on_session_shutdown(_event: SessionShutdownEvent) -> None:
            telemetry = self._telemetry
            if telemetry is None:
                return
            telemetry.shutdown()

        # ``POST`` priority: any observability atom subscribed at the
        # default ``NORMAL`` tier gets to emit its closing
        # ``agentm.session.end`` log record before we drain + close the
        # exporters. Without this the closing record would land on a
        # shut-down processor and silently disappear.
        self._bus.on(
            SessionShutdownEvent.CHANNEL,
            _on_session_shutdown,
            priority=BusPriority.POST,
        )
        self._shutdown_handler_registered = True


def _default_session_telemetry_factory(
    *, cwd: str, session_id: str, scenario: str | None
) -> SessionTelemetryFactory:
    """Bind a :func:`setup_session_telemetry` invocation to a session.

    Extracted so callers (notably tests) can wrap or override the factory
    without rebuilding the bind logic. The bound callable takes no
    arguments — the holder calls it exactly once.
    """

    def _build() -> SessionTelemetry:
        return setup_session_telemetry(
            session_id=session_id,
            cwd=Path(cwd) if cwd else Path.cwd(),
            scenario_name=scenario,
        )

    return _build


# --- Concrete impl ----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExtensionAPIScope:
    """Session-scoped bundle handed to every ``_ExtensionAPIImpl``.

    One scope is built per :class:`AgentSession` and reused for every
    ``_make_api(owner)`` call; only :attr:`_ExtensionAPIImpl._owner_name`
    varies per atom. Bundling here keeps the impl constructor from growing
    a kwarg every time a new pluggability axis lands.

    Build via :func:`build_extension_api_scope` so defaults stay in one
    place — direct construction is fine but every field is then mandatory.
    """

    bus: EventBus
    cwd: str
    session_id: str
    root_session_id: str
    parent_session_id: str | None
    purpose: str
    scenario: str | None
    session: ReadonlySession
    tools: list[Tool]
    commands: dict[str, CommandSpec]
    providers: dict[str, ProviderConfig]
    renderers: dict[str, Renderer]
    pending_user_messages: list[str | list[Any]]
    model_getter: Any
    provider_getter: Any
    gateway: _SessionGateway
    operations: _OperationsHolder
    project_layout: ProjectLayout
    catalog: CatalogService
    child_session_factory: ChildSessionFactory
    resource_writer: _ResourceWriterHolder
    telemetry: _SessionTelemetryHolder
    service_registry: dict[str, Any]


def build_extension_api_scope(
    *,
    bus: EventBus,
    cwd: str,
    session_id: str,
    root_session_id: str | None = None,
    parent_session_id: str | None = None,
    purpose: str = "root",
    scenario: str | None = None,
    session: ReadonlySession,
    tools: list[Tool],
    commands: dict[str, CommandSpec],
    providers: dict[str, ProviderConfig],
    renderers: dict[str, Renderer],
    pending_user_messages: list[str | list[Any]],
    model_getter: Any,
    provider_getter: Any,
    gateway: _SessionGateway | None = None,
    operations: _OperationsHolder | None = None,
    project_layout: ProjectLayout | None = None,
    catalog: CatalogService | None = None,
    child_session_factory: ChildSessionFactory | None = None,
    resource_writer: ResourceWriter | None = None,
    telemetry_factory: SessionTelemetryFactory | None = None,
    service_registry: dict[str, Any] | None = None,
) -> ExtensionAPIScope:
    """Resolve service defaults and return an :class:`ExtensionAPIScope`.

    Centralises the fallback choices (local operations, default project
    layout, git-backed resource writer) so the impl constructor stays a
    pure assignment.
    """

    resolved_layout = project_layout or default_project_layout(cwd)
    return ExtensionAPIScope(
        bus=bus,
        cwd=cwd,
        session_id=session_id,
        # If no trace_id is supplied, the session is its own trace —
        # collapse to session_id. Callers that want strict OTel shape
        # (32-hex trace_id vs 16-hex span_id) should supply both
        # ``session_id`` and ``root_session_id`` on the config.
        root_session_id=root_session_id or session_id,
        parent_session_id=parent_session_id,
        purpose=purpose,
        scenario=scenario,
        session=session,
        tools=tools,
        commands=commands,
        providers=providers,
        renderers=renderers,
        pending_user_messages=pending_user_messages,
        model_getter=model_getter,
        provider_getter=provider_getter,
        gateway=gateway or _NoopSessionGateway(),
        operations=operations if operations is not None else _OperationsHolder(),
        project_layout=resolved_layout,
        catalog=catalog or default_catalog_service(),
        child_session_factory=child_session_factory or _NoopChildSessionFactory(),
        resource_writer=_ResourceWriterHolder(
            resource_writer
            or GitBackedResourceWriter(cwd=cwd, session_id=session_id, bus=bus)
        ),
        telemetry=_SessionTelemetryHolder(
            telemetry_factory
            or _default_session_telemetry_factory(
                cwd=cwd, session_id=session_id, scenario=scenario
            ),
            bus=bus,
        ),
        service_registry=service_registry if service_registry is not None else {},
    )


class _ExtensionAPIImpl:
    """Concrete ``ExtensionAPI`` owned by an ``AgentSession``.

    Intentionally not exported at the module top-level: callers should never
    construct one directly. ``AgentSession`` builds its own instance during
    ``create``.

    The impl is a thin shim — every meaningful action delegates to the
    session's ``EventBus`` or the registries the session exposes.
    """

    def __init__(self, scope: ExtensionAPIScope, *, owner_name: str = "<unknown>") -> None:
        self._bus = scope.bus
        self._cwd = scope.cwd
        self._session_id = scope.session_id
        self._root_session_id = scope.root_session_id
        self._parent_session_id = scope.parent_session_id
        self._purpose = scope.purpose
        self._scenario = scope.scenario
        self._session = scope.session
        self._tools = scope.tools
        self._commands = scope.commands
        self._providers = scope.providers
        self._renderers = scope.renderers
        self._pending_user_messages = scope.pending_user_messages
        self._model_getter = scope.model_getter
        self._provider_getter = scope.provider_getter
        self._gateway: _SessionGateway = scope.gateway
        self._owner_name = owner_name
        self._stale = False
        self._child_session_factory: ChildSessionFactory = scope.child_session_factory
        self._operations_holder: _OperationsHolder = scope.operations
        self._project_layout: ProjectLayout = scope.project_layout
        self._catalog = scope.catalog
        self._resource_writer_holder: _ResourceWriterHolder = scope.resource_writer
        self._telemetry_holder: _SessionTelemetryHolder = scope.telemetry
        self._services = scope.service_registry

    def mark_stale(self) -> None:
        self._stale = True

    def _assert_active(self) -> None:
        if self._stale:
            raise ExtensionStaleError(
                f"Extension {self._owner_name!r} was reloaded; this api/ctx "
                f"reference is stale. Re-acquire via the new install() call. "
                f"To exit gracefully on reload, catch ExtensionStaleError "
                f"around long-running operations that capture api or ctx."
            )

    # --- Event subscription ------------------------------------------------

    def on(
        self,
        channel: str,
        handler: Any,
        *,
        priority: int = 500,
    ) -> Any:
        self._assert_active()
        return self._bus.on(channel, handler, priority=priority)

    def add_observer(self, callback: Any) -> Any:
        self._assert_active()
        return self._bus.add_observer(callback)

    # --- Registrations ----------------------------------------------------

    def _emit_register(
        self,
        kind: Literal["tool", "command", "provider", "renderer"],
        name: str,
        payload: Any,
    ) -> None:
        from agentm.core.abi.events import ApiRegisterEvent

        self._bus.emit_sync(
            ApiRegisterEvent.CHANNEL,
            ApiRegisterEvent(
                kind=kind,
                name=name,
                extension=current_installing_extension(),
                payload=payload,
            ),
        )

    def register_tool(self, tool: Tool) -> None:
        self._assert_active()
        self._tools.append(tool)
        self._emit_register("tool", tool.name, tool)

    def register_command(self, name: str, spec: CommandSpec) -> None:
        self._assert_active()
        self._commands[name] = spec
        self._emit_register("command", name, spec)

    def register_provider(self, name: str, config: ProviderConfig) -> None:
        self._assert_active()
        self._providers[name] = config
        self._emit_register("provider", name, config)

    def has_provider(self, name: str) -> bool:
        self._assert_active()
        return name in self._providers

    def register_operations(
        self, *, file: FileOperations, bash: BashOperations
    ) -> None:
        """Register the session's Operations bundle (see ExtensionAPI docs)."""
        self._assert_active()
        if self._operations_holder.bundle is not None:
            raise KeyError(
                "Operations bundle is already registered for this session; "
                "an earlier atom called api.register_operations(...). "
                "Only one Operations atom per scenario."
            )
        self._operations_holder.bundle = Operations(file=file, bash=bash)

    def register_message_renderer(self, custom_type: str, renderer: Renderer) -> None:
        self._assert_active()
        self._renderers[custom_type] = renderer
        self._emit_register("renderer", custom_type, renderer)

    def register_tool_renderer(self, tool_name: str, renderer: Renderer) -> None:
        self._assert_active()
        self._renderers[f"tool:{tool_name}"] = renderer
        self._emit_register("renderer", f"tool:{tool_name}", renderer)

    # --- Actions -----------------------------------------------------------

    def send_user_message(self, content: str | list[Any]) -> None:
        """Queue a user message to be prepended to the next ``prompt`` turn."""
        self._assert_active()
        from agentm.core.abi.events import ApiSendUserMessageEvent

        self._pending_user_messages.append(content)
        self._bus.emit_sync(
            ApiSendUserMessageEvent.CHANNEL,
            ApiSendUserMessageEvent(
                extension=current_installing_extension(), content=content
            ),
        )

    async def spawn_child_session(
        self, config: Any | None = None, **kwargs: Any
    ) -> Any:
        """Spawn a child session via the runtime-injected factory."""
        self._assert_active()
        if config is not None and kwargs:
            raise TypeError(
                "spawn_child_session accepts either a config object or keyword args, not both"
            )
        if config is None:
            config = kwargs
        return await self._child_session_factory(config)

    def set_service(self, name: str, obj: Any) -> None:
        self._assert_active()
        if name in self._services:
            raise KeyError(f"service {name!r} is already registered")
        self._services[name] = obj

    def get_service(self, name: str) -> Any | None:
        self._assert_active()
        return self._services.get(name)

    def reload_atom(
        self,
        name: str,
        new_source: str,
        *,
        agent_initiated: bool = True,
        rationale: str | None = None,
    ) -> ReloadResult:
        self._assert_active()
        return self._gateway.reload_atom(
            name,
            new_source,
            agent_initiated=agent_initiated,
            rationale=rationale,
        )

    def install_atom(
        self,
        *,
        name: str,
        source: str,
        target_path: str | None = None,
        config: dict[str, Any] | None = None,
        rationale: str | None = None,
        agent_initiated: bool = True,
    ) -> InstallAtomResult:
        self._assert_active()
        return self._gateway.install_atom(
            name=name,
            source=source,
            target_path=target_path,
            config=config,
            rationale=rationale,
            agent_initiated=agent_initiated,
        )

    def unload_atom(
        self,
        name: str,
        *,
        agent_initiated: bool = True,
    ) -> UnloadAtomResult:
        self._assert_active()
        return self._gateway.unload_atom(
            name,
            agent_initiated=agent_initiated,
        )

    def freeze_current(self, name: str) -> str:
        self._assert_active()
        return self._gateway.freeze_current(name)

    def list_atoms(self) -> list[AtomInfo]:
        self._assert_active()
        return self._gateway.list_atoms()

    def is_constitution_path(self, path: str) -> bool:
        self._assert_active()
        return self._gateway.is_constitution_path(path)

    def get_resource_writer(self) -> ResourceWriter:
        self._assert_active()
        return self._resource_writer_holder.writer

    def get_session_telemetry(self) -> SessionTelemetry:
        """Return this session's :class:`SessionTelemetry` handle.

        Lazily constructs the underlying OTLP/JSON ndjson exporters on the
        first call (see :mod:`agentm.core.runtime.otel_export`); subsequent
        calls return the same handle. The substrate also installs a
        ``SessionShutdownEvent`` handler that drains the batch processors
        and closes the file handles, so atoms do not have to call
        :meth:`SessionTelemetry.shutdown` explicitly.
        """
        self._assert_active()
        return self._telemetry_holder.get()

    def register_resource_writer(self, writer: ResourceWriter) -> None:
        """Replace the session's :class:`ResourceWriter` (see ExtensionAPI docs)."""
        self._assert_active()
        if self._resource_writer_holder.replaced:
            raise KeyError(
                "ResourceWriter is already replaced for this session; "
                "an earlier atom called api.register_resource_writer(...). "
                "Only one ResourceWriter atom per scenario."
            )
        self._resource_writer_holder.writer = writer
        self._resource_writer_holder.replaced = True

    # --- Read-only context -------------------------------------------------

    @property
    def cwd(self) -> str:
        self._assert_active()
        return self._cwd

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def root_session_id(self) -> str:
        return self._root_session_id

    @property
    def parent_session_id(self) -> str | None:
        return self._parent_session_id

    @property
    def purpose(self) -> str:
        return self._purpose

    @property
    def scenario(self) -> str | None:
        return self._scenario

    @property
    def tools(self) -> list[Tool]:
        self._assert_active()
        return self._tools

    @property
    def session(self) -> ReadonlySession:
        self._assert_active()
        return self._session

    @property
    def model(self) -> Model | None:
        self._assert_active()
        return self._model_getter()

    @property
    def provider(self) -> ProviderConfig | None:
        self._assert_active()
        return self._provider_getter()

    @property
    def events(self) -> EventBus:
        """Shared bus access remains valid even from stale API references."""
        return self._bus

    # --- Service facades ----------------------------------------------------

    def get_operations(self) -> Operations:
        """Return the atom-registered operations bundle for this session."""
        self._assert_active()
        bundle = self._operations_holder.bundle
        if bundle is None:
            raise RuntimeError(
                "no atom registered Operations; the active scenario manifest "
                "must list an atom that calls api.register_operations(...) "
                "(default: agentm.extensions.builtin.operations_local)"
            )
        return bundle

    def get_project_layout(self) -> ProjectLayout:
        self._assert_active()
        return self._project_layout

    @property
    def catalog(self) -> CatalogService:
        self._assert_active()
        return self._catalog


# --- Loader -----------------------------------------------------------------


def load_extension(
    module_path: str,
    api: ExtensionAPI,
    config: dict[str, Any],
) -> None | Awaitable[None]:
    """Import ``module_path`` and invoke its ``install(api, config)``.

    Returns whatever ``install`` returns:
    - ``None`` for sync extensions (caller need not await).
    - An awaitable for async extensions (caller must await).

    Raises ``ExtensionLoadError`` on any failure (missing module, missing
    ``install`` symbol, exception thrown by ``install`` itself).

    While ``install`` runs, ``current_installing_extension()`` returns
    ``module_path`` so observers can attribute side effects.
    """

    try:
        module = importlib.import_module(module_path)
    except Exception as exc:  # noqa: BLE001
        raise ExtensionLoadError(module_path, exc) from exc

    install = getattr(module, "install", None)
    if install is None or not callable(install):
        raise ExtensionLoadError(
            module_path,
            AttributeError(f"module {module_path!r} has no callable 'install' symbol"),
        )

    token = _INSTALLING_EXTENSION.set(module_path)
    try:
        result = install(api, config)
    except Exception as exc:  # noqa: BLE001
        _INSTALLING_EXTENSION.reset(token)
        raise ExtensionLoadError(module_path, exc) from exc
    if not inspect.isawaitable(result):
        _INSTALLING_EXTENSION.reset(token)
        return None
    awaitable_result = result
    _INSTALLING_EXTENSION.reset(token)

    async def _await_install() -> None:
        inner_token = _INSTALLING_EXTENSION.set(module_path)
        try:
            await awaitable_result
        except Exception as exc:  # noqa: BLE001
            raise ExtensionLoadError(module_path, exc) from exc
        finally:
            _INSTALLING_EXTENSION.reset(inner_token)

    return _await_install()


__all__ = [
    "AtomInfo",
    "ChildSessionFactory",
    "CommandDispatcher",
    "CommandDispatchResult",
    "CommandSpec",
    "ExtensionAPI",
    "ExtensionAPIScope",
    "ExtensionFactory",
    "ExtensionLoadError",
    "ExtensionStaleError",
    "Handler",
    "InstallAtomResult",
    "ProviderConfig",
    "ReadonlySession",
    "ReloadResult",
    "Renderer",
    "UnknownCommandError",
    "UnloadAtomResult",
    "Unsubscribe",
    "SessionTelemetryFactory",
    "_ExtensionAPIImpl",
    "_NoopChildSessionFactory",
    "_NoopSessionGateway",
    "_OperationsHolder",
    "_SessionGateway",
    "_SessionTelemetryHolder",
    "build_extension_api_scope",
    "load_extension",
]
