"""Extension model, ExtensionAPI, and loader.

Implements §2 (Extension Model) and §3 (ExtensionAPI v0 surface) of
``.claude/designs/extension-as-scenario.md``. An extension is a Python module
exporting an ``install(api, config)`` callable. The harness loads it via
``importlib`` and invokes it. There is no class hierarchy, no manifest, no
privileged path: built-ins, providers, and user plugins are all the same shape.

Pluggability hard rule (see ``.claude/designs/pluggable-architecture.md`` §1):
this module imports only stdlib + ``agentm.core.abi``. It MUST NOT import
any legacy ``agentm.harness.*`` module.
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

from agentm.core.abi import (
    AgentMessage,
    BusPriority,
    EventBus,
    LoopConfig,
    Model,
    ObserverRegistration,
    ProviderConfig,
    Tool,
)
from agentm.core.abi.operations import (
    BashOperations,
    FileOperations,
    Operations,
)
from agentm.core.abi.project_layout import ProjectLayout
from agentm.harness.services import (
    CatalogService,
    CompactionService,
    PromptTemplatesService,
    SkillsService,
    default_catalog_service,
    default_compaction_service,
    default_project_layout,
    default_prompt_templates_service,
    default_skills_service,
)
from agentm.harness.resource_writer import (
    GitBackedResourceWriter,
    ResourceWriter,
)


# --- Type aliases ----------------------------------------------------------

ExtensionFactory = Callable[["ExtensionAPI", dict[str, Any]], "None | Awaitable[None]"]
"""The single callable shape every extension exports as ``install``."""

Renderer = Callable[[Any], str]
"""Renders a custom message payload to a string. Placeholder for v0 — actual
UI rendering is deferred to the mode layer."""

Handler = Callable[[Any], Any] | Callable[[Any], Awaitable[Any]]
"""Event handler signature, mirroring the kernel ``EventBus`` contract."""

Unsubscribe = Callable[[], None]
"""Returned by ``ExtensionAPI.on``; calling it removes the subscription."""


class _SessionGateway(Protocol):
    def reload_atom(
        self,
        name: str,
        new_source: str,
        *,
        agent_initiated: bool = True,
        rationale: str | None = None,
    ) -> "ReloadResult": ...

    def install_atom(
        self,
        *,
        name: str,
        source: str,
        target_path: str | None,
        config: dict[str, Any] | None,
        rationale: str | None,
        agent_initiated: bool,
    ) -> "InstallAtomResult": ...

    def unload_atom(
        self,
        name: str,
        *,
        agent_initiated: bool = True,
        rationale: str | None = None,
    ) -> "UnloadAtomResult": ...

    def freeze_current(self, name: str) -> str: ...

    def list_atoms(self) -> list["AtomInfo"]: ...

    def is_constitution_path(self, path: str) -> bool: ...


class _NoopSessionGateway:
    def reload_atom(
        self,
        name: str,
        new_source: str,
        *,
        agent_initiated: bool = True,
        rationale: str | None = None,
    ) -> "ReloadResult":
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
    ) -> "InstallAtomResult":
        del name, source, target_path, config, rationale, agent_initiated
        raise RuntimeError("install_atom is unavailable on this ExtensionAPI instance")

    def unload_atom(
        self,
        name: str,
        *,
        agent_initiated: bool = True,
        rationale: str | None = None,
    ) -> "UnloadAtomResult":
        del name, agent_initiated, rationale
        raise RuntimeError("unload_atom is unavailable on this ExtensionAPI instance")

    def freeze_current(self, name: str) -> str:
        del name
        raise RuntimeError(
            "freeze_current is unavailable on this ExtensionAPI instance"
        )

    def list_atoms(self) -> list["AtomInfo"]:
        return []

    def is_constitution_path(self, path: str) -> bool:
        del path
        return False


# Forward-declared so we don't have to import ``agentm.harness.session`` here.
# The concrete factory closes over ``AgentSession.create`` and is injected by
# the session at api-construction time. Returning ``Any`` keeps the import
# graph one-way; callers expecting an ``AgentSession`` rely on docs/tests.
ChildSessionFactory = Callable[[Any], Awaitable[Any]]


class _NoopChildSessionFactory:
    async def __call__(self, _config: Any) -> Any:
        raise RuntimeError(
            "spawn_child_session is unavailable on this ExtensionAPI instance"
        )


# Set by ``load_extension`` for the duration of an ``install`` call so
# attribution-aware listeners (e.g. the ``observability`` atom) can tag
# handlers and registrations with the originating extension's module path.
# Reads ``"<unknown>"`` outside of an install scope.
_INSTALLING_EXTENSION: ContextVar[str] = ContextVar(
    "agentm_installing_extension", default="<unknown>"
)


def current_installing_extension() -> str:
    """Return the module path of the extension currently inside ``install``.

    Returns ``"<unknown>"`` when called outside an install scope. Used by the
    ``observability`` atom to attribute handler registrations to extensions.
    """
    return _INSTALLING_EXTENSION.get()


# --- Specs ------------------------------------------------------------------


@dataclass
class CommandSpec:
    """Slash-command registration record.

    The ``handler`` receives the rest-of-line argument string and the
    ``ExtensionAPI`` (so the command may register more things, send messages,
    etc.). It may be sync or async; async handlers are awaited by the runner.
    """

    description: str
    handler: Callable[[str, "ExtensionAPI"], Awaitable[None] | None]


@dataclass(frozen=True, slots=True)
class CommandDispatchResult:
    """Result returned by the command-dispatch service facade."""

    handled: bool
    owner: str | None
    messages: list[AgentMessage]


@runtime_checkable
class CommandDispatcher(Protocol):
    """Typed atom-facing port for registered slash command execution."""

    async def dispatch(self, name: str, args: str) -> CommandDispatchResult: ...


@dataclass(frozen=True, slots=True)
class ReloadResult:
    ok: bool
    name: str
    old_hash: str | None
    new_hash: str | None
    error: str | None = None
    rolled_back: bool = False


@dataclass(frozen=True, slots=True)
class InstallAtomResult:
    """Outcome of ``api.install_atom``. ``new_hash`` is the post-write git
    SHA when the file landed under a managed path; ``None`` in advisory
    mode. ``file_created`` distinguishes a brand-new file from overwriting
    an existing one (the latter is rare but allowed for fixup cases).
    """

    ok: bool
    name: str
    module_path: str | None
    target_path: str | None
    new_hash: str | None
    file_created: bool
    error: str | None = None


@dataclass(frozen=True, slots=True)
class UnloadAtomResult:
    """Outcome of ``api.unload_atom``. The module bytes remain on disk
    and in git history; only the running session forgets the atom.
    """

    ok: bool
    name: str
    module_path: str | None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class AtomInfo:
    name: str
    current_hash: str | None
    tier: int
    api_version: int
    source_path: str | None = None
    config: dict[str, Any] | None = None


# --- Errors ----------------------------------------------------------------


class UnknownCommandError(ValueError):
    """Raised by ``AgentSession.prompt`` when a slash-command name is not
    registered. The message preserves the offending name (with leading slash)
    so callers can surface it to the user verbatim.
    """


class ExtensionLoadError(Exception):
    """Raised when ``load_extension`` cannot import or invoke an extension.

    Wraps the original cause so callers can introspect; the message always
    names the offending module path for debuggability.
    """

    def __init__(self, module_path: str, cause: BaseException | None = None) -> None:
        msg = f"Failed to load extension {module_path!r}"
        if cause is not None:
            msg += f": {cause!r}"
        super().__init__(msg)
        self.module_path = module_path
        self.cause = cause


class ExtensionStaleError(RuntimeError):
    """Raised when a stale ExtensionAPI reference is used after reload."""


# --- ExtensionAPI Protocol --------------------------------------------------


@runtime_checkable
class ReadonlySession(Protocol):
    """Window into the active ``SessionManager`` for extensions.

    Mostly read-only — the one mutation is ``append_entry``, which lets an
    extension persist a structured payload (compaction summary, hypothesis
    snapshot, plan submission, etc.) into the session entry tree without
    exposing the full ``SessionManager`` surface (fork/navigate are off-limits
    to extensions). See ``extension-as-scenario.md`` §10b.7.
    """

    def get_messages(self) -> list[AgentMessage]: ...

    def get_branch(self) -> list[Any]: ...

    def get_leaf_id(self) -> str | None: ...

    def get_entry(self, entry_id: str) -> Any | None: ...

    def get_loop_config(self) -> LoopConfig: ...

    def append_entry(
        self,
        type: str,
        payload: Any,
        parent_id: str | None = None,
    ) -> str:
        """Append a custom entry to the active branch and return its id.

        ``parent_id`` defaults to the current active leaf, so successive calls
        chain naturally. Returns the new entry's id so callers can build
        parent-pointer chains across multiple appends.
        """
        ...


@runtime_checkable
class ExtensionAPI(Protocol):
    """The full v0 surface every extension sees.

    See design §3 for rationale on why this list is minimal. Methods are added
    only when a real extension needs them.
    """

    # --- Event subscription -------------------------------------------------
    def on(
        self,
        channel: str,
        handler: Handler,
        *,
        priority: int = BusPriority.NORMAL,
    ) -> Unsubscribe: ...
    def add_observer(self, callback: ObserverRegistration) -> Unsubscribe: ...

    # --- Registrations ------------------------------------------------------
    def register_tool(self, tool: Tool) -> None: ...
    def register_command(self, name: str, spec: CommandSpec) -> None: ...
    def register_provider(self, name: str, config: ProviderConfig) -> None: ...
    def register_message_renderer(
        self, custom_type: str, renderer: Renderer
    ) -> None: ...
    def register_tool_renderer(self, tool_name: str, renderer: Renderer) -> None: ...

    # --- Actions ------------------------------------------------------------
    def send_user_message(self, content: str | list[Any]) -> None: ...
    async def spawn_child_session(
        self, config: Any | None = None, **kwargs: Any
    ) -> Any:
        """Create a nested ``AgentSession`` rooted at this one.

        ``config`` is an ``AgentSessionConfig`` (typed ``Any`` here to avoid
        pulling ``agentm.harness.session`` into the §11 import allow-list).
        The harness fills in ``parent_bus`` and ``parent_session_id`` from
        the current session — caller-supplied values for those fields are
        ignored. Returns the constructed child session.

        Replaces the legacy pattern of dynamically importing
        ``agentm.harness.session`` from inside an extension, which used to
        bypass the §11.4.5 import contract.
        """
        ...

    def set_service(self, name: str, obj: Any) -> None: ...
    def get_service(self, name: str) -> Any | None: ...
    def reload_atom(
        self,
        name: str,
        new_source: str,
        *,
        agent_initiated: bool = True,
        rationale: str | None = None,
    ) -> ReloadResult: ...
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
        """Install a brand-new atom into the running session.

        ``name`` becomes the atom's ``MANIFEST.name`` (must agree); ``source``
        is the full module text. ``target_path`` is filesystem path (relative
        to ``cwd`` or absolute) where the source will be written through
        ``ResourceWriter`` so it lands as a git commit. When omitted the
        harness writes to ``<cwd>/.agentm/atoms/<name>.py`` so agent-installed
        atoms are isolated from the framework's builtin tree.

        Hard rejections (return ``ok=False`` with an explanatory ``error``):

        - ``target_path`` is in the constitution
        - an atom with ``name`` is already loaded
        - the source fails the §11 single-file extension validator
        - ``MANIFEST.tier == 2``: agent-installed atoms cannot ship at
          tier 2; promotion requires human review (separate flow)

        Successful installs emit ``ExtensionInstallEvent`` (phase=start/end)
        with ``trigger="agent"`` (or ``"propose_change_approved"`` when the
        request flows through the future approval gate). The atom appears
        in ``api.list_atoms()`` immediately and its registrations are
        active for the next bus event.
        """
        ...

    def unload_atom(
        self,
        name: str,
        *,
        agent_initiated: bool = True,
        rationale: str | None = None,
    ) -> UnloadAtomResult:
        """Remove an installed atom from the running session.

        The on-disk file and git history are untouched; only the live
        session forgets the atom. Reverse of ``install_atom``: removes
        every handler/tool/command/renderer the atom registered, drops
        its module from ``sys.modules``, marks its captured ``ExtensionAPI``
        as stale, and fires ``ExtensionUnloadEvent``.

        Refused (with ``ok=False``):

        - ``name`` not loaded
        - the atom is the active provider (would leave the loop without
          ``stream_fn``)
        - the atom's source is in the constitution layer
        """
        ...

    def freeze_current(self, name: str) -> str: ...
    def list_atoms(self) -> list[AtomInfo]: ...
    def is_constitution_path(self, path: str) -> bool: ...
    def get_resource_writer(self) -> ResourceWriter: ...

    # --- Read-only context --------------------------------------------------
    @property
    def cwd(self) -> str: ...
    @property
    def session_id(self) -> str: ...
    @property
    def tools(self) -> list[Tool]:
        """The live tool-catalog list for the session.

        Returns the same list reference every call — mutations (append,
        replace by index, ``tools[:] = kept``) are visible to the kernel
        and to other extensions on subsequent turns. Atoms that need to
        unregister or wrap registered tools (``tool_filter``,
        ``file_mutation_queue``) rely on this contract.
        """
        ...

    @property
    def session(self) -> ReadonlySession: ...
    @property
    def model(self) -> Model | None: ...
    @property
    def provider(self) -> ProviderConfig | None: ...
    @property
    def events(self) -> EventBus: ...

    # --- Service facades ----------------------------------------------------
    # See ``harness/services.py`` for the per-service Protocols. Atoms reach
    # ``core._internal`` exclusively via these handles so the §11 import
    # contract can forbid ``agentm.core._internal`` outright.
    def get_operations(self) -> Operations:
        """Return the session's constitution-selected operations bundle.

        Operations are injectable at session construction, not replaceable by
        atoms at runtime; there is intentionally no ``register_operations`` API.
        """
        ...

    def get_project_layout(self) -> ProjectLayout: ...
    @property
    def skills(self) -> SkillsService: ...
    @property
    def prompt_templates(self) -> PromptTemplatesService: ...
    @property
    def catalog(self) -> CatalogService: ...
    @property
    def compaction(self) -> CompactionService: ...


# --- Concrete impl ----------------------------------------------------------


class _ExtensionAPIImpl:
    """Concrete ``ExtensionAPI`` owned by an ``AgentSession``.

    Intentionally not exported at the module top-level: callers should never
    construct one directly. ``AgentSession`` builds its own instance during
    ``create``.

    The impl is a thin shim — every meaningful action delegates to the
    session's ``EventBus`` or the registries the session exposes.
    """

    def __init__(
        self,
        *,
        bus: EventBus,
        cwd: str,
        session_id: str,
        session: ReadonlySession,
        tools: list[Tool],
        commands: dict[str, CommandSpec],
        providers: dict[str, ProviderConfig],
        renderers: dict[str, Renderer],
        pending_user_messages: list[str | list[Any]],
        model_getter: Callable[[], Model | None],
        provider_getter: Callable[[], ProviderConfig | None],
        gateway: _SessionGateway | None = None,
        owner_name: str = "<unknown>",
        operations: Operations | None = None,
        skills_service: SkillsService | None = None,
        prompt_templates_service: PromptTemplatesService | None = None,
        catalog_service: CatalogService | None = None,
        compaction_service: CompactionService | None = None,
        project_layout: ProjectLayout | None = None,
        child_session_factory: ChildSessionFactory | None = None,
        resource_writer: ResourceWriter | None = None,
        service_registry: dict[str, Any] | None = None,
    ) -> None:
        self._bus = bus
        self._cwd = cwd
        self._session_id = session_id
        self._session = session
        self._tools = tools
        self._commands = commands
        self._providers = providers
        self._renderers = renderers
        self._pending_user_messages = pending_user_messages
        self._model_getter = model_getter
        self._provider_getter = provider_getter
        self._gateway: _SessionGateway = gateway or _NoopSessionGateway()
        self._owner_name = owner_name
        self._stale = False
        self._child_session_factory: ChildSessionFactory = (
            child_session_factory or _NoopChildSessionFactory()
        )
        self._operations = operations or _default_local_operations(cwd=cwd)
        self._project_layout: ProjectLayout = project_layout or default_project_layout(
            cwd
        )
        self._skills = skills_service or default_skills_service(self._project_layout)
        self._prompt_templates = (
            prompt_templates_service
            or default_prompt_templates_service(self._project_layout)
        )
        self._catalog = catalog_service or default_catalog_service()
        self._compaction = compaction_service or default_compaction_service()
        self._resource_writer = resource_writer or GitBackedResourceWriter(
            cwd=cwd,
            session_id=session_id,
            bus=bus,
        )
        self._services = service_registry if service_registry is not None else {}

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
        handler: Handler,
        *,
        priority: int = BusPriority.NORMAL,
    ) -> Unsubscribe:
        self._assert_active()
        return self._bus.on(channel, handler, priority=priority)

    def add_observer(self, callback: ObserverRegistration) -> Unsubscribe:
        self._assert_active()
        return self._bus.add_observer(callback)

    # --- Registrations ----------------------------------------------------

    def _emit_register(
        self,
        kind: Literal["tool", "command", "provider", "renderer"],
        name: str,
        payload: Any,
    ) -> None:
        from agentm.harness.events import ApiRegisterEvent

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
        """Queue a user message to be prepended to the next ``prompt`` turn.

        The session drains this list at the top of ``prompt`` and appends each
        queued message before the caller's text. Used by ``sub_agent``'s
        ``inject_instruction`` and any extension that wants to nudge the
        conversation without driving a synchronous turn.
        """
        self._assert_active()
        from agentm.harness.events import ApiSendUserMessageEvent

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
        """Spawn a child session via the harness-injected factory.

        See ``ExtensionAPI.spawn_child_session`` for the contract; the
        factory is closed-over by ``AgentSession.create``.
        """
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
        rationale: str | None = None,
    ) -> UnloadAtomResult:
        self._assert_active()
        return self._gateway.unload_atom(
            name,
            agent_initiated=agent_initiated,
            rationale=rationale,
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
        return self._resource_writer

    # --- Read-only context -------------------------------------------------

    @property
    def cwd(self) -> str:
        self._assert_active()
        return self._cwd

    @property
    def session_id(self) -> str:
        return self._session_id

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
        """Return the session's construction-time operations bundle."""
        self._assert_active()
        return self._operations

    def get_project_layout(self) -> ProjectLayout:
        self._assert_active()
        return self._project_layout

    @property
    def skills(self) -> SkillsService:
        self._assert_active()
        return self._skills

    @property
    def prompt_templates(self) -> PromptTemplatesService:
        self._assert_active()
        return self._prompt_templates

    @property
    def catalog(self) -> CatalogService:
        self._assert_active()
        return self._catalog

    @property
    def compaction(self) -> CompactionService:
        self._assert_active()
        return self._compaction


def _default_local_operations(cwd: str | None = None) -> Operations:
    """Build the default ``Operations`` bundle backed by local stdlib I/O.

    ``cwd`` — when provided — anchors relative paths handed to the file ops
    against the session's working directory rather than the process's. The
    bash op already plumbs cwd through ``asyncio.create_subprocess_shell``,
    so without this anchor relative paths in ``read``/``write``/``edit``/
    ``find``/``grep`` resolve against whatever directory the operator
    happened to launch ``agentm`` from, while ``bash`` resolves against
    ``--cwd``. The split is invisible to the agent until it reads ``foo.py``
    and gets ``[Errno 2] No such file`` next to a ``bash cat foo.py`` that
    works — at which point the loop is unworkable.

    Imports the concrete impls lazily so the ``ExtensionAPI`` module itself
    does not pull in ``core._internal`` at import time (the harness is
    constitution and may import it, but keeping the dependency lazy lets
    test fixtures supply alternative bundles without any teardown).
    """

    from agentm.core._internal.operations_impl import (
        LocalBashOperations,
        LocalFileOperations,
    )

    file_ops: FileOperations = LocalFileOperations(cwd=cwd)
    bash_ops: BashOperations = LocalBashOperations()
    return Operations(file=file_ops, bash=bash_ops)


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
    # Reset eagerly here — sync portion of install() has finished. The
    # awaitable runs in its own task/context where the contextvar default
    # already reads ``module_path`` only if the caller awaits inside the
    # same context (which is fine for the harness's sequential await).
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
    "CommandDispatcher",
    "CommandDispatchResult",
    "CommandSpec",
    "ExtensionAPI",
    "ExtensionFactory",
    "ExtensionLoadError",
    "ExtensionStaleError",
    "Handler",
    "ProviderConfig",
    "ReadonlySession",
    "ReloadResult",
    "Renderer",
    "Unsubscribe",
    "UnknownCommandError",
    "current_installing_extension",
    "load_extension",
]
