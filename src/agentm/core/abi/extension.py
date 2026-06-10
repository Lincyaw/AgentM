"""ExtensionAPI atom-facing surface — Protocols, dataclasses, exceptions.

Implements §2 (Extension Model) and §3 (ExtensionAPI v0 surface) of
``.claude/designs/extension-as-scenario.md``. This module is part of the
ABI layer — atoms import from here. The matching impl (``_ExtensionAPIImpl``,
``ExtensionAPIScope``, ``build_extension_api_scope``, ``load_extension``,
``_OperationsHolder``) lives in :mod:`agentm.core.runtime.extension`.

Pluggability hard rule (see ``.claude/designs/pluggable-architecture.md`` §1):
this module imports only stdlib + other ``agentm.core.abi`` submodules. It
MUST NOT import anything from ``agentm.core.runtime``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextlib import AbstractContextManager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

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
from agentm.core.abi.catalog import CatalogService
from agentm.core.abi.operations import (
    BashOperations,
    FileOperations,
    Operations,
)
from agentm.core.abi.project_layout import ProjectLayout
from agentm.core.abi.resource import ResourceWriter

if TYPE_CHECKING:
    # Re-exported through ``agentm.core.abi`` for atoms (see
    # ``agentm.core.abi.__init__``). Importing here under ``TYPE_CHECKING``
    # keeps the ABI module side-effect-free at import time and preserves
    # the "abi never imports runtime at runtime" property — the symbol is
    # used only for type annotations on the Protocol method below.
    from agentm.core.runtime.otel_export import SessionTelemetry


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

# Forward-declared so callers don't have to import ``agentm.core.runtime.session`` here.
# The concrete factory closes over ``AgentSession.create`` and is injected by
# the session at api-construction time. Returning ``Any`` keeps the import
# graph one-way; callers expecting an ``AgentSession`` rely on docs/tests.
ChildSessionFactory = Callable[[Any], Awaitable[Any]]


# --- Install-attribution ContextVar ---------------------------------------

# Set by ``load_extension`` for the duration of an ``install`` call so
# attribution-aware listeners (e.g. the ``observability`` atom) can tag
# handlers and registrations with the originating extension's module path.
# Reads ``"<unknown>"`` outside of an install scope.
#
# Single owner: the runtime loader imports this same ContextVar and writes
# to it. Duplicating the var elsewhere silently breaks attribution because
# the writer and reader would see different storage.
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


# --- Session gateway (impl-facing Protocol used by api wiring) -------------


class _SessionGateway(Protocol):
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
        target_path: str | None,
        config: dict[str, Any] | None,
        rationale: str | None,
        agent_initiated: bool,
    ) -> InstallAtomResult: ...

    def unload_atom(
        self,
        name: str,
        *,
        agent_initiated: bool = True,
    ) -> UnloadAtomResult: ...

    def freeze_current(self, name: str) -> str: ...

    def list_atoms(self) -> list[AtomInfo]: ...

    def is_constitution_path(self, path: str) -> bool: ...


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

    def get_session_id(self) -> str:
        """Return the persisted ``SessionManager`` header id, or empty
        string when the session is in-memory / not yet persisted.

        Distinct from :attr:`ExtensionAPI.root_session_id` (the OTel
        trace_id assigned by the substrate at session-construction time):
        atoms that need to correlate sidecar files with the on-disk
        session JSONL must use this id, since the persisted file name is
        derived from the manager header — see
        ``JsonlSessionManager.new_session``.
        """
        ...

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


# Catalog Protocol — forward import (lives in core.abi.catalog after Step 4).
# Until then, importers use ``Any`` for the property type via TYPE_CHECKING.
# We rely on duck-typing at runtime: the impl wires a real CatalogService.


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
    def has_provider(self, name: str) -> bool: ...

    def register_operations(
        self, *, file: FileOperations, bash: BashOperations
    ) -> None:
        """Register the session's ``Operations`` bundle (at most once)."""
        ...

    def register_resource_writer(self, writer: ResourceWriter) -> None:
        """Override the session's ``ResourceWriter`` (at most once)."""
        ...

    # --- Actions ------------------------------------------------------------
    def post_inbox(
        self,
        *,
        source: str,
        payload: Any,
        dedup_key: str | None = None,
        terminal: bool = False,
    ) -> None:
        """Push an item onto the session inbox for the next turn.

        ``source`` routes rendering; ``dedup_key`` replaces an earlier
        same-key item; ``terminal=True`` stops the loop after delivery.
        """
        ...

    def track_background(self) -> AbstractContextManager[None]:
        """Context manager that keeps the session non-idle while a background unit runs."""
        ...

    def send_user_message(self, content: str | list[Any]) -> None: ...
    async def spawn_child_session(
        self, config: Any | None = None, **kwargs: Any
    ) -> Any:
        """Create a child ``AgentSession`` inheriting this session's context."""
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
        """Install a new atom into the running session."""
        ...

    def unload_atom(
        self,
        name: str,
        *,
        agent_initiated: bool = True,
    ) -> UnloadAtomResult:
        """Remove an atom from the running session (on-disk files untouched)."""
        ...

    def list_atoms(self) -> list[AtomInfo]: ...
    def get_resource_writer(self) -> ResourceWriter: ...
    def get_session_telemetry(self) -> SessionTelemetry:
        """Return this session's :class:`SessionTelemetry` (lazily constructed)."""
        ...

    # --- Read-only context --------------------------------------------------
    @property
    def cwd(self) -> str: ...
    @property
    def session_id(self) -> str:
        """OTel span_id (16 hex) for this session."""
        ...
    @property
    def root_session_id(self) -> str:
        """OTel trace_id (32 hex) shared across the agent tree."""
        ...
    @property
    def parent_session_id(self) -> str | None:
        """Parent's session_id, or ``None`` for root sessions."""
        ...
    @property
    def purpose(self) -> str: ...
    @property
    def scenario(self) -> str | None: ...
    @property
    def tools(self) -> list[Tool]:
        """Live mutable tool list; same reference every call."""
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
    def get_operations(self) -> Operations: ...

    def get_project_layout(self) -> ProjectLayout: ...
    @property
    def catalog(self) -> CatalogService: ...


__all__ = [
    "AtomInfo",
    "CatalogService",
    "ChildSessionFactory",
    "CommandDispatcher",
    "CommandDispatchResult",
    "CommandSpec",
    "ExtensionAPI",
    "ExtensionFactory",
    "ExtensionLoadError",
    "ExtensionStaleError",
    "Handler",
    "InstallAtomResult",
    "ProviderConfig",
    "ReadonlySession",
    "ReloadResult",
    "Renderer",
    "Unsubscribe",
    "UnknownCommandError",
    "UnloadAtomResult",
    "_INSTALLING_EXTENSION",
    "_SessionGateway",
    "current_installing_extension",
]
