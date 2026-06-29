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
from typing import Any, Protocol, runtime_checkable

from agentm.core.abi import (
    AgentMessage,
    BusPriority,
    EventBus,
    Handler,
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
from agentm.core.abi.telemetry import SessionTelemetry


# --- Type aliases ----------------------------------------------------------

ExtensionFactory = Callable[["ExtensionAPI", dict[str, Any]], "None | Awaitable[None]"]
"""The single callable shape every extension exports as ``install``."""

Renderer = Callable[[Any], str]
"""Renders a custom message payload to a string. Placeholder for v0 — actual
UI rendering is deferred to the mode layer."""

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
            msg += f": {cause}"
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
    def has_provider(self, name: str) -> bool:
        """Return ``True`` if a provider has already been registered under ``name``.

        Atoms that publish under a dynamic name (e.g. the OpenAI-compatible
        ``llm_openai`` atom may register as ``"openai"`` / ``"doubao"`` /
        ``"litellm"`` depending on config) use this to short-circuit duplicate
        installations cleanly instead of relying on ``LastRegisteredWins``
        silently shadowing a previous registration.
        """
        ...

    def register_operations(
        self, *, file: FileOperations, bash: BashOperations
    ) -> None:
        """Register the session's ``Operations`` bundle.

        Must be called at most once before freeze; a second call raises
        ``KeyError`` (mirroring ``set_service`` semantics for the
        already-registered case). If no atom calls this by freeze time,
        ``get_operations()`` will raise — the default scenario manifest
        is expected to list an atom that registers a bundle (the
        ``operations`` builtin covers the local-FS / asyncio-bash
        case).
        """
        ...

    def register_resource_writer(self, writer: ResourceWriter) -> None:
        """Replace the session's ``ResourceWriter``.

        Must be called at most once before freeze; a second call raises
        ``KeyError``. Atoms that want to redirect ``write`` /
        ``edit`` / ``tool_propose_change`` writes to a non-default
        target (e.g. a sandbox-backed filesystem) call this at install
        time. If no atom calls it, ``get_resource_writer()`` returns the
        default :class:`~agentm.core.runtime.resource_writer.GitBackedResourceWriter`
        the substrate constructed from ``cwd``.

        Implementations must honour the constitution-path refusal contract
        of :class:`ResourceWriter`. Sandbox-backed writers typically reject
        every host path (the sandbox cannot see the AgentM tree) and
        accept only paths inside their managed work-dir.
        """
        ...

    def register_message_renderer(
        self, custom_type: str, renderer: Renderer
    ) -> None: ...
    def register_tool_renderer(self, tool_name: str, renderer: Renderer) -> None: ...

    # --- Actions ------------------------------------------------------------
    def post_inbox(
        self,
        *,
        source: str,
        payload: Any,
        dedup_key: str | None = None,
        terminal: bool = False,
    ) -> None:
        """Push an item onto the session inbox — the generic producer entry.

        ``source`` is the mechanism-level routing tag (``"user"`` /
        ``"background"`` / ``"ticker"`` / ``"monitor"`` / ``"subagent"``) that
        decides how the item is rendered at the next turn boundary (see
        ``SessionInbox.render_item``). ``payload`` is rendered per ``source``.
        A producer that supersedes its own prior, not-yet-drained item passes a
        stable ``dedup_key``: a later push with the same key replaces the
        earlier item in place rather than stacking (e.g. a background ticker's
        rolling status line).

        ``terminal=True`` carries a *terminate intent* into the loop (#177): a
        backgrounded tool whose detached completion is a
        :class:`ToolTerminate` posts with ``terminal=True`` so the runtime stops
        the loop (``ToolTerminated``) once the item is delivered, rather than
        keeping the agent alive on the non-empty inbox. The message still lands
        in the conversation first.

        :meth:`send_user_message` is the ``source="user"`` sugar over this; new
        producers (``background_exec``, ``monitor``, the future ``sub_agent``
        rewrite) post through ``post_inbox`` directly. See
        ``.claude/designs/session-inbox.md`` (step-3 design decisions).
        """
        ...

    def track_background(self) -> AbstractContextManager[None]:
        """Bracket a detached background unit so the host can wait it out (#179).

        A producer that runs work in an ``asyncio.Task`` outliving the agent's
        turn (auto-backgrounded tools, child subagent sessions) wraps the unit's
        lifetime in this context manager::

            with api.track_background():
                await self._watch(state, task)

        While any tracked unit is live the session is **not idle**: a one-shot
        host (``agentm -p``) blocks in :meth:`AgentSession.idle` until every
        tracked unit finishes, so a late completion is never dropped for lack of
        an event loop. Recurring signals (monitor wakeups / condition polls /
        tickers) deliberately do NOT track — they are not work to drain before
        exit and would keep a one-shot host alive forever. The pairing is
        structural (``__exit__`` always decrements), so an exception in the unit
        cannot leak the count.
        """
        ...

    async def wait_inbox_nonempty(self) -> bool:
        """Wait until the core session inbox has pending items.

        Producer atoms use this as a mechanism-level wakeup only. The method
        does not drain or claim the item; the session's runtime-owned context
        handler remains the only inbox drain site. It returns ``False`` when a
        wakeup was only a driver kick and no item is actually pending.
        """
        ...

    def send_user_message(self, content: str | list[Any]) -> None: ...
    async def spawn_child_session(
        self, config: Any | None = None, **kwargs: Any
    ) -> Any:
        """Create a nested ``AgentSession`` rooted at this one.

        ``config`` is an ``AgentSessionConfig`` (typed ``Any`` here to avoid
        pulling ``agentm.core.runtime.session`` into the import allow-list).
        The runtime fills in ``parent_bus`` and ``parent_session_id`` from
        the current session — caller-supplied values for those fields are
        ignored. Returns the constructed child session.
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
        runtime writes to ``<cwd>/.agentm/atoms/<name>.py`` so agent-installed
        atoms are isolated from the framework's builtin tree.

        Hard rejections (return ``ok=False`` with an explanatory ``error``):

        - ``target_path`` is in the constitution
        - an atom with ``name`` is already loaded
        - the source fails the single-file extension validator
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
    def get_session_telemetry(self) -> "SessionTelemetry":
        """Return this session's :class:`SessionTelemetry` handle."""
        ...

    # --- Read-only context --------------------------------------------------
    @property
    def cwd(self) -> str: ...
    @property
    def scenario_dir(self) -> str | None:
        """Directory containing the loaded scenario manifest, or None."""
        ...
    @property
    def session_id(self) -> str:
        """This session's OTel ``span_id`` — 8 bytes / 16 hex chars.
        Always set; identifies the *session-root span* inside the trace.
        The observability sink uses it as the JSONL filename so each
        session lands in ``$AGENTM_HOME/observability/<session_id>.jsonl``
        by default.
        Cross-process embedders that already maintain an OTel span id
        can supply it on :class:`AgentSessionConfig.session_id`."""
        ...

    @property
    def root_session_id(self) -> str:
        """The OTel ``trace_id`` — 16 bytes / 32 hex chars — shared
        across the whole agent tree (this session + every transitive
        child). The observability sink stamps it as the ``trace_id``
        field of every event line so a single ``trace_id =`` filter
        recovers the entire trace regardless of which JSONL file each
        span lives in. For a session with no parent the substrate
        generates a fresh trace_id; for spawned children it inherits
        from the parent verbatim."""
        ...

    @property
    def parent_session_id(self) -> str | None:
        """``None`` for root sessions; the parent's ``session_id`` for
        any session created via :meth:`spawn_child_session` — the OTel
        ``parent_span_id`` of this session-root span. Surfaced so that
        atoms (notably the observability sink) can chain spans across
        sessions without an external mapping table."""
        ...

    @property
    def purpose(self) -> str:
        """Caller-defined label from :class:`AgentSessionConfig.purpose`,
        defaulting to ``"root"``. Used by the observability sink and any
        atom that needs to discriminate parent vs spawned child sessions
        (e.g. ``cognitive_audit_extractor`` / ``cognitive_audit_auditor``)
        without inferring it from the loaded module list."""
        ...

    @property
    def scenario(self) -> str | None:
        """Scenario name from :class:`AgentSessionConfig.scenario` (e.g.
        ``"rca:harness.sync"``), or ``None`` when atoms were assembled
        directly without going through a manifest. Exposed so the
        observability sink can stamp it onto ``session.start`` and the
        fingerprint span — otherwise the trace file gives no in-band
        signal of which scenario produced it."""
        ...

    @property
    def lineage(self) -> dict[str, Any] | None:
        """Caller-supplied provenance metadata from
        :class:`AgentSessionConfig.lineage`.

        Atoms should treat this as read-only descriptive context for trace
        reconstruction. It has no runtime semantics."""
        ...

    @property
    def experiment(self) -> dict[str, Any] | None:
        """Caller-supplied study metadata from
        :class:`AgentSessionConfig.experiment`.

        Used by evals, ablations, and reminder-injection runs to attach
        external experiment context to a session without coupling that logic to
        the core runtime."""
        ...

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
    def get_operations(self) -> Operations:
        """Return the active ``Operations`` bundle for this session.

        An atom must have called :meth:`register_operations` before
        ``get_operations`` is invoked (the default scenario lists the
        ``operations`` atom which does this at install time). If
        no bundle has been registered, this raises ``RuntimeError``.
        """
        ...

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
