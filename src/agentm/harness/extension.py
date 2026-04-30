"""Extension model, ExtensionAPI, and loader.

Implements §2 (Extension Model) and §3 (ExtensionAPI v0 surface) of
``.claude/designs/extension-as-scenario.md``. An extension is a Python module
exporting an ``install(api, config)`` callable. The harness loads it via
``importlib`` and invokes it. There is no class hierarchy, no manifest, no
privileged path: built-ins, providers, and user plugins are all the same shape.

Pluggability hard rule (see ``.claude/designs/pluggable-architecture.md`` §1):
this module imports only stdlib + ``agentm.core.kernel``. It MUST NOT import
any legacy ``agentm.harness.*`` module.
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from agentm.core.kernel import (
    AgentMessage,
    EventBus,
    Model,
    StreamFn,
    Tool,
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
class ProviderConfig:
    """LLM provider registration record.

    A provider extension calls ``api.register_provider(name, ProviderConfig)``
    to publish its ``StreamFn`` and default ``Model``. The harness picks the
    most recently registered provider as the active one. Frozen so an
    extension cannot silently mutate another extension's registration.
    """

    stream_fn: StreamFn
    model: Model
    name: str


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
    def on(self, channel: str, handler: Handler) -> Unsubscribe: ...

    # --- Registrations ------------------------------------------------------
    def register_tool(self, tool: Tool) -> None: ...
    def register_command(self, name: str, spec: CommandSpec) -> None: ...
    def register_provider(self, name: str, config: ProviderConfig) -> None: ...
    def register_message_renderer(
        self, custom_type: str, renderer: Renderer
    ) -> None: ...

    # --- Actions ------------------------------------------------------------
    def send_user_message(self, content: str | list[Any]) -> None: ...

    # --- Read-only context --------------------------------------------------
    @property
    def cwd(self) -> str: ...
    @property
    def tools(self) -> list[Tool]: ...
    @property
    def session(self) -> ReadonlySession: ...
    @property
    def model(self) -> Model | None: ...
    @property
    def provider(self) -> ProviderConfig | None: ...
    @property
    def events(self) -> EventBus: ...


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
        session: ReadonlySession,
        tools: list[Tool],
        commands: dict[str, CommandSpec],
        providers: dict[str, ProviderConfig],
        renderers: dict[str, Renderer],
        pending_user_messages: list[str | list[Any]],
        model_getter: Callable[[], Model | None],
        provider_getter: Callable[[], ProviderConfig | None],
    ) -> None:
        self._bus = bus
        self._cwd = cwd
        self._session = session
        self._tools = tools
        self._commands = commands
        self._providers = providers
        self._renderers = renderers
        self._pending_user_messages = pending_user_messages
        self._model_getter = model_getter
        self._provider_getter = provider_getter

    # --- Event subscription ------------------------------------------------

    def on(self, channel: str, handler: Handler) -> Unsubscribe:
        return self._bus.on(channel, handler)

    # --- Registrations ----------------------------------------------------

    def register_tool(self, tool: Tool) -> None:
        self._tools.append(tool)

    def register_command(self, name: str, spec: CommandSpec) -> None:
        self._commands[name] = spec

    def register_provider(self, name: str, config: ProviderConfig) -> None:
        self._providers[name] = config

    def register_message_renderer(
        self, custom_type: str, renderer: Renderer
    ) -> None:
        self._renderers[custom_type] = renderer

    # --- Actions -----------------------------------------------------------

    def send_user_message(self, content: str | list[Any]) -> None:
        """Queue a user message to be prepended to the next ``prompt`` turn.

        The session drains this list at the top of ``prompt`` and appends each
        queued message before the caller's text. Used by ``sub_agent``'s
        ``inject_instruction`` and any extension that wants to nudge the
        conversation without driving a synchronous turn.
        """
        self._pending_user_messages.append(content)

    # --- Read-only context -------------------------------------------------

    @property
    def cwd(self) -> str:
        return self._cwd

    @property
    def tools(self) -> list[Tool]:
        return self._tools

    @property
    def session(self) -> ReadonlySession:
        return self._session

    @property
    def model(self) -> Model | None:
        return self._model_getter()

    @property
    def provider(self) -> ProviderConfig | None:
        return self._provider_getter()

    @property
    def events(self) -> EventBus:
        return self._bus


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
    """

    try:
        module = importlib.import_module(module_path)
    except Exception as exc:  # noqa: BLE001
        raise ExtensionLoadError(module_path, exc) from exc

    install = getattr(module, "install", None)
    if install is None or not callable(install):
        raise ExtensionLoadError(
            module_path,
            AttributeError(
                f"module {module_path!r} has no callable 'install' symbol"
            ),
        )

    try:
        result = install(api, config)
    except Exception as exc:  # noqa: BLE001
        raise ExtensionLoadError(module_path, exc) from exc

    if inspect.isawaitable(result):
        return result
    return None


__all__ = [
    "CommandSpec",
    "ExtensionAPI",
    "ExtensionFactory",
    "ExtensionLoadError",
    "Handler",
    "ProviderConfig",
    "ReadonlySession",
    "Renderer",
    "Unsubscribe",
    "UnknownCommandError",
    "load_extension",
]
