"""MCP client wrapper — owns transport lifecycle for one server.

This module is a thin async adapter around the official ``mcp`` Python SDK.
We do not hand-roll the JSON-RPC framing — the SDK's ``ClientSession``,
``stdio_client``, and ``streamablehttp_client`` already do that correctly.

The wrapper exists for one reason: tying the transport's
``AsyncExitStack`` lifetime to *our* shutdown call so the session-close
hook can deterministically tear every server down in a single ``await``.

See ``.claude/designs/mcp-integration.md`` §4 for the broader contract.
"""

from __future__ import annotations

import os
import re
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any, Final, Protocol


# --- Config dataclasses ----------------------------------------------------


@dataclass(frozen=True, slots=True)
class StdioServerSpec:
    """stdio-transport server: spawn ``command`` as a subprocess."""

    name: str
    command: list[str]
    env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class HttpServerSpec:
    """http-transport server: connect to a streamable-HTTP MCP endpoint."""

    name: str
    url: str
    headers: dict[str, str] = field(default_factory=dict)


ServerSpec = StdioServerSpec | HttpServerSpec


# --- ${VAR} interpolation --------------------------------------------------

_INTERP_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


# --- Test seam (module-level, not in config schema) -----------------------
#
# Tests that want to point the bridge at an in-process stub MCP server
# call :func:`set_test_session_factory` during setup; ``install`` consumes
# the value exactly once via :func:`consume_test_session_factory`. Keeping
# this out of the public config schema means scenario authors can't
# accidentally smuggle a callable through YAML.

_TEST_SESSION_FACTORY: Any = None


def set_test_session_factory(factory: Any) -> None:
    """Install an async factory ``(spec, exit_stack) -> _SessionLike``.

    Single-shot — the next ``install()`` consumes and clears it.
    """

    global _TEST_SESSION_FACTORY
    _TEST_SESSION_FACTORY = factory


def consume_test_session_factory() -> Any:
    """Return and clear the most recently installed test factory, if any."""

    global _TEST_SESSION_FACTORY
    factory = _TEST_SESSION_FACTORY
    _TEST_SESSION_FACTORY = None
    return factory


def _interp(value: str) -> str:
    """Replace ``${VAR}`` with ``os.environ['VAR']``.

    Missing variables resolve to the empty string — the typical use case is
    bearer tokens, and an empty header value will fail loudly upstream
    rather than silently falling back to an accidentally-correct default.
    """

    return _INTERP_RE.sub(lambda m: os.environ.get(m.group(1), ""), value)


def _interp_dict(d: dict[str, str]) -> dict[str, str]:
    return {k: _interp(v) for k, v in d.items()}


def parse_server_spec(raw: dict[str, Any]) -> ServerSpec:
    """Parse one entry from ``config['servers']`` into a typed spec.

    Raises ``ValueError`` on missing required fields or unknown transport.
    """

    name = raw.get("name")
    transport = raw.get("transport")
    if not isinstance(name, str) or not name:
        raise ValueError(f"mcp_bridge: server entry missing 'name': {raw!r}")
    if transport == "stdio":
        command = raw.get("command")
        if not isinstance(command, list) or not command:
            raise ValueError(
                f"mcp_bridge: server {name!r}: stdio transport requires "
                "non-empty 'command' list"
            )
        env = raw.get("env") or {}
        if not isinstance(env, dict):
            raise ValueError(f"mcp_bridge: server {name!r}: 'env' must be a dict")
        return StdioServerSpec(
            name=name,
            command=[str(c) for c in command],
            env=_interp_dict({str(k): str(v) for k, v in env.items()}),
        )
    if transport == "http":
        url = raw.get("url")
        if not isinstance(url, str) or not url:
            raise ValueError(
                f"mcp_bridge: server {name!r}: http transport requires 'url'"
            )
        headers = raw.get("headers") or {}
        if not isinstance(headers, dict):
            raise ValueError(
                f"mcp_bridge: server {name!r}: 'headers' must be a dict"
            )
        return HttpServerSpec(
            name=name,
            url=_interp(url),
            headers=_interp_dict({str(k): str(v) for k, v in headers.items()}),
        )
    raise ValueError(
        f"mcp_bridge: server {name!r}: unknown transport {transport!r} "
        "(expected 'stdio' or 'http')"
    )


# --- Client session protocol (duck-typed for testability) -----------------


class _SessionLike(Protocol):
    """Subset of :class:`mcp.ClientSession` the bridge actually uses.

    Declared here so the test stub can provide an in-memory session
    without depending on the real SDK transports.
    """

    async def initialize(self) -> Any: ...
    async def list_tools(self) -> Any: ...
    async def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> Any: ...


# --- Manager that owns the AsyncExitStack lifetime -------------------------


@dataclass(slots=True)
class _Connected:
    spec: ServerSpec
    session: _SessionLike


class MCPSessionManager:
    """Owns the open MCP client sessions for one AgentM session.

    Construct via :meth:`connect_all`; tear down via :meth:`aclose`.

    The manager is the single object published under
    ``api.set_service("mcp", ...)``. Follow-up atoms (``mcp_prompts`` /
    ``mcp_resources``) reuse the live sessions through this handle.
    """

    def __init__(self) -> None:
        self._stack = AsyncExitStack()
        self._servers: dict[str, _Connected] = {}
        self._closed = False

    @property
    def servers(self) -> dict[str, _Connected]:
        return self._servers

    def get_session(self, server: str) -> _SessionLike:
        if self._closed:
            raise RuntimeError("mcp_bridge: manager closed")
        try:
            return self._servers[server].session
        except KeyError as exc:
            raise KeyError(f"mcp_bridge: unknown server {server!r}") from exc

    async def connect_all(
        self,
        specs: list[ServerSpec],
        *,
        session_factory: Any = None,
    ) -> None:
        """Open transports + ``initialize`` for each spec, in order.

        On *any* failure the whole stack is closed and the original
        exception is re-raised — snapshot semantics require a fully-known
        tool set or none at all (design §8).

        ``session_factory`` is an optional injection seam: when provided,
        it must be an async callable taking the :class:`ServerSpec` and
        the manager's :class:`AsyncExitStack` and returning a
        :class:`_SessionLike`. Used by the test suite to inject an
        in-process stub session without spawning real subprocesses.
        """

        try:
            for spec in specs:
                if session_factory is not None:
                    session = await session_factory(spec, self._stack)
                else:
                    session = await self._open_real(spec)
                await session.initialize()
                self._servers[spec.name] = _Connected(spec=spec, session=session)
        except BaseException:
            await self.aclose()
            raise

    async def _open_real(self, spec: ServerSpec) -> _SessionLike:
        """Open a real MCP transport using the official SDK.

        Kept tiny so the test path (which bypasses this entirely via
        ``session_factory``) carries no SDK-specific edge cases.
        """

        # Imported lazily so importing ``mcp_bridge`` (e.g. for catalog
        # introspection) does not eagerly pull the SDK's anyio stack.
        from mcp import ClientSession, StdioServerParameters  # type: ignore[import-not-found]
        from mcp.client.stdio import stdio_client  # type: ignore[import-not-found]
        from mcp.client.streamable_http import (  # type: ignore[import-not-found]
            streamablehttp_client,
        )

        if isinstance(spec, StdioServerSpec):
            params = StdioServerParameters(
                command=spec.command[0],
                args=list(spec.command[1:]),
                env=dict(spec.env) or None,
            )
            read, write = await self._stack.enter_async_context(stdio_client(params))
        elif isinstance(spec, HttpServerSpec):
            read, write, _ = await self._stack.enter_async_context(
                streamablehttp_client(spec.url, headers=spec.headers)
            )
        else:  # pragma: no cover - exhaustive match
            raise ValueError(f"unhandled server spec: {spec!r}")

        session = await self._stack.enter_async_context(ClientSession(read, write))
        return session  # type: ignore[no-any-return]

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._stack.aclose()
        self._servers.clear()


__all__: Final = [
    "HttpServerSpec",
    "MCPSessionManager",
    "ServerSpec",
    "StdioServerSpec",
    "_SessionLike",
    "consume_test_session_factory",
    "parse_server_spec",
    "set_test_session_factory",
]
