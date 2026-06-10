"""Live-inspector atom — broadcast a session's events over a WebSocket so
an external UI (aegis-ui) can render the agent tree in real time.

This is the substrate for live audit visualization. UI work is out of scope;
this atom defines the wire protocol and gets the events flowing.

Why WebSocket (not raw TCP)
---------------------------

Browsers cannot speak raw TCP, so the UI's backend would otherwise need to
run a TCP-to-WS bridge — defeating the "scenario atom → UI direct connect"
design. ``websockets>=12.0`` is pure-Python, ~70 KB, no transitive deps;
listed in the top-level ``[project] dependencies`` so it ships with every
install.

Wire protocol (v1)
------------------

Transport: WebSocket text frames; one JSON object per frame. The server URL
is logged to stderr at startup as
``LIVE INSPECT: ws://<host>:<port>/inspect?root=<root_session_id>``.

Client → server (first frame, optional):
    ``{"type": "subscribe", "root_session_id": "<id>"}``
    Informational in v1 (one server per root). Sent or omitted, the
    connection is bound to whichever root this server serves.

Server → client (every connection, in order):
    1. Hello:
       ``{"type": "hello", "root_session_id": "<id>", "schema_version": 1}``
    2. For every session in the tree (root + spawned children) as they start:
       ``{"type": "session_started", "session_id", "parent_session_id",
           "purpose", "cwd", "ts"}``
    3. For every EventBus dispatch in any session:
       ``{"type": "event", "session_id", "ts", "kind": "<event class name>",
           "channel": "<bus channel>", "payload": {...}}``
    4. For every entry appended to a session's entry tree (assistant
       messages, ``llmharness.audit_event``, ``llmharness.verdict``,
       ``llmharness.audit_graph_op``, plan submissions, …):
       ``{"type": "entry", "session_id", "ts", "entry_type": "<type>",
           "entry_id", "parent_id", "payload": {...}}``
    5. When a session ends:
       ``{"type": "session_ended", "session_id", "ts"}``
    6. After initial backlog replay:
       ``{"type": "backlog_done"}``

Backpressure
------------

Each connection has a bounded outbound queue (cap ``_QUEUE_HIGHWATER`` =
256). When the queue is full the NEWEST message is dropped and a warning
logged — slow clients cannot deadlock the broadcaster.

Child sessions
--------------

The root ``install()`` subscribes to :class:`ChildSessionExtendingEvent` on
the parent bus — a typed bus channel the substrate fires synchronously
before spawning each child. Our handler returns one
``(module, config)`` tuple contributing the inspector to the child's load
order with ``role: "child"`` + ``parent_root`` + ``parent_port`` baked in.
The substrate dedupes by module path (operator override on the child
config wins), then runs the factory. The child's own ``install()`` finds
the existing server via the process-level ``_SERVERS`` registry keyed by
``root_session_id`` and joins instead of binding a fresh one. Net effect:
one WebSocket socket, the whole agent tree.

Entry broadcasting
------------------

Subscribes to :class:`EntryAppendedEvent` on each session's bus. Replaced
the v0 polling approach the moment we added the core ABI event — no more
O(branch_length) walk per bus emit.

Auth (v2)
---------

This first cut is unauthenticated and binds ``127.0.0.1`` by default. v2
will add ``?token=`` to the connect URL, validated against
``AGENTM_LIVE_INSPECT_TOKEN``. Operators who flip the bind to ``0.0.0.0``
today do so at their own risk.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any

# WebSocket transport. ``websockets`` ships in the top-level ``[project]
# dependencies`` so this import is guaranteed to resolve in any AgentM
# install — no try/except fallback needed.
import websockets
from websockets.asyncio.server import ServerConnection, serve

from agentm.core.abi.events import (
    ChildSessionEndEvent,
    ChildSessionExtendingEvent,
    ChildSessionStartEvent,
    EntryAppendedEvent,
    EventBusObserver,
    SessionShutdownEvent,
)
from agentm.core.abi.extension import ExtensionAPI, Handler
from pydantic import BaseModel

from agentm.core.lib import to_jsonable
from agentm.extensions import ExtensionManifest

logger = logging.getLogger(__name__)


class LiveInspectorConfig(BaseModel):
    bind: str = "127.0.0.1"
    port: int = 0
    url_file: str | None = None
    role: str = "root"
    parent_root: str | None = None
    parent_port: int | None = None


MANIFEST = ExtensionManifest(
    name="live_inspector",
    description=(
        "Stream session events (bus dispatches + entry-tree appends + "
        "child-session lifecycle) over a WebSocket so an external UI "
        "can render the agent tree live."
    ),
    registers=(
        f"event:{ChildSessionStartEvent.CHANNEL}",
        f"event:{ChildSessionEndEvent.CHANNEL}",
        f"event:{ChildSessionExtendingEvent.CHANNEL}",
        f"event:{EntryAppendedEvent.CHANNEL}",
        f"event:{SessionShutdownEvent.CHANNEL}",
    ),
    config_schema=LiveInspectorConfig,
)


# Channels whose payloads we skip from the broadcast — pure per-token noise
# (``stream_delta``) that would otherwise dominate the wire.
_NOISY_CHANNELS: frozenset[str] = frozenset({"stream_delta"})


# Per-connection outbound queue cap. Exceeding this drops the NEWEST message
# so a slow client never starves the broadcaster.
_QUEUE_HIGHWATER: int = 256


# Process-level registry: one server per root_session_id. Child sessions in
# the same process share the parent's server by looking up their root id
# here in their own ``install()``.
_SERVERS: dict[str, _Server] = {}
_SERVERS_LOCK = threading.Lock()


def _now() -> float:
    return time.time()


# Sentinel pushed onto a client's outbound queue to signal "drain & close".
class _CloseSentinel:
    __slots__ = ()


_CLOSE: _CloseSentinel = _CloseSentinel()


@dataclass(eq=False)
class _Client:
    # ``eq=False`` so instances stay hashable by identity — they live in
    # ``_Server.clients`` (a set). The default dataclass ``__eq__`` would
    # make the class unhashable and ``set.add`` would raise.
    queue: asyncio.Queue[Any]
    dropped: int = 0


@dataclass
class _Server:
    """One WebSocket server bound to one root session.

    Owns its own asyncio event loop in a background thread so the AgentM
    runtime stays out of its way (atoms install synchronously, may or may
    not be inside an asyncio context).
    """

    root_session_id: str
    bind: str
    port: int  # 0 = OS-assigned
    refcount: int = 0  # number of sessions in this root tree that hold us
    loop: asyncio.AbstractEventLoop | None = None
    thread: threading.Thread | None = None
    server: Any = None  # websockets.asyncio.server.Server
    actual_port: int = 0
    clients: set[_Client] = field(default_factory=set)
    _backlog: list[dict[str, Any]] = field(default_factory=list)
    _backlog_cap: int = 2048
    _closed: bool = False

    def start(self) -> None:
        """Start the loop thread and block until the server is bound."""
        ready = threading.Event()
        bind_error: list[BaseException] = []

        async def _bootstrap() -> Any:
            # ``serve(...)`` constructs a ``Server`` object whose __init__
            # calls ``asyncio.get_running_loop()`` — so it must be invoked
            # inside a coroutine, not from ``run_until_complete(serve(...))``.
            srv = serve(
                self._handle_client,
                host=self.bind,
                port=self.port,
                # Only serve ``/inspect``; everything else gets a 404.
                process_request=_only_inspect_path,
            )
            # ``srv`` is an async context manager. Start it; entering the
            # ctx binds the listening socket. We hold the entered ctx for
            # the lifetime of the server and rely on ``shutdown()`` to
            # call ``.close()`` explicitly.
            await srv.__aenter__()
            return srv

        def runner() -> None:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            try:
                # ``self.server`` holds the ``websockets.asyncio.server.serve``
                # context-manager wrapper; the live ``asyncio.Server`` is at
                # ``self.server.server`` (Server.sockets is the listening
                # socket tuple).
                self.server = self.loop.run_until_complete(_bootstrap())
                asyncio_server = getattr(self.server, "server", None)
                sockets = (asyncio_server.sockets if asyncio_server else None) or ()
                if sockets:
                    sockname = sockets[0].getsockname()
                    self.actual_port = (
                        sockname[1] if isinstance(sockname, tuple) else self.port
                    )
                ready.set()
                self.loop.run_forever()
            except BaseException as exc:  # noqa: BLE001 — propagate to .start()
                bind_error.append(exc)
                ready.set()
            finally:
                try:
                    if self.loop is not None and not self.loop.is_closed():
                        self.loop.close()
                except Exception:
                    pass

        self.thread = threading.Thread(
            target=runner,
            name=f"live-inspector-{self.root_session_id[:8]}",
            daemon=True,
        )
        self.thread.start()
        ready.wait(timeout=5.0)
        if bind_error:
            raise bind_error[0]

    async def _handle_client(self, ws: ServerConnection) -> None:
        client = _Client(queue=asyncio.Queue(maxsize=_QUEUE_HIGHWATER))

        # Replay-ordering invariant: hello + backlog snapshot + backlog_done
        # must reach the client BEFORE any live broadcast that fires during
        # the handshake. We achieve this by enqueueing the prologue under a
        # single atomic block of ``put_nowait`` calls (no awaits) and only
        # THEN adding the client to ``self.clients``. Live broadcasts
        # (``_fanout``) only iterate clients present in that set, so they
        # cannot interleave with the prologue — the asyncio event loop is
        # single-threaded and we never yield between the prologue and the
        # set insertion. Any broadcast that fires concurrently with the
        # handshake will land in ``_backlog`` first (the broadcaster
        # appends to ``_backlog`` before scheduling ``_fanout``), so our
        # backlog snapshot picks it up — and then ``_fanout`` will skip
        # this client because it's not yet in ``self.clients``, so no
        # duplicate delivery.
        self._enqueue_sync(
            client,
            {
                "type": "hello",
                "root_session_id": self.root_session_id,
                "schema_version": 1,
            },
        )
        for record in list(self._backlog):
            self._enqueue_sync(client, record)
        self._enqueue_sync(client, {"type": "backlog_done"})
        self.clients.add(client)

        # Drain client subscribe frame(s) so the socket doesn't stall on
        # chatty clients. We don't act on subscribe content in v1.
        async def drain_reads() -> None:
            try:
                async for message in ws:
                    with contextlib.suppress(Exception):
                        json.loads(message)
            except websockets.ConnectionClosed:
                return

        reader_task = asyncio.create_task(drain_reads())

        try:
            while True:
                item = await client.queue.get()
                if isinstance(item, _CloseSentinel):
                    return
                try:
                    await ws.send(item)
                except websockets.ConnectionClosed:
                    return
        finally:
            reader_task.cancel()
            # Await the cancelled reader so a stray exception surfaces in
            # the log instead of slipping out as an unhandled-task warning
            # when the loop tears down.
            await asyncio.gather(reader_task, return_exceptions=True)
            self.clients.discard(client)
            with contextlib.suppress(Exception):
                await ws.close()

    def _enqueue_sync(self, client: _Client, record: dict[str, Any]) -> None:
        """Synchronous variant — used by the connect prologue.

        Identical drop-newest backpressure as :meth:`_enqueue_one`; the
        only difference is the lack of ``async`` so callers can run a
        batch (hello + backlog + backlog_done) atomically on the loop
        thread without yielding control.
        """
        try:
            line = json.dumps(record, default=str, ensure_ascii=False)
        except Exception:
            logger.exception("live_inspector: failed to serialize record")
            return
        try:
            client.queue.put_nowait(line)
        except asyncio.QueueFull:
            client.dropped += 1
            if client.dropped == 1 or client.dropped % 100 == 0:
                logger.warning(
                    "live_inspector: slow client dropped %d messages (root=%s)",
                    client.dropped,
                    self.root_session_id,
                )

    async def _enqueue_one(self, client: _Client, record: dict[str, Any]) -> None:
        self._enqueue_sync(client, record)

    def broadcast(self, record: dict[str, Any]) -> None:
        """Thread-safe broadcast to every connected client.

        Cheap on the hot path: appends to the backlog (under the loop
        thread's natural single-writer guarantee — we only call this from
        the agent thread / from sync handlers) and schedules a fanout
        coroutine on the server's loop.
        """
        if self._closed or self.loop is None:
            return
        # The truncate-then-append pair runs across multiple bytecode ops
        # and ``broadcast()`` may be called from non-loop threads (any
        # sync event handler on the agent thread). We rely on CPython's
        # GIL atomicity on list ops — at worst a concurrent broadcast
        # sees one stale length-cap check and inserts an extra record;
        # the backlog never drops below ``_backlog_cap * 3/4`` so the
        # invariant "late-joining client gets a recent window" is safe.
        # If this ever moves to free-threaded Python, swap in a
        # ``threading.Lock``.
        if len(self._backlog) >= self._backlog_cap:
            del self._backlog[: max(1, self._backlog_cap // 4)]
        self._backlog.append(record)

        async def _fanout() -> None:
            for client in list(self.clients):
                await self._enqueue_one(client, record)

        try:
            asyncio.run_coroutine_threadsafe(_fanout(), self.loop)
        except RuntimeError:
            # Loop closed mid-broadcast — drop.
            pass

    def acquire(self) -> None:
        self.refcount += 1

    def release(self) -> None:
        self.refcount -= 1
        if self.refcount <= 0:
            self.shutdown()

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True

        async def _close() -> None:
            for client in list(self.clients):
                with contextlib.suppress(Exception):
                    client.queue.put_nowait(_CLOSE)
            if self.server is not None:
                # Exit the async-CM the ``serve(...)`` wrapper returned;
                # that calls ``Server.close()`` + ``wait_closed()``.
                with contextlib.suppress(Exception):
                    await self.server.__aexit__(None, None, None)

        if self.loop is not None and not self.loop.is_closed():
            try:
                fut = asyncio.run_coroutine_threadsafe(_close(), self.loop)
                with contextlib.suppress(Exception):
                    fut.result(timeout=2.0)
                self.loop.call_soon_threadsafe(self.loop.stop)
            except RuntimeError:
                pass
        if self.thread is not None:
            self.thread.join(timeout=2.0)

        with _SERVERS_LOCK:
            existing = _SERVERS.get(self.root_session_id)
            if existing is self:
                _SERVERS.pop(self.root_session_id, None)


async def _only_inspect_path(connection: ServerConnection, request: Any) -> Any:
    """Reject any HTTP path that isn't ``/inspect``.

    Hook into ``websockets.serve(process_request=...)``: returning ``None``
    lets the handshake proceed; returning a Response short-circuits with
    that response.
    """
    path = getattr(request, "path", "/")
    # Strip query string before comparing.
    base = path.split("?", 1)[0]
    if base != "/inspect":
        return connection.respond(404, "not found\n")
    return None


def _get_or_create_server(
    root_session_id: str,
    bind: str,
    port: int,
    *,
    announce: bool,
    url_file: str | None,
) -> _Server:
    with _SERVERS_LOCK:
        existing = _SERVERS.get(root_session_id)
        if existing is not None:
            existing.acquire()
            return existing
        srv = _Server(root_session_id=root_session_id, bind=bind, port=port)
        srv.start()
        srv.acquire()
        _SERVERS[root_session_id] = srv

    if announce:
        url = f"ws://{bind}:{srv.actual_port}/inspect?root={root_session_id}"
        try:
            sys.stderr.write(f"LIVE INSPECT: {url}\n")
            sys.stderr.flush()
        except Exception:
            pass
        if url_file:
            try:
                with open(url_file, "w", encoding="utf-8") as fh:
                    fh.write(url + "\n")
            except OSError:
                logger.warning(
                    "live_inspector: failed to write url_file=%s", url_file
                )
    return srv


class _BusObserver(EventBusObserver):
    """Captures every ``EventBus.emit`` on this session's bus."""

    def __init__(self, session_id: str, server: _Server) -> None:
        self._session_id = session_id
        self._server = server

    def on_emit_start(self, channel: str, event: Any) -> None:
        return None

    def on_handler_done(
        self,
        channel: str,
        handler: Handler,
        event: Any,
        result: Any,
        error: BaseException | None,
        duration_ns: int,
    ) -> None:
        return None

    def on_emit_end(
        self, channel: str, event: Any, results: list[Any]
    ) -> None:
        if channel in _NOISY_CHANNELS:
            return
        # ``EntryAppendedEvent`` is handled on its own typed channel where
        # the inspector emits a richer ``entry`` record; skip the generic
        # event broadcast so we don't double-publish entry writes.
        if channel == EntryAppendedEvent.CHANNEL:
            return
        try:
            payload = to_jsonable(event)
        except Exception:
            payload = {"_repr": repr(event)}
        self._server.broadcast(
            {
                "type": "event",
                "session_id": self._session_id,
                "ts": _now(),
                "kind": type(event).__name__,
                "channel": channel,
                "payload": payload,
            }
        )


def install(api: ExtensionAPI, config: LiveInspectorConfig) -> None:
    bind = config.bind
    port = config.port
    url_file = config.url_file
    role = config.role
    parent_port_raw = config.parent_port

    # Env-var overrides for ROOT only. Children inherit the resolved
    # bind:port from the spawn-wrap config so they cannot drift.
    if role == "root":
        env_bind = os.environ.get("AGENTM_LIVE_INSPECT_BIND")
        env_port = os.environ.get("AGENTM_LIVE_INSPECT_PORT")
        env_url_file = os.environ.get("AGENTM_LIVE_INSPECT_URL_FILE")
        if env_bind:
            bind = env_bind
        if env_port:
            try:
                port = int(env_port)
            except ValueError:
                pass
        if env_url_file:
            url_file = env_url_file
    else:
        # For child role: use the parent's already-resolved port verbatim.
        # This is the unambiguous join key — we never re-bind in a child.
        if parent_port_raw is not None:
            port = parent_port_raw

    # For a child, the per-root server already exists in this process; the
    # registry-keyed-by-root lookup finds it. For a root, we bind a fresh
    # server.
    server = _get_or_create_server(
        root_session_id=api.root_session_id,
        bind=bind,
        port=port,
        announce=(role == "root"),
        url_file=url_file,
    )

    session_id = api.session_id
    server.broadcast(
        {
            "type": "session_started",
            "session_id": session_id,
            "parent_session_id": api.parent_session_id,
            "purpose": api.purpose,
            "cwd": api.cwd,
            "ts": _now(),
        }
    )

    observer = _BusObserver(session_id=session_id, server=server)
    api.add_observer(observer)

    def _on_entry_appended(event: EntryAppendedEvent) -> None:
        try:
            payload = to_jsonable(event.payload)
        except Exception:
            payload = {"_repr": repr(event.payload)}
        # Forward the event's session_id (the SessionManager header id
        # under which the entry was written) verbatim, rather than the
        # install-time-captured ``session_id`` (api.session_id, the OTel
        # span id). They can differ — get_session_id() returns the
        # persisted header id; the inspector should report the id the
        # entry was actually written under.
        server.broadcast(
            {
                "type": "entry",
                "session_id": event.session_id,
                "ts": _now(),
                "entry_id": event.entry_id,
                "entry_type": event.entry_type,
                "parent_id": event.parent_id,
                "payload": payload,
            }
        )

    def _on_child_start(event: ChildSessionStartEvent) -> None:
        server.broadcast(
            {
                "type": "session_started",
                "session_id": event.child_session_id,
                "parent_session_id": event.parent_session_id,
                "purpose": event.purpose,
                # cwd is unknown from the parent's perspective; the child's
                # own session_started (broadcast by its inspector install)
                # will carry the cwd.
                "cwd": None,
                "ts": _now(),
            }
        )

    def _on_child_end(event: ChildSessionEndEvent) -> None:
        server.broadcast(
            {
                "type": "session_ended",
                "session_id": event.child_session_id,
                "ts": _now(),
            }
        )

    def _on_shutdown(_: SessionShutdownEvent) -> None:
        server.broadcast(
            {
                "type": "session_ended",
                "session_id": session_id,
                "ts": _now(),
            }
        )
        server.release()

    api.on(EntryAppendedEvent.CHANNEL, _on_entry_appended)
    api.on(ChildSessionStartEvent.CHANNEL, _on_child_start)
    api.on(ChildSessionEndEvent.CHANNEL, _on_child_end)
    api.on(SessionShutdownEvent.CHANNEL, _on_shutdown)

    # Auto-attach the inspector to every spawned child session so the
    # whole agent tree publishes to the same WebSocket. The substrate
    # fires ``ChildSessionExtendingEvent`` on the parent bus before
    # spawning; our handler contributes one ``(module, config)`` tuple,
    # the substrate dedupes by module so operator overrides on the
    # child config win, and the child's own ``install()`` finds the
    # per-root server via ``_SERVERS`` lookup instead of binding a new
    # one. Only the ROOT install registers the contributor — children
    # don't need to (transitive grandchildren walk back to the same
    # root via the trace_id). Replaces the previous monkey-wrap of
    # ``api.spawn_child_session``: this is a typed bus channel, no
    # api-method overwrite, no fragility.
    if role == "root":
        root_port = server.actual_port
        root_bind = bind
        root_id = api.root_session_id

        def _on_child_extending(
            _event: ChildSessionExtendingEvent,
        ) -> list[tuple[str, dict[str, Any]]]:
            return [
                (
                    "contrib.extensions.live_inspector",
                    {
                        "bind": root_bind,
                        "port": root_port,
                        "role": "child",
                        "parent_root": root_id,
                        "parent_port": root_port,
                    },
                )
            ]

        api.on(ChildSessionExtendingEvent.CHANNEL, _on_child_extending)
