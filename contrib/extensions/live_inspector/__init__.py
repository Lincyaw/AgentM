"""Live-inspector atom — broadcast a session's events over a TCP socket so
an external UI can render the agent tree in real time.

This is the substrate for live audit visualization. UI / aegis-ui work is
out of scope; this atom defines the wire protocol and gets the events flowing.

Why raw asyncio TCP (no ``websockets`` / ``aiohttp``)
---------------------------------------------------

The AgentM tree currently does not depend on either library and the §11
single-file atom contract restricts atoms to standard-library + ABI imports.
Rather than pull in a new runtime dependency just for the first cut, we ship
a newline-delimited-JSON over raw TCP server. The protocol is trivial enough
that a thin browser-side shim can adapt it to WebSocket later; for current
internal tooling a plain ``nc`` / Python client suffices.

Wire protocol (v1)
------------------

Transport: TCP, newline-delimited UTF-8 JSON. One JSON object per line in
each direction. The server URL is logged to stderr at startup as
``LIVE INSPECT: tcp://<host>:<port>/inspect?root=<root_session_id>``.

Client → server (first line, optional):
    ``{"type": "subscribe", "root_session_id": "<id>"}``
    If omitted, the client subscribes to whichever root this server is
    bound to (one server == one root in v1, so this is informational).

Server → client (every connection, in order):
    1. Hello:
       ``{"type": "hello", "root_session_id": "<id>", "schema_version": 1}``
    2. For every session in the tree (root + children) as they start:
       ``{"type": "session_started", "session_id", "parent_session_id",
           "purpose", "cwd", "ts"}``
    3. For every EventBus dispatch in any session:
       ``{"type": "event", "session_id", "ts", "kind": "<event class name>",
           "channel": "<bus channel>", "payload": {...}}``
    4. For every entry appended to a session's entry tree (messages,
       ``llmharness.audit_event``, ``llmharness.verdict``, …):
       ``{"type": "entry", "session_id", "ts", "entry_type": "<type>",
           "entry_id", "payload": {...}}``
    5. When a session ends:
       ``{"type": "session_ended", "session_id", "ts"}``

Backpressure
------------

Each connection has a bounded outbound queue (``_QUEUE_HIGHWATER``). When the
queue is full the NEWEST message is dropped and a warning logged — slow
clients never deadlock the broadcasting hot path.

Open design question: ``entry`` broadcasting
--------------------------------------------

There is no ABI event today for ``ReadonlySession.append_entry``. For this
first cut we POLL ``api.session.get_branch()`` after every bus dispatch and
broadcast newly-appended entries. The boundary stays clean (ABI-only — no
``core.runtime.*`` import, no file tail) and it captures every entry source
(message bodies, llmharness audit entries, plan submissions). The cost is
one branch-length comparison per bus emit, which is O(1) amortised.

The cleaner long-term fix is option (c) in the design call: an
``EntryAppendedEvent`` fired by ``SessionView.append_entry`` so atoms can
subscribe directly. That requires a small core/ABI change and is logged as
a follow-up. Option (b) — tail the on-disk JSONL — was rejected: the file
is rewritten (``_rewrite_file``) on session creation and forks, which
breaks naive tailers.

Child sessions (extractor / auditor / tool_eval_run)
----------------------------------------------------

``ChildSessionStartEvent`` fires on the parent bus with the child's
``session_id`` and ``purpose`` — we broadcast that as a ``session_started``
event so the UI can render the child node. However the inspector atom does
NOT receive the child's internal events unless it is also installed in the
child session's extension set. ``llmharness`` currently builds its child
``AgentSessionConfig`` with a fixed extension list and does not propagate
the parent's scenario. Full child-tree visibility requires either:

  * a follow-up to ``llmharness.adapters.agentm`` to append
    ``contrib.extensions.live_inspector`` to ``extractor_extensions`` /
    ``auditor_extensions`` when the parent has the inspector loaded, or
  * a core mechanism for "session-tree-wide observers" that the substrate
    auto-installs in spawned children.

This first cut documents the gap; child lifecycle is visible, child internal
events are not.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import socket
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from agentm.core.abi.events import (
    ChildSessionEndEvent,
    ChildSessionStartEvent,
    EventBusObserver,
    SessionShutdownEvent,
)
from agentm.core.abi.extension import ExtensionAPI, Handler
from agentm.core.lib import to_jsonable
from agentm.extensions import ExtensionManifest

logger = logging.getLogger(__name__)


MANIFEST = ExtensionManifest(
    name="live_inspector",
    description=(
        "Stream session events (bus dispatches + entry-tree appends + "
        "child-session lifecycle) over a TCP newline-JSON socket so an "
        "external UI can render the agent tree live."
    ),
    registers=(
        f"event:{ChildSessionStartEvent.CHANNEL}",
        f"event:{ChildSessionEndEvent.CHANNEL}",
        f"event:{SessionShutdownEvent.CHANNEL}",
    ),
    config_schema={
        "type": "object",
        "properties": {
            "bind": {"type": "string"},
            "port": {"type": "integer"},
            "url_file": {"type": ["string", "null"]},
        },
        "additionalProperties": False,
    },
)


# Channels whose payloads we skip from the broadcast — these are pure
# per-token noise (``stream_delta``) or handler-internal signals that add no
# value to a UI viewer and would otherwise dominate the wire.
_NOISY_CHANNELS: frozenset[str] = frozenset({"stream_delta"})


# Per-connection outbound queue cap. Exceeding this drops the NEWEST message
# (so a slow client never starves the broadcaster).
_QUEUE_HIGHWATER: int = 256


# Process-level registry: one server per root_session_id. Child sessions in
# the same process share the parent's server (when the atom is loaded in
# them, which it isn't by default — see module docstring).
_SERVERS: dict[str, _Server] = {}
_SERVERS_LOCK = threading.Lock()


def _now() -> float:
    return time.time()


# Sentinel pushed onto a client's queue to signal "drain and close".
# Identity-checked (``msg is _CLOSE_SENTINEL``); a real broadcast payload
# is always at least ``b"{...}\n"`` so there's no collision risk.
_CLOSE_SENTINEL: bytes = b"\x00__close__\x00"


@dataclass(eq=False)
class _Client:
    # ``eq=False`` so instances stay hashable by identity — they live in
    # ``_Server.clients`` (a set). The default dataclass ``__eq__`` would
    # make the class unhashable and ``set.add`` would raise.
    writer: asyncio.StreamWriter
    queue: asyncio.Queue[bytes]
    dropped: int = 0


@dataclass
class _Server:
    """One TCP server bound to one root session.

    Owns its own asyncio event loop in a background thread so the AgentM
    runtime (which may or may not be inside an asyncio context at install
    time, depending on caller) stays out of its way.
    """

    root_session_id: str
    bind: str
    port: int  # 0 = OS-assigned
    refcount: int = 0  # number of sessions in this root tree that hold us
    loop: asyncio.AbstractEventLoop | None = None
    thread: threading.Thread | None = None
    server: asyncio.base_events.Server | None = None
    actual_port: int = 0
    clients: set[_Client] = field(default_factory=set)
    _backlog: list[dict[str, Any]] = field(default_factory=list)
    _backlog_cap: int = 2048
    _closed: bool = False

    def start(self) -> None:
        """Start the loop thread and block until the server is bound."""
        ready = threading.Event()
        bind_error: list[BaseException] = []

        def runner() -> None:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            try:
                self.server = self.loop.run_until_complete(
                    asyncio.start_server(
                        self._handle_client, host=self.bind, port=self.port
                    )
                )
                # Resolve actual port (in case port=0).
                sockets = self.server.sockets or ()
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
                    if self.server is not None:
                        self.server.close()
                except Exception:
                    pass
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

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        client = _Client(writer=writer, queue=asyncio.Queue(maxsize=_QUEUE_HIGHWATER))
        self.clients.add(client)
        # Send hello + backlog.
        await self._enqueue_one(
            client,
            {
                "type": "hello",
                "root_session_id": self.root_session_id,
                "schema_version": 1,
            },
        )
        for record in list(self._backlog):
            await self._enqueue_one(client, record)
        await self._enqueue_one(client, {"type": "backlog_done"})

        # Drain client subscribe line (optional, ignored content) so the
        # socket buffer doesn't fill on chatty clients.
        async def drain_reads() -> None:
            try:
                while True:
                    line = await reader.readline()
                    if not line:
                        return
                    # Best-effort decode; ignore malformed.
                    try:
                        json.loads(line.decode("utf-8"))
                    except Exception:
                        pass
            except (ConnectionResetError, asyncio.IncompleteReadError):
                return

        reader_task = asyncio.create_task(drain_reads())

        try:
            while True:
                payload = await client.queue.get()
                if payload is _CLOSE_SENTINEL:
                    return
                try:
                    writer.write(payload)
                    await writer.drain()
                except (ConnectionResetError, BrokenPipeError):
                    return
        finally:
            reader_task.cancel()
            self.clients.discard(client)
            with contextlib.suppress(Exception):
                writer.close()
                await writer.wait_closed()

    async def _enqueue_one(self, client: _Client, record: dict[str, Any]) -> None:
        try:
            line = (json.dumps(record, default=str, ensure_ascii=False) + "\n").encode(
                "utf-8"
            )
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

    def broadcast(self, record: dict[str, Any]) -> None:
        """Thread-safe broadcast to every connected client.

        Cheap on the hot path: schedules one coroutine on the server's loop
        and returns. The coroutine fan-outs to per-client queues; per-client
        writes happen on the loop thread.
        """
        if self._closed or self.loop is None:
            return
        # Append to backlog (capped) so late connecters get state.
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
            # Signal every client to drain & close.
            for client in list(self.clients):
                with contextlib.suppress(Exception):
                    client.queue.put_nowait(_CLOSE_SENTINEL)
            if self.server is not None:
                self.server.close()
                with contextlib.suppress(Exception):
                    await self.server.wait_closed()

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


def _get_or_create_server(
    root_session_id: str,
    bind: str,
    port: int,
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

    # Side effects outside the lock.
    url = f"tcp://{bind}:{srv.actual_port}/inspect?root={root_session_id}"
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
            logger.warning("live_inspector: failed to write url_file=%s", url_file)
    return srv


class _BusObserver(EventBusObserver):
    """Captures every ``EventBus.emit`` on this session's bus."""

    def __init__(self, session_id: str, server: _Server) -> None:
        self._session_id = session_id
        self._server = server
        # Entry-tree polling state. The branch is a list of SessionEntry-like
        # objects with stable ``.id`` (uuid4 hex); we track the set we've
        # already broadcast so a new append yields exactly one ``entry`` msg.
        self._seen_entry_ids: set[str] = set()
        self._branch_getter: Any = None  # set by install()

    def attach_branch_getter(self, getter: Any) -> None:
        self._branch_getter = getter

    def on_emit_start(self, channel: str, event: Any) -> None:  # noqa: D401
        return None

    def on_handler_done(
        self,
        channel: str,
        handler: Handler,
        result: Any,
        error: BaseException | None,
        duration_ns: int,
    ) -> None:
        return None

    def on_emit_end(
        self, channel: str, event: Any, results: list[Any]
    ) -> None:
        if channel not in _NOISY_CHANNELS:
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
        self._poll_entries()

    def _poll_entries(self) -> None:
        if self._branch_getter is None:
            return
        try:
            branch = self._branch_getter()
        except Exception:
            return
        for entry in branch:
            entry_id = getattr(entry, "id", None)
            if entry_id is None or entry_id in self._seen_entry_ids:
                continue
            self._seen_entry_ids.add(entry_id)
            try:
                payload = to_jsonable(getattr(entry, "payload", None))
            except Exception:
                payload = {"_repr": repr(getattr(entry, "payload", None))}
            self._server.broadcast(
                {
                    "type": "entry",
                    "session_id": self._session_id,
                    "ts": _now(),
                    "entry_id": entry_id,
                    "entry_type": getattr(entry, "type", "unknown"),
                    "payload": payload,
                }
            )


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    bind = str(config.get("bind", "127.0.0.1"))
    port = int(config.get("port", 0))
    url_file_raw = config.get("url_file")
    url_file = str(url_file_raw) if url_file_raw else None

    # Env-var overrides so an operator can flip the bind/port without
    # rewriting the manifest. Useful for the aegis-ui dev loop where the
    # UI binds a known port and the agent should publish there.
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

    server = _get_or_create_server(
        root_session_id=api.root_session_id,
        bind=bind,
        port=port,
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
    # Attach the branch getter via the session view — pure ABI access, no
    # ``core.runtime.*`` import.
    observer.attach_branch_getter(api.session.get_branch)
    api.add_observer(observer)

    def _on_child_start(event: ChildSessionStartEvent) -> None:
        server.broadcast(
            {
                "type": "session_started",
                "session_id": event.child_session_id,
                "parent_session_id": event.parent_session_id,
                "purpose": event.purpose,
                # ``cwd`` is unknown from the parent's perspective; the
                # child will broadcast its own session_started with cwd if
                # the inspector atom is loaded inside it.
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
        # Release our refcount. Server shuts down when the last holder
        # releases — typically the root session at process exit.
        server.release()

    api.on(ChildSessionStartEvent.CHANNEL, _on_child_start)
    api.on(ChildSessionEndEvent.CHANNEL, _on_child_end)
    api.on(SessionShutdownEvent.CHANNEL, _on_shutdown)


# --- Test helpers -----------------------------------------------------------
# Exposed for unit tests in ``tests/test_live_inspector.py``. Not part of
# the atom's runtime contract; do not import from other atoms.

def _connect_and_read(host: str, port: int, n_messages: int, timeout: float = 2.0) -> list[dict[str, Any]]:
    """Open a blocking TCP socket, read up to ``n_messages`` JSON lines.

    Test-only helper; uses stdlib ``socket`` rather than asyncio to keep the
    test harness simple.
    """
    sock = socket.create_connection((host, port), timeout=timeout)
    sock.settimeout(timeout)
    out: list[dict[str, Any]] = []
    buf = b""
    try:
        while len(out) < n_messages:
            chunk = sock.recv(4096)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if not line.strip():
                    continue
                out.append(json.loads(line.decode("utf-8")))
                if len(out) >= n_messages:
                    break
    finally:
        with contextlib.suppress(Exception):
            sock.close()
    return out
