"""Fail-stop tests for the ``live_inspector`` atom.

Scope per CLAUDE.md "Testing philosophy" — only positions where the atom's
value proposition fails if broken:

1. **Wire protocol round-trip over WebSocket** — server emits ``hello``,
   replays backlog, and forwards broadcasts in order with the right schema.
   If this breaks, every UI client sees garbage.

2. **Backpressure** — a slow client does not deadlock the broadcaster. If
   this breaks, one stuck websocket-on-a-laptop wedges the whole agent.

3. **Child-session attach via spawn-wrap** — when the inspector wraps
   ``api.spawn_child_session``, every spawned child's ``extensions`` list
   acquires the inspector entry with ``role: child`` so the child joins
   the root's server instead of starting another. This is the core
   correctness invariant for visualising the agent tree.

4. **Singleton-per-root** — two ``_get_or_create_server`` calls with the
   same root_session_id reuse the same server. Load-bearing for child
   sessions sharing the parent's socket.

Per-event-kind happy paths are NOT covered — payload serialisation is
delegated to :func:`agentm.core.lib.to_jsonable`, already tested upstream.
"""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any

import pytest
from websockets.sync.client import connect

from contrib.extensions.live_inspector import (
    _SERVERS,
    _get_or_create_server,
    _wrap_spawn_child_session,
)


@pytest.fixture(autouse=True)
def _clear_registry() -> None:
    # Each test should start from a clean slate so refcounts don't bleed.
    for srv in list(_SERVERS.values()):
        srv.shutdown()
    _SERVERS.clear()
    yield
    for srv in list(_SERVERS.values()):
        srv.shutdown()
    _SERVERS.clear()


def _read_n_messages(url: str, n: int, timeout: float = 5.0) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with connect(url, open_timeout=timeout, close_timeout=timeout) as ws:
        for _ in range(n):
            try:
                msg = ws.recv(timeout=timeout)
            except TimeoutError:
                break
            if isinstance(msg, (bytes, bytearray)):
                msg = msg.decode("utf-8")
            out.append(json.loads(msg))
    return out


def test_wire_protocol_round_trip() -> None:
    """Connect a client; assert hello + backlog replay + ordered delivery."""

    server = _get_or_create_server(
        root_session_id="root-abc",
        bind="127.0.0.1",
        port=0,
        announce=False,
        url_file=None,
    )
    try:
        server.broadcast(
            {
                "type": "session_started",
                "session_id": "s1",
                "parent_session_id": None,
                "purpose": "root",
                "cwd": "/tmp",
                "ts": 1.0,
            }
        )
        server.broadcast(
            {
                "type": "event",
                "session_id": "s1",
                "ts": 1.1,
                "kind": "TurnEndEvent",
                "channel": "turn_end",
                "payload": {"turn_index": 0},
            }
        )

        url = f"ws://127.0.0.1:{server.actual_port}/inspect?root=root-abc"
        result: dict[str, list[dict[str, Any]]] = {"messages": []}

        def reader() -> None:
            result["messages"] = _read_n_messages(url, 4)

        t = threading.Thread(target=reader)
        t.start()
        t.join(timeout=5.0)
        assert not t.is_alive(), "reader thread hung"

        msgs = result["messages"]
        assert len(msgs) == 4, msgs
        assert msgs[0]["type"] == "hello"
        assert msgs[0]["root_session_id"] == "root-abc"
        assert msgs[0]["schema_version"] == 1
        assert msgs[1]["type"] == "session_started"
        assert msgs[1]["session_id"] == "s1"
        assert msgs[2]["type"] == "event"
        assert msgs[2]["channel"] == "turn_end"
        assert msgs[2]["kind"] == "TurnEndEvent"
        assert msgs[3]["type"] == "backlog_done"
    finally:
        server.release()


def test_backpressure_does_not_deadlock_server() -> None:
    """Flooding broadcasts while a slow client is connected must not stall.

    A fresh fast client connecting AFTER the flood must still see hello +
    at least one backlog item + backlog_done within the timeout.
    """

    server = _get_or_create_server(
        root_session_id="root-slow",
        bind="127.0.0.1",
        port=0,
        announce=False,
        url_file=None,
    )
    try:
        url = f"ws://127.0.0.1:{server.actual_port}/inspect?root=root-slow"

        slow_ws = connect(url, open_timeout=5.0)
        try:
            for i in range(2000):
                server.broadcast({"type": "event", "i": i, "session_id": "s1"})

            messages = _read_n_messages(url, 3)
            assert messages, "fast client received nothing"
            assert messages[0]["type"] == "hello"
        finally:
            slow_ws.close()
    finally:
        server.release()


def test_server_singleton_per_root() -> None:
    """Two ``_get_or_create_server`` calls with the same root reuse one server.

    Load-bearing for child-session sharing.
    """

    a = _get_or_create_server(
        "root-shared", "127.0.0.1", 0, announce=False, url_file=None
    )
    try:
        b = _get_or_create_server(
            "root-shared", "127.0.0.1", 0, announce=False, url_file=None
        )
        try:
            assert a is b
            assert a.refcount == 2
        finally:
            b.release()
        assert a.refcount == 1
    finally:
        a.release()


# --- Child-session spawn-wrap -----------------------------------------------


class _FakeChildConfig:
    """Minimal stand-in for ``AgentSessionConfig`` — only what the wrap touches."""

    def __init__(self) -> None:
        self.extensions: list[tuple[str, dict[str, Any]]] = []


class _FakeAPI:
    """Captures what the wrap did to ``spawn_child_session`` calls."""

    def __init__(self) -> None:
        self.spawned_configs: list[Any] = []

        async def _original_spawn(config: Any = None, **kwargs: Any) -> Any:
            self.spawned_configs.append(
                config if config is not None else dict(kwargs)
            )
            return object()

        self.spawn_child_session = _original_spawn


def test_spawn_wrap_appends_inspector_to_child_extensions() -> None:
    """The wrap must append the inspector entry with ``role: child`` and the
    right ``parent_root`` / ``parent_port``. Without this, child sessions are
    invisible to the UI.
    """

    api = _FakeAPI()
    _wrap_spawn_child_session(
        api=api,  # type: ignore[arg-type]
        bind="127.0.0.1",
        actual_port=4242,
        root_session_id="root-xyz",
    )

    cfg = _FakeChildConfig()
    asyncio.run(api.spawn_child_session(cfg))

    assert len(cfg.extensions) == 1, cfg.extensions
    module, child_config = cfg.extensions[0]
    assert module == "contrib.extensions.live_inspector"
    assert child_config["role"] == "child"
    assert child_config["parent_root"] == "root-xyz"
    assert child_config["parent_port"] == 4242
    assert child_config["bind"] == "127.0.0.1"


def test_spawn_wrap_is_idempotent_when_operator_already_listed_inspector() -> None:
    """If the child config already lists ``live_inspector``, the wrap leaves
    the list alone — operator override wins.
    """

    api = _FakeAPI()
    _wrap_spawn_child_session(
        api=api,  # type: ignore[arg-type]
        bind="127.0.0.1",
        actual_port=4242,
        root_session_id="root-xyz",
    )

    cfg = _FakeChildConfig()
    cfg.extensions.append(
        (
            "contrib.extensions.live_inspector",
            {"bind": "127.0.0.1", "port": 9999, "role": "child"},
        )
    )

    asyncio.run(api.spawn_child_session(cfg))

    assert len(cfg.extensions) == 1
    assert cfg.extensions[0][1]["port"] == 9999


def test_spawn_wrap_handles_kwargs_call_style() -> None:
    """``api.spawn_child_session(**kwargs)`` is a legal call style; the wrap
    must mutate the kwargs dict's ``extensions`` list too.
    """

    api = _FakeAPI()
    _wrap_spawn_child_session(
        api=api,  # type: ignore[arg-type]
        bind="127.0.0.1",
        actual_port=4242,
        root_session_id="root-kw",
    )

    asyncio.run(api.spawn_child_session(extensions=[], cwd="/tmp"))

    captured = api.spawned_configs[0]
    assert isinstance(captured, dict)
    exts = captured["extensions"]
    assert len(exts) == 1
    assert exts[0][0] == "contrib.extensions.live_inspector"
    assert exts[0][1]["parent_root"] == "root-kw"
