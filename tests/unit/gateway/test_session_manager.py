"""Fail-stop: SessionManager get_or_create + ChatSessionMap resume (§3.3).

The session dict + ChatSessionMap are the gateway's identity layer.
get_or_create must (a) return the same instance for a repeated
session_key, (b) record new sessions in the map, and (c) resume by
prior session_id after a daemon restart — otherwise a chat's transcript
is lost on the second message.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agentm.gateway.chat_session_map import ChatSessionMap
from agentm.gateway.session_manager import SessionManager
from agentm.gateway.wire import InboundBody


class _FakeSessionManager:
    def __init__(self, sid: str) -> None:
        self._sid = sid

    def get_session_id(self) -> str:
        return self._sid


class _FakeSession:
    """Minimal AgentSession stand-in: records installed atoms + services."""

    def __init__(self, sid: str, resume: str | None) -> None:
        self.session_manager = _FakeSessionManager(sid)
        self.resumed_from = resume
        self.services: dict[str, Any] = {}
        self.installed: list[str] = []
        self.shut = False

    def set_service(self, name: str, obj: Any) -> None:
        self.services[name] = obj

    def install_atom(self, name: str) -> None:
        self.installed.append(name)

    async def shutdown(self) -> None:
        self.shut = True


def _make_manager(tmp_path: Path, factory_log: list[tuple[str, str | None]]):
    chat_map = ChatSessionMap(tmp_path / "map.json")
    counter = {"n": 0}

    async def factory(
        cwd: str,
        session_key: str,
        scenario: str | None,
        resume: str | None,
        wire_services: dict[str, Any],
    ):
        counter["n"] += 1
        factory_log.append((session_key, resume))
        # When resuming, keep the same id; else mint a fresh one. The gateway
        # now seeds the wire_driver services BEFORE atom install and mounts
        # wire_driver inside the factory; mirror both on the fake so the
        # stamp/install assertions still hold.
        sid = resume or f"sid-{counter['n']}"
        sess = _FakeSession(sid, resume)
        for name, obj in wire_services.items():
            sess.set_service(name, obj)
        sess.install_atom("wire_driver")
        return sess

    async def sink(body: dict) -> None:
        return None

    return (
        SessionManager(
            cwd=str(tmp_path),
            chat_map=chat_map,
            session_factory=factory,
            outbound_sink=sink,
        ),
        chat_map,
    )


def _inbound() -> InboundBody:
    return InboundBody(channel="terminal", chat_id="t1", sender_id="u1", content="hi")


@pytest.mark.asyncio
async def test_first_call_creates_maps_and_stamps_wire_driver(tmp_path: Path) -> None:
    log: list[tuple[str, str | None]] = []
    mgr, chat_map = _make_manager(tmp_path, log)
    sess = await mgr.get_or_create("terminal:t1", "local", _inbound())
    assert chat_map.get("terminal:t1") == sess.session_manager.get_session_id()
    assert "wire_driver" in sess.installed
    assert sess.services["session_key"] == "terminal:t1"
    assert "wire_outbound" in sess.services


@pytest.mark.asyncio
async def test_second_call_returns_same_instance(tmp_path: Path) -> None:
    log: list[tuple[str, str | None]] = []
    mgr, _ = _make_manager(tmp_path, log)
    a = await mgr.get_or_create("terminal:t1", "local", _inbound())
    b = await mgr.get_or_create("terminal:t1", "local", _inbound())
    assert a is b
    assert len(log) == 1  # factory called once


@pytest.mark.asyncio
async def test_restart_resumes_by_prior_session_id(tmp_path: Path) -> None:
    # First daemon lifetime: create + map.
    log1: list[tuple[str, str | None]] = []
    mgr1, _ = _make_manager(tmp_path, log1)
    s1 = await mgr1.get_or_create("terminal:t1", "local", _inbound())
    prior_id = s1.session_manager.get_session_id()

    # Second daemon lifetime: same map file, fresh manager.
    log2: list[tuple[str, str | None]] = []
    mgr2, _ = _make_manager(tmp_path, log2)
    s2 = await mgr2.get_or_create("terminal:t1", "local", _inbound())
    assert log2 == [("terminal:t1", prior_id)]  # factory got the resume id
    assert s2.resumed_from == prior_id


@pytest.mark.asyncio
async def test_shutdown_keeps_map_forget_clears_it(tmp_path: Path) -> None:
    log: list[tuple[str, str | None]] = []
    mgr, chat_map = _make_manager(tmp_path, log)
    await mgr.get_or_create("terminal:t1", "local", _inbound())
    await mgr.shutdown_session("terminal:t1")
    # /new semantics: map entry survives.
    assert chat_map.get("terminal:t1") is not None
    # /end semantics: forget clears it.
    mgr.forget("terminal:t1")
    assert chat_map.get("terminal:t1") is None


@pytest.mark.asyncio
async def test_session_id_prefers_live_session(tmp_path: Path) -> None:
    log: list[tuple[str, str | None]] = []
    mgr, _ = _make_manager(tmp_path, log)
    sess = await mgr.get_or_create("terminal:t1", "local", _inbound())
    assert mgr.session_id("terminal:t1") == sess.session_manager.get_session_id()


def test_session_id_falls_back_to_chat_map(tmp_path: Path) -> None:
    # After a restart there is no live session, but the persisted chat map
    # still points at the prior session id. /context (via session_id) must
    # resolve that id rather than report "no active session".
    chat_map = ChatSessionMap(tmp_path / "map.json")
    chat_map.set("terminal:t1", "sid-persisted")

    async def _sink(body: dict) -> None:
        return None

    async def _factory(*a: Any) -> Any:
        raise AssertionError("session_id must not create a session")

    mgr = SessionManager(
        cwd=str(tmp_path),
        chat_map=chat_map,
        session_factory=_factory,
        outbound_sink=_sink,
    )
    assert mgr.session_id("terminal:t1") == "sid-persisted"
    # Neither live nor mapped -> None.
    assert mgr.session_id("terminal:unknown") is None
