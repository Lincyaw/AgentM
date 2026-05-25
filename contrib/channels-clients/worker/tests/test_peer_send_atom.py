"""Unit tests for the ``tool_peer_send`` atom and runner reply matching.

No real wire client, no real AgentSession — we hand the atom a fake
``PeerMessaging`` and a minimal ``ExtensionAPI`` stub that only
implements ``get_service`` / ``register_tool`` (the only ExtensionAPI
methods the atom calls).
"""

from __future__ import annotations

import asyncio
from typing import Any


from agentm.core.abi import FunctionTool, ToolResult

from agentm_worker.peer_send_atom import (
    PEER_MESSAGING_SERVICE,
    install,
)


class _FakePeerMessaging:
    """In-memory PeerMessaging stub: capture sends, resolve replies."""

    def __init__(self) -> None:
        self.sends: list[dict[str, Any]] = []
        self._futures: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._next_id = 0

    def new_correlation_id(self) -> str:
        self._next_id += 1
        return f"cid-{self._next_id}"

    async def send_peer(
        self, *, to: str, content: str, correlation_id: str
    ) -> None:
        self.sends.append(
            {"to": to, "content": content, "correlation_id": correlation_id}
        )

    async def await_peer_reply(
        self, correlation_id: str, timeout_seconds: float
    ) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        fut = self._futures.setdefault(correlation_id, loop.create_future())
        try:
            return await asyncio.wait_for(fut, timeout=timeout_seconds)
        except asyncio.TimeoutError as exc:
            self._futures.pop(correlation_id, None)
            raise TimeoutError(str(exc)) from exc

    def deliver(self, correlation_id: str, body: dict[str, Any]) -> None:
        loop = asyncio.get_event_loop()
        fut = self._futures.setdefault(correlation_id, loop.create_future())
        if not fut.done():
            fut.set_result(body)


class _FakeAPI:
    """Minimum ExtensionAPI surface ``install`` actually touches."""

    def __init__(self, services: dict[str, Any]) -> None:
        self._services = services
        self.registered: list[FunctionTool] = []
        self.cwd = "/tmp/_fake"

    def get_service(self, name: str) -> Any | None:
        return self._services.get(name)

    def register_tool(self, tool: FunctionTool) -> None:
        self.registered.append(tool)


def _install(api: _FakeAPI) -> FunctionTool:
    install(api, {})  # type: ignore[arg-type]
    assert len(api.registered) == 1
    return api.registered[0]




async def test_peer_send_wait_resolves_on_reply() -> None:
    peer = _FakePeerMessaging()
    api = _FakeAPI({PEER_MESSAGING_SERVICE: peer})
    tool = _install(api)

    async def deliver_soon() -> None:
        await asyncio.sleep(0.05)
        peer.deliver("cid-1", {"content": "the answer is 42"})

    asyncio.get_running_loop().create_task(deliver_soon())
    result: ToolResult = await tool.fn(
        {
            "to": "worker-B",
            "content": "what is the answer?",
            "wait_for_reply": True,
            "timeout_seconds": 2,
        }
    )
    assert result.is_error is False
    text = result.content[0].text  # type: ignore[union-attr]
    assert "the answer is 42" in text


async def test_peer_send_times_out() -> None:
    peer = _FakePeerMessaging()
    api = _FakeAPI({PEER_MESSAGING_SERVICE: peer})
    tool = _install(api)
    # Tool handler coerces timeout_seconds through ``float()``, so the
    # int-typed schema is only a hint to the LLM. Pass a sub-second
    # value directly to keep the test fast.
    result: ToolResult = await tool.fn(
        {
            "to": "worker-B",
            "content": "hi",
            "wait_for_reply": True,
            "timeout_seconds": 0.1,
        }
    )
    assert result.is_error is True
    text = result.content[0].text  # type: ignore[union-attr]
    assert "timed out" in text




# -- runner unit test: correlation_id reply resolves future ------------




async def test_runner_late_reply_dropped() -> None:
    """A KIND_OUTBOUND with an unknown correlation_id is logged and
    dropped — no exception, no zombie state."""
    from agentm_channels.wire import KIND_OUTBOUND, WIRE_VERSION, Envelope
    from agentm_worker.runner import WorkerRunner

    class _NullClient:
        async def send(self, env: Any) -> None:
            pass

    async def _factory(cwd: str, bus: Any, resume: Any) -> Any:
        return None

    runner = WorkerRunner(
        client=_NullClient(),  # type: ignore[arg-type]
        cwd="/tmp",
        scenario="x",
        session_factory=_factory,
    )
    env = Envelope(
        v=WIRE_VERSION,
        id="r1",
        kind=KIND_OUTBOUND,
        ts=0.0,
        body={"content": "ok"},
        correlation_id="never-seen",
    )
    # Just don't crash.
    await runner.handle_outbound_envelope(env)
    assert "never-seen" not in runner._pending_replies
