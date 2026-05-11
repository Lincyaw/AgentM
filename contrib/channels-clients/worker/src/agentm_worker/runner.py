"""Worker run loop.

Connects to an ``agentm-gateway --bind unix://…``, advertises a list of
scenarios as ``capabilities.scenarios`` at hello, then on every
forwarded inbound envelope drives the same code paths the gateway used
to run in-process. Internally we instantiate a private
:class:`MessageBus` + :class:`Gateway` so slash-command dispatch,
approval bridge, turn-complete signalling, and session resumption all
keep working unchanged — the worker is a *thin shim* on top of the
existing harness.

Outbound from the local bus is intercepted by a consumer task and
re-emitted as ``KIND_OUTBOUND`` envelopes on the wire. The gateway
routes those back to the chat client.

Concurrency: a single ``asyncio.Semaphore`` caps in-flight
``session.prompt`` calls (``--max-concurrency``). The cap is a guard
against runaway concurrent LLM traffic from one busy chat surface; the
per-session-key serialization that the gateway's per-route lock
already provides is what keeps a single conversation's turns ordered.

Phase 6 addition — agent-to-agent (A2A) call support
----------------------------------------------------
The runner implements :class:`agentm_worker.peer_send_atom.PeerMessaging`
and publishes itself onto each newly-created :class:`AgentSession`'s
service registry under the key ``peer_messaging``. The optional
``tool_peer_send`` atom looks that key up at install time to register
its tool. Reply matching uses a per-runner ``correlation_id → Future``
dict; outbound envelopes from the wire whose ``correlation_id`` is in
the dict resolve the future instead of feeding back into a local
``AgentSession``.

Late replies (after timeout / wait_for_reply=False) are logged and
dropped; no zombie state survives across turns.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any

from agentm_channels.bus import (
    InboundMessage,
    MessageBus,
    OutboundKind,
    OutboundMessage,
)
from agentm_channels.client import WireClient
from agentm_channels.gateway import Gateway, GatewayConfig, SessionFactory
from agentm_channels.wire import (
    KIND_INBOUND,
    KIND_OUTBOUND,
    WIRE_VERSION,
    Envelope,
)

log = logging.getLogger("agentm_worker.runner")

PEER_MESSAGING_SERVICE = "peer_messaging"


class WorkerRunner:
    """End-to-end glue: wire ↔ local MessageBus ↔ in-process Gateway.

    Also implements the ``PeerMessaging`` protocol consumed by
    ``tool_peer_send``. Atoms reach this implementation through
    :meth:`AgentSession.set_service` — the runner subscribes to the
    gateway's session-created path and stamps the service in before the
    first turn runs.
    """

    def __init__(
        self,
        *,
        client: WireClient,
        cwd: str,
        scenario: str | None,
        session_factory: SessionFactory,
        max_concurrency: int = 4,
    ) -> None:
        self._client = client
        self._cwd = cwd
        self._scenario = scenario
        # Wrap the user-supplied factory so we can stamp peer_messaging
        # onto every AgentSession the Gateway materialises. Keeping the
        # wrap here (instead of in cli.py) means anyone embedding
        # WorkerRunner gets A2A support without re-wiring the factory.
        self._session_factory = self._wrap_factory(session_factory)
        self._semaphore = asyncio.Semaphore(max(1, max_concurrency))
        self._bus = MessageBus()
        # The Gateway is happy with no command_registry; it discovers
        # one from cwd. That is the same behaviour the in-process path
        # had before the split.
        self._gateway = Gateway(
            bus=self._bus,
            config=GatewayConfig(cwd=cwd, scenario=scenario),
            session_factory=self._session_factory,
        )
        self._outbound_task: asyncio.Task[None] | None = None
        self._stopped = asyncio.Event()
        # Pending peer_send replies awaiting their KIND_OUTBOUND echo.
        self._pending_replies: dict[str, asyncio.Future[dict[str, Any]]] = {}
        # The last forwarded inbound we processed for each session_key.
        # peer_send copies root_session_key off the inbound envelope so
        # the gateway can route approval cards back to the user chat.
        # Keyed by session_key (channel:chat_id) because that is the
        # natural scope an AgentSession turn runs in.
        self._inflight_root_session_key: dict[str, str] = {}

    def _wrap_factory(self, inner: SessionFactory) -> SessionFactory:
        async def factory(cwd: str, bus: Any, resume: str | None) -> Any:
            session = await inner(cwd, bus, resume)
            # Best-effort: older sessions lacking set_service still work
            # (the atom raises a clear error at install time).
            setter = getattr(session, "set_service", None)
            if callable(setter):
                try:
                    setter(PEER_MESSAGING_SERVICE, self)
                except KeyError:
                    # Already registered (e.g. resumed session) — leave
                    # the existing entry alone.
                    pass
            return session

        return factory

    # -- lifecycle ---------------------------------------------------

    async def start(self) -> None:
        await self._gateway.start()
        self._outbound_task = asyncio.create_task(
            self._consume_outbound(), name="worker-outbound"
        )

    async def stop(self) -> None:
        if self._stopped.is_set():
            return
        self._stopped.set()
        if self._outbound_task is not None:
            self._outbound_task.cancel()
            try:
                await self._outbound_task
            except (asyncio.CancelledError, Exception):
                pass
        # Fail all pending peer_send waiters so caller turns unwind.
        for fut in list(self._pending_replies.values()):
            if not fut.done():
                fut.set_exception(RuntimeError("worker shutting down"))
        self._pending_replies.clear()
        # Gateway.stop() iterates routes and calls session.shutdown()
        # for each — that is also our per-chat session teardown.
        await self._gateway.stop()

    # -- inbound: wire → bus -----------------------------------------

    async def handle_inbound_envelope(self, env: Envelope) -> None:
        """Translate a ``KIND_INBOUND`` envelope to an :class:`InboundMessage`
        on the local bus.

        The gateway's worker-routing path forwards inbounds as
        ``KIND_INBOUND``; the worker treats them exactly like a fresh
        user turn. Concurrency is bounded by ``--max-concurrency``.
        """
        if env.kind != KIND_INBOUND:
            return
        body = env.body if isinstance(env.body, dict) else {}
        channel_name = str(body.get("channel") or "")
        chat_id = str(body.get("chat_id") or "")
        if not channel_name or not chat_id:
            log.warning(
                "worker dropped malformed inbound id=%s (missing channel/chat_id)",
                env.id,
            )
            return
        sender_id = str(body.get("sender_id") or "")
        content = str(body.get("content") or "")
        button_value_raw = body.get("button_value")
        button_value = (
            str(button_value_raw) if button_value_raw is not None else None
        )
        # Phase 6: remember the root_session_key for the in-flight turn
        # so peer_send can propagate it. Falls back to the local
        # session_key when the envelope has no explicit value (first
        # hop from a chat client without the field set).
        session_key = f"{channel_name}:{chat_id}"
        self._inflight_root_session_key[session_key] = (
            env.root_session_key or session_key
        )
        # Session-as-routing-primary: the gateway hands us the
        # ``resume_id`` for this session_key when the binding store has
        # one (typically because a previous host crashed). Stash it in
        # the local chat_session_map so the Gateway's session_factory
        # picks it up on the very next ``session_factory(cwd, bus,
        # resume_id)`` call. First-bind inbounds carry no resume_id —
        # the Gateway then writes its freshly-created session_id back
        # into chat_session_map, which we surface to the gateway-side
        # binding store on the next outbound via ``_session_id_hint``.
        resume_id_raw = body.get("resume_id")
        if isinstance(resume_id_raw, str) and resume_id_raw:
            self._gateway._chat_map.set(session_key, resume_id_raw)  # type: ignore[attr-defined]
        async with self._semaphore:
            await self._bus.publish_inbound(
                InboundMessage(
                    channel=channel_name,
                    sender_id=sender_id,
                    chat_id=chat_id,
                    content=content,
                    button_value=button_value,
                )
            )

    async def handle_outbound_envelope(self, env: Envelope) -> None:
        """Phase 6: incoming reply for an in-flight peer_send.

        Called by the CLI's ``on_outbound`` callback whenever the
        gateway routes a ``KIND_OUTBOUND`` envelope back to this worker
        because it carries our worker as ``to``. Match by
        ``correlation_id``: if a future is waiting, resolve it; otherwise
        log and drop (a late reply after timeout / wait_for_reply=False).
        """
        if env.kind != KIND_OUTBOUND:
            return
        cid = env.correlation_id
        if cid is None:
            return
        fut = self._pending_replies.pop(cid, None)
        if fut is None:
            log.info(
                "worker dropping late peer_send reply correlation_id=%s",
                cid,
            )
            return
        if fut.done():
            return
        body = env.body if isinstance(env.body, dict) else {}
        fut.set_result(dict(body))

    # -- outbound: bus → wire ----------------------------------------

    async def _consume_outbound(self) -> None:
        """Drain the local bus and ship each message as a wire envelope."""
        try:
            while not self._stopped.is_set():
                msg: OutboundMessage = await self._bus.consume_outbound()
                session_key = f"{msg.channel}:{msg.chat_id}"
                root = self._inflight_root_session_key.get(session_key)
                # Surface the locally-known AgentSession id back to the
                # gateway so the binding store records it. Gateway uses
                # the hint to populate ``resume_id`` on the binding;
                # next time this session_key rebinds to a fresh host,
                # the new host receives the hint and resumes.
                session_id_hint = self._gateway._chat_map.get(session_key)  # type: ignore[attr-defined]
                env = _outbound_to_envelope(
                    msg,
                    root_session_key=root,
                    session_id_hint=session_id_hint,
                )
                try:
                    await self._client.send(env)
                except Exception:
                    log.exception("worker failed to forward outbound to gateway")
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("worker outbound consumer crashed")

    # -- PeerMessaging protocol --------------------------------------

    def new_correlation_id(self) -> str:
        return uuid.uuid4().hex

    async def send_peer(
        self,
        *,
        to: str,
        content: str,
        correlation_id: str,
    ) -> None:
        """Build + send a peer-targeted inbound envelope.

        Copies the in-flight root_session_key off the most recent
        forwarded inbound so the gateway can route approval cards back
        to the user-facing chat session. ``hops`` is set to 1 — the
        worker is hop #1 from the original chat client (which was
        hop #0); the gateway increments on forward and the hop-limit
        guard fires from there.
        """
        # Best-effort lookup: a peer_send call always happens during a
        # turn, and exactly one session_key has an inflight turn on this
        # worker per session. When there are multiple, the most recent
        # is the one whose tool the LLM just invoked; the dict order
        # gives that.
        root = next(reversed(self._inflight_root_session_key.values()), None)
        if root is None:
            # Fired outside a turn context (tests, future direct API
            # use). The gateway will drop with missing_root_session_key
            # — surface a clear error to the caller via the future path.
            raise RuntimeError(
                "peer_send cannot be invoked outside a forwarded-turn "
                "context: no root_session_key is currently bound"
            )
        env = Envelope(
            v=WIRE_VERSION,
            id=f"a2a-{int(time.time() * 1_000_000)}",
            kind=KIND_INBOUND,
            ts=time.time(),
            body={
                "channel": "_a2a",
                "chat_id": correlation_id,
                "sender_id": "agent",
                "content": content,
            },
            to=to,
            correlation_id=correlation_id,
            root_session_key=root,
            peer_kind="agent_worker",
            hops=1,
        )
        # Pre-register the future BEFORE sending so a fast reply can't
        # race us into the "late reply" branch.
        loop = asyncio.get_running_loop()
        self._pending_replies.setdefault(correlation_id, loop.create_future())
        await self._client.send(env)

    async def await_peer_reply(
        self,
        correlation_id: str,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        fut = self._pending_replies.get(correlation_id)
        if fut is None:
            fut = loop.create_future()
            self._pending_replies[correlation_id] = fut
        try:
            return await asyncio.wait_for(fut, timeout=timeout_seconds)
        except asyncio.TimeoutError as exc:
            # Clean up so a late reply doesn't resolve a stale future.
            self._pending_replies.pop(correlation_id, None)
            raise TimeoutError(
                f"peer_send timed out after {timeout_seconds:g}s "
                f"(correlation_id={correlation_id})"
            ) from exc


def _outbound_to_envelope(
    msg: OutboundMessage,
    *,
    root_session_key: str | None = None,
    session_id_hint: str | None = None,
) -> Envelope:
    body: dict[str, Any] = {
        "channel": msg.channel,
        "chat_id": msg.chat_id,
        "content": msg.content,
        "kind": msg.kind.value
        if isinstance(msg.kind, OutboundKind)
        else str(msg.kind),
    }
    if msg.buttons:
        body["buttons"] = [
            {"label": b.label, "value": b.value, "style": b.style}
            for b in msg.buttons
        ]
    if msg.metadata:
        body["metadata"] = dict(msg.metadata)
    if session_id_hint:
        # Side-channel: the gateway's WireBridge strips this before
        # forwarding to the chat client and writes it onto the
        # binding store. See the WireBridge.handle_worker_outbound
        # comment for the rationale.
        body["_session_id_hint"] = session_id_hint
    return Envelope(
        v=WIRE_VERSION,
        id=f"wkr-out-{int(time.time() * 1_000_000)}",
        kind=KIND_OUTBOUND,
        ts=time.time(),
        body=body,
        root_session_key=root_session_key,
    )


__all__ = ["WorkerRunner", "PEER_MESSAGING_SERVICE"]
