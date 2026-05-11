"""Bridge between :class:`WireServer` and the in-process v0 :class:`Gateway`.

The chosen extension point is **synthetic-channel registration**: each
connected wire peer (identified by the ``channel`` field on its first
inbound envelope) is injected into the running :class:`ChannelManager`
as a :class:`_WireChannel`, a :class:`BaseChannel` whose ``send()``
enqueues the outbound onto the peer's outbox. This reuses the existing
inbound/outbound MessageBus path unchanged: gateway dispatch logic, slash
commands, approvals, turn-complete signalling, channel allow-from — all
of it still works, with no new conditional code in :class:`Gateway`.

Rationale (vs. calling ``gateway._dispatch`` directly): going through
``MessageBus`` keeps a single dispatch contract for v0 and wire peers,
so observability / approval / retry semantics are identical. The cost
is one ``BaseChannel`` instance per wire peer — cheap and short-lived.

Phase 5a addition — ``agent_worker`` peers
------------------------------------------
A second peer kind on the wire: workers that hold the agent session.
The router policy (scenario-equality + sticky session_key) lives in
:class:`agentm_channels.worker_registry.WorkerRegistry`. The bridge
applies it on every inbound, and forwards worker-originated
``KIND_OUTBOUND`` envelopes back to the originating chat client by
matching ``body["channel"]`` against the synthetic-channel map.

Phase 6 additions — agent-to-agent (A2A) calls
----------------------------------------------
* **Hop limit.** ``max_a2a_hops`` (default 10) bounds delegation depth.
  The bridge increments ``hops`` on every forwarded inbound; envelopes
  that would exceed the cap are dropped and a ``KIND_ERROR`` with
  ``reason="hop_limit_exceeded"`` is sent back to the source peer.
* **root_session_key propagation.** When a chat client sends an
  inbound, the bridge fills ``root_session_key = "{channel}:{chat_id}"``
  on the forwarded envelope. When a worker peer sends an inbound
  (peer_send), the worker MUST set ``root_session_key`` itself; the
  bridge rejects worker-originated inbounds without it.
* **Worker→worker routing.** When the source peer is a worker, route
  by the envelope's ``to`` field (a peer_id) instead of the scenario
  registry. Unknown ``to`` → ``KIND_ERROR`` with ``reason="unknown_to"``.
* **Approval card override.** Outbound envelopes whose
  ``body.metadata.kind == "approval_request"`` are routed to the chat
  client owning ``root_session_key``, not to the synthetic channel of
  the worker that emitted them. The user's button click flows back to
  the *emitting* worker via the existing channel→peer mapping that the
  chat client's synthetic channel still owns.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from .base import BaseChannel
from .bus import InboundMessage, MessageBus, OutboundKind, OutboundMessage
from .manager import ChannelManager
from .outbox import OutboxStore
from .peer import PeerSession
from .wire import (
    KIND_ERROR,
    KIND_INBOUND,
    KIND_OUTBOUND,
    WIRE_VERSION,
    Envelope,
)
from .session_bindings import SessionBindingStore
from .worker_registry import WorkerRegistry

log = logging.getLogger("agentm_channels.wire_bridge")

DEFAULT_MAX_A2A_HOPS: int = 10


class _WireChannel(BaseChannel):
    """Outbound sink for one wire peer.

    Not auto-discovered; instantiated and injected directly into
    :class:`ChannelManager.channels` by :class:`WireBridge` on hello.
    """

    name = "_wire"
    display_name = "Wire peer"
    # Synthetic bridge channel — not a v0 in-process platform adapter,
    # so the BaseChannel deprecation warning does not apply.
    _is_stub_fixture = True

    def __init__(
        self,
        config: dict[str, Any],
        bus: MessageBus,
        *,
        peer_id: str,
        channel_name: str,
        outbox: OutboxStore,
    ) -> None:
        super().__init__(config, bus)
        self._peer_id = peer_id
        self.name = channel_name  # per-instance: the manager keys on this
        self._outbox = outbox
        self._stopped = asyncio.Event()

    async def start(self) -> None:
        self._running = True
        await self._stopped.wait()

    async def stop(self) -> None:
        self._running = False
        self._stopped.set()

    async def send(self, msg: OutboundMessage) -> None:
        # Turn-complete is internal control; chat clients want it as
        # an envelope too, but for v1 we ship it on the wire so peers
        # can mirror it (terminal channel uses it to print a marker).
        env = _outbound_to_envelope(msg, peer_id=self._peer_id)
        await asyncio.to_thread(self._outbox.enqueue, self._peer_id, env)


def _outbound_to_envelope(msg: OutboundMessage, *, peer_id: str) -> Envelope:
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
    if msg.stream_id is not None:
        body["stream_id"] = msg.stream_id
    if msg.final:
        body["final"] = True
    return Envelope(
        v=WIRE_VERSION,
        id=f"out-{peer_id}-{int(time.time() * 1_000_000)}",
        kind=KIND_OUTBOUND,
        ts=time.time(),
        body=body,
    )


def _is_approval_request(env: Envelope) -> bool:
    """Return True when ``env.body.metadata.kind == "approval_request"``.

    The approval bridge tags requests this way on the way out so chat
    renderers know to draw the buttons. Phase 6 reuses the tag to
    override worker→chat routing for cross-peer (A2A) approvals.
    """
    body = env.body if isinstance(env.body, dict) else {}
    metadata = body.get("metadata")
    if not isinstance(metadata, dict):
        return False
    return str(metadata.get("kind", "")) == "approval_request"


class WireBridge:
    """Glue object: turns wire inbound envelopes into v0 inbound messages.

    Holds the :class:`ChannelManager` (so it can inject/remove synthetic
    :class:`_WireChannel` instances on the fly) and the
    :class:`OutboxStore` (so the synthetic channel can enqueue replies).

    With Phase 5a, also holds a :class:`WorkerRegistry`. When a worker
    matches an inbound's scenario, the bridge bypasses the in-process
    MessageBus path and enqueues the original inbound envelope onto the
    worker's outbox; reply outbounds from the worker are routed back to
    the originating chat client's outbox by ``body["channel"]``.
    """

    def __init__(
        self,
        *,
        bus: MessageBus,
        manager: ChannelManager,
        outbox: OutboxStore,
        bindings: SessionBindingStore | None = None,
        worker_registry: WorkerRegistry | None = None,
        scenario: str | None = None,
        allow_inproc: bool = True,
        max_a2a_hops: int = DEFAULT_MAX_A2A_HOPS,
    ) -> None:
        self._bus = bus
        self._manager = manager
        self._outbox = outbox
        # The binding store is the routing source of truth. Tests omit
        # the kwarg and we fall back to an in-memory sqlite — fine
        # because tests construct one bridge per case. Production paths
        # in ``cli.py`` supply an on-disk store next to ``outbox.sqlite``
        # so the routing table survives gateway restart.
        self._bindings = bindings or SessionBindingStore(":memory:")
        self._workers = worker_registry or WorkerRegistry(self._bindings)
        self._scenario = scenario or ""
        self._allow_inproc = allow_inproc
        self._max_a2a_hops = max(1, int(max_a2a_hops))
        # peer_id → channel_name registered for that peer (so we can
        # tear it down on disconnect).
        self._peer_channels: dict[str, str] = {}
        # Reverse: channel_name → chat_client peer_id. Used to route
        # worker → chat outbounds back to the right peer's outbox.
        self._channel_to_peer: dict[str, str] = {}

    @property
    def worker_registry(self) -> WorkerRegistry:
        return self._workers

    @property
    def max_a2a_hops(self) -> int:
        return self._max_a2a_hops

    # -- WireServer hooks --------------------------------------------

    async def handle_peer_hello(self, session: PeerSession) -> None:
        """Register worker peers in the registry on hello.

        Chat-client peers are still registered lazily on first inbound
        (the synthetic-channel name is taken from ``body.channel`` and
        not known until then).
        """
        if session.peer_kind == "agent_worker":
            self._workers.register(session.peer_id, dict(session.capabilities))

    async def handle_peer_disconnect(self, session: PeerSession) -> None:
        """Tear down per-peer state on disconnect.

        For workers: drop the online-set entry but **leave the
        persistent session bindings untouched**. The next inbound for
        a stranded session_key will rebind to any other online host
        advertising the same scenario, and the host receives the
        binding's ``resume_id`` so the conversation continues. If no
        replacement is available, the chat client gets the standard
        "no host" outbound (same surface as a never-bound session)
        instead of a separate "worker disconnected" error.

        For chat clients: drop the synthetic channel.
        """
        if session.peer_kind == "agent_worker":
            self._workers.unregister(session.peer_id)
            return
        # chat_client (or unknown): drop the synthetic channel.
        await self.on_peer_disconnect(session.peer_id)

    async def handle_inbound(self, peer_session: PeerSession, env: Envelope) -> None:
        """Invoked by :class:`WireServer` for each ``inbound`` envelope.

        Three sources are possible:

        * Chat client → worker (Phase 5a path; resolved by scenario).
        * Worker → worker (Phase 6 A2A; resolved by ``env.to``).
        * Worker → chat client (Phase 6 A2A reply; resolved by
          ``env.to``, peer-kind ``chat_client``).
        """
        if env.kind != KIND_INBOUND:
            return  # ping/pong/bye are handled by the server already

        if peer_session.peer_kind == "agent_worker":
            await self._handle_worker_originated_inbound(peer_session, env)
            return

        # Chat-client (or other non-worker) origin: original Phase 5a
        # path, with the Phase 6 hop bump + root_session_key tagging.
        await self._handle_chat_originated_inbound(peer_session, env)

    async def _handle_chat_originated_inbound(
        self, peer_session: PeerSession, env: Envelope
    ) -> None:
        body = env.body if isinstance(env.body, dict) else {}
        channel_name = str(body.get("channel") or "")
        chat_id = str(body.get("chat_id") or "")
        sender_id = str(body.get("sender_id") or peer_session.peer_id)
        content = str(body.get("content") or "")
        button_value_raw = body.get("button_value")
        button_value = (
            str(button_value_raw) if button_value_raw is not None else None
        )
        if not channel_name or not chat_id:
            log.warning(
                "wire inbound rejected: missing channel/chat_id (peer=%s id=%s)",
                peer_session.peer_id,
                env.id,
            )
            return
        await self._ensure_channel(peer_session.peer_id, channel_name)
        session_key = f"{channel_name}:{chat_id}"
        # Chat clients don't set root_session_key; fill it in here so
        # downstream peers (workers, peer_send fan-out) have a stable
        # identifier of the user-facing conversation.
        root_session_key = env.root_session_key or session_key

        worker_peer, resume_id = self._workers.find_host(
            self._scenario, session_key
        )
        if worker_peer is not None:
            forwarded_hops = env.hops + 1
            if forwarded_hops > self._max_a2a_hops:
                await self._send_error(
                    peer_session.peer_id,
                    "hop_limit_exceeded",
                    {
                        "max_a2a_hops": self._max_a2a_hops,
                        "correlation_id": env.correlation_id,
                        "envelope_id": env.id,
                    },
                )
                return
            # Propagate the persisted resume_id (if any) so a host that
            # took over from a crashed predecessor resumes the prior
            # AgentSession instead of starting a fresh one — the whole
            # point of session-as-routing-primary.
            forwarded_body = dict(body)
            if resume_id is not None:
                forwarded_body["resume_id"] = resume_id
            forwarded = Envelope(
                v=WIRE_VERSION,
                id=f"fwd-{worker_peer}-{int(time.time() * 1_000_000)}",
                kind=KIND_INBOUND,
                ts=time.time(),
                body=forwarded_body,
                root_session_key=root_session_key,
                correlation_id=env.correlation_id,
                hops=forwarded_hops,
            )
            await asyncio.to_thread(self._outbox.enqueue, worker_peer, forwarded)
            return

        if not self._allow_inproc:
            log.warning(
                "no worker available for scenario=%r session_key=%s; "
                "gateway was started with --no-inproc-worker",
                self._scenario,
                session_key,
            )
            await self._bus.publish_outbound(
                OutboundMessage(
                    channel=channel_name,
                    chat_id=chat_id,
                    content=(
                        f"no worker available for scenario {self._scenario!r}; "
                        "gateway started with --no-inproc-worker. "
                        "start an `agentm-worker --connect <gateway>` and retry."
                    ),
                )
            )
            await self._bus.publish_outbound(
                OutboundMessage(
                    channel=channel_name,
                    chat_id=chat_id,
                    kind=OutboundKind.TURN_COMPLETE,
                )
            )
            return

        await self._bus.publish_inbound(
            InboundMessage(
                channel=channel_name,
                sender_id=sender_id,
                chat_id=chat_id,
                content=content,
                button_value=button_value,
            )
        )

    async def _handle_worker_originated_inbound(
        self, peer_session: PeerSession, env: Envelope
    ) -> None:
        """Worker→peer A2A inbound. Route by ``env.to``."""
        if not env.root_session_key:
            await self._send_error(
                peer_session.peer_id,
                "missing_root_session_key",
                {
                    "envelope_id": env.id,
                    "correlation_id": env.correlation_id,
                    "detail": (
                        "worker-originated inbound must carry "
                        "root_session_key; copy it from the inbound that "
                        "triggered this delegation"
                    ),
                },
            )
            return
        forwarded_hops = env.hops + 1
        if forwarded_hops > self._max_a2a_hops:
            await self._send_error(
                peer_session.peer_id,
                "hop_limit_exceeded",
                {
                    "max_a2a_hops": self._max_a2a_hops,
                    "correlation_id": env.correlation_id,
                    "envelope_id": env.id,
                },
            )
            return
        target = env.to or ""
        if not target:
            await self._send_error(
                peer_session.peer_id,
                "missing_to",
                {"envelope_id": env.id, "correlation_id": env.correlation_id},
            )
            return
        # Resolve ``to``: known worker peer_id, or a chat-client peer_id
        # that already owns a synthetic channel.
        if self._workers.has(target):
            dest_peer_id = target
        elif target in self._peer_channels:
            dest_peer_id = target
        else:
            await self._send_error(
                peer_session.peer_id,
                "unknown_to",
                {
                    "to": target,
                    "envelope_id": env.id,
                    "correlation_id": env.correlation_id,
                },
            )
            return
        forwarded = Envelope(
            v=WIRE_VERSION,
            id=f"a2a-fwd-{dest_peer_id}-{int(time.time() * 1_000_000)}",
            kind=KIND_INBOUND,
            ts=time.time(),
            body=dict(env.body) if isinstance(env.body, dict) else {},
            to=target,
            correlation_id=env.correlation_id,
            root_session_key=env.root_session_key,
            peer_kind=env.peer_kind,
            hops=forwarded_hops,
        )
        await asyncio.to_thread(self._outbox.enqueue, dest_peer_id, forwarded)

    async def handle_worker_outbound(
        self, worker_session: PeerSession, env: Envelope
    ) -> None:
        """Route a worker-originated ``KIND_OUTBOUND`` to the right peer.

        Three cases:

        * **Approval request** (``body.metadata.kind == "approval_request"``):
          route to the chat client identified by ``root_session_key``,
          NOT to the worker chain. This is the cross-peer approval
          forwarding described in §7.7 of the design doc.
        * **A2A reply** (``correlation_id`` is set AND a worker peer
          named in ``env.to`` is alive): forward to that worker so its
          pending ``peer_send`` future resolves.
        * **Default** (Phase 5a): match by ``body["channel"]`` → owning
          chat_client peer_id and enqueue onto that peer's outbox.
        """
        body = env.body if isinstance(env.body, dict) else {}

        # Side-channel: workers report their AgentSession id back via
        # ``body["_session_id_hint"]`` on outbound envelopes (the worker
        # runner adds it after create/resume). We persist it onto the
        # binding so a future rebind hands the next host the right
        # resume anchor. Strip the hint before forwarding — chat
        # clients don't need to see this plumbing.
        hint = body.pop("_session_id_hint", None) if isinstance(body, dict) else None
        if isinstance(hint, str) and hint:
            ch = str(body.get("channel") or "")
            cid = str(body.get("chat_id") or "")
            if ch and cid:
                await asyncio.to_thread(
                    self._workers.record_resume_id, f"{ch}:{cid}", hint
                )

        # Case 1: approval routing override. Must come before the
        # default path because approval-request bodies still carry a
        # channel field that points at the worker's synthetic channel.
        if _is_approval_request(env) and env.root_session_key:
            channel_name, _, chat_id = env.root_session_key.partition(":")
            chat_peer_id = self._channel_to_peer.get(channel_name)
            if chat_peer_id is None:
                log.warning(
                    "approval_request from worker=%s has no live chat peer "
                    "for root_session_key=%s",
                    worker_session.peer_id,
                    env.root_session_key,
                )
                return
            # Rewrite the body so the chat renderer sees its own
            # channel/chat_id (not the worker's synthetic ``_a2a`` one)
            # and can attribute the click back through the existing
            # synthetic-channel send path. The original channel/chat_id
            # are preserved under ``metadata.origin`` so a future
            # auditor can trace the chain if needed.
            rewritten_body = dict(body)
            metadata = dict(rewritten_body.get("metadata") or {})
            metadata.setdefault(
                "origin",
                {
                    "channel": rewritten_body.get("channel"),
                    "chat_id": rewritten_body.get("chat_id"),
                    "worker_peer_id": worker_session.peer_id,
                },
            )
            rewritten_body["channel"] = channel_name
            rewritten_body["chat_id"] = chat_id
            rewritten_body["metadata"] = metadata
            forwarded = Envelope(
                v=WIRE_VERSION,
                id=f"out-{chat_peer_id}-{int(time.time() * 1_000_000)}",
                kind=KIND_OUTBOUND,
                ts=time.time(),
                body=rewritten_body,
                correlation_id=env.correlation_id,
                root_session_key=env.root_session_key,
            )
            await asyncio.to_thread(
                self._outbox.enqueue, chat_peer_id, forwarded
            )
            return

        # Case 2: A2A reply — outbound targeted at another worker via
        # ``to`` and ``correlation_id``.
        target = env.to or ""
        if env.correlation_id and target and self._workers.has(target):
            forwarded = Envelope(
                v=WIRE_VERSION,
                id=f"a2a-rep-{target}-{int(time.time() * 1_000_000)}",
                kind=KIND_OUTBOUND,
                ts=time.time(),
                body=dict(body),
                to=target,
                correlation_id=env.correlation_id,
                root_session_key=env.root_session_key,
            )
            await asyncio.to_thread(self._outbox.enqueue, target, forwarded)
            return

        # Case 3 (default Phase 5a path): chat-client by channel name.
        channel_name = str(body.get("channel") or "")
        if not channel_name:
            log.warning(
                "worker outbound rejected: missing channel (worker=%s id=%s)",
                worker_session.peer_id,
                env.id,
            )
            return
        chat_peer_id = self._channel_to_peer.get(channel_name)
        if chat_peer_id is None:
            log.warning(
                "worker outbound for channel=%r has no live chat peer "
                "(worker=%s id=%s)",
                channel_name,
                worker_session.peer_id,
                env.id,
            )
            return
        # Rebuild so the envelope id is unique on the chat client's
        # outbox (workers may reuse their own id-space). The body is
        # the contract; the id is plumbing.
        forwarded = Envelope(
            v=WIRE_VERSION,
            id=f"out-{chat_peer_id}-{int(time.time() * 1_000_000)}",
            kind=KIND_OUTBOUND,
            ts=time.time(),
            body=dict(body),
            correlation_id=env.correlation_id,
            root_session_key=env.root_session_key,
        )
        await asyncio.to_thread(
            self._outbox.enqueue, chat_peer_id, forwarded
        )

    # -- helpers -----------------------------------------------------

    async def _send_error(
        self, peer_id: str, reason: str, extra: dict[str, Any]
    ) -> None:
        env = Envelope(
            v=WIRE_VERSION,
            id=f"err-{peer_id}-{int(time.time() * 1_000_000)}",
            kind=KIND_ERROR,
            ts=time.time(),
            body={"reason": reason, **extra},
        )
        await asyncio.to_thread(self._outbox.enqueue, peer_id, env)

    async def _ensure_channel(self, peer_id: str, channel_name: str) -> None:
        existing = self._peer_channels.get(peer_id)
        if existing == channel_name:
            return
        if existing is not None:
            # Peer renamed; drop the old synthetic channel first.
            await self._drop_channel(existing)
        # Inject into ChannelManager so its dispatch loop routes
        # outbounds with this channel name through us. Bypass
        # allow_from enforcement — the wire peer authenticated via
        # peer-cred; per-sender ACL is deferred (§6.3).
        ch = _WireChannel(
            {"enabled": True, "allow_from": ["*"]},
            self._bus,
            peer_id=peer_id,
            channel_name=channel_name,
            outbox=self._outbox,
        )
        self._manager.inject_channel(channel_name, ch)
        self._peer_channels[peer_id] = channel_name
        self._channel_to_peer[channel_name] = peer_id
        log.info(
            "wire peer registered as channel %r (peer_id=%s)", channel_name, peer_id
        )

    async def _drop_channel(self, channel_name: str) -> None:
        ch = self._manager.channels.pop(channel_name, None)
        # Drop reverse mapping if it points at any peer (we don't know
        # which one without a scan, but channel_name itself is the key).
        self._channel_to_peer.pop(channel_name, None)
        if ch is None:
            return
        try:
            await ch.stop()
        except Exception:  # noqa: BLE001
            log.exception("wire channel %r stop raised", channel_name)

    async def on_peer_disconnect(self, peer_id: str) -> None:
        channel = self._peer_channels.pop(peer_id, None)
        if channel is None:
            return
        await self._drop_channel(channel)
        log.info("wire peer disconnected: channel %r dropped", channel)


__all__ = ["DEFAULT_MAX_A2A_HOPS", "WireBridge", "_WireChannel"]
