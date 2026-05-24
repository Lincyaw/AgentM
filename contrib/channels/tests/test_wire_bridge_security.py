"""Fail-stop coverage for the three blocker-grade security fixes in
``contrib/channels/``.

Each test pins a specific spoofing / squatting / timing-oracle path
that, if regressed, would let an authenticated peer impersonate another
or exfiltrate token bits. They are deliberately narrow — they don't
re-verify the full envelope contract (other tests do that).

* ``test_inbound_sender_id_preserved`` — a chat client relays many
  humans, so the per-message ``sender_id`` (the channel-local author)
  is the client's to assert and MUST be preserved end-to-end, not
  collapsed onto the connection principal. Peer impersonation is
  prevented by channel binding (the squatting test below), not by
  forcing author == principal.
* ``test_channel_name_squatting_rejected`` — once peer A binds
  ``channel_name=X``, peer B sending another inbound with the same
  ``channel_name`` must get a ``KIND_ERROR`` and NOT have its inbound
  forwarded.
* ``test_token_authenticator_uses_constant_time`` — structural assertion
  that the authenticator's membership check funnels through
  :func:`hmac.compare_digest` rather than ``in`` on a ``set``. We can't
  meaningfully measure timing in a unit test, so we test the structure
  that guarantees the property.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass, field

import pytest

from agentm_channels.auth.token import TokenAuthenticator, _constant_time_membership
from agentm_channels.manager import ChannelManager
from agentm_channels.bus import MessageBus
from agentm_channels.peer import PeerSession
from agentm_channels.wire import (
    KIND_ERROR,
    KIND_INBOUND,
    WIRE_VERSION,
    Envelope,
)
from agentm_channels.wire_bridge import WireBridge


@dataclass
class _RecordedOutbox:
    """In-memory stand-in for :class:`SqliteOutbox`.

    The bridge only calls ``enqueue(peer_id, env)``; nothing else needs
    to be implemented for the tests below.
    """

    items: list[tuple[str, Envelope]] = field(default_factory=list)

    def enqueue(self, peer_id: str, env: Envelope) -> None:
        self.items.append((peer_id, env))


def _new_bus() -> MessageBus:
    return MessageBus()


def _peer(peer_id: str) -> PeerSession:
    return PeerSession(
        peer_id=peer_id,
        peer_kind="chat_client",
        transport_writer=None,  # type: ignore[arg-type]
    )


def _inbound(
    *,
    env_id: str,
    channel: str,
    chat_id: str,
    sender_id: str,
    content: str = "hello",
) -> Envelope:
    return Envelope(
        v=WIRE_VERSION,
        id=env_id,
        kind=KIND_INBOUND,
        ts=time.time(),
        body={
            "channel": channel,
            "chat_id": chat_id,
            "sender_id": sender_id,
            "content": content,
        },
    )


@pytest.mark.asyncio
async def test_inbound_sender_id_preserved() -> None:
    """A chat client's per-message author must survive to the bus.

    The connecting peer (``feishu-client``) relays a message authored
    by a human (``ou_alice``) whose id is deliberately distinct from
    the peer principal. The bridge must forward that author verbatim —
    not overwrite it with the connection principal — so downstream
    per-human authorization (approvals) and observability stay honest.
    No ``KIND_ERROR`` may be sent back.
    """
    bus = _new_bus()
    manager = ChannelManager({}, bus)
    outbox = _RecordedOutbox()
    bridge = WireBridge(
        bus=bus, manager=manager, outbox=outbox  # type: ignore[arg-type]
    )

    client = _peer("feishu-client")
    inbound = _inbound(
        env_id="in-1",
        channel="feishu",
        chat_id="oc_chat",
        sender_id="ou_alice",  # the human, NOT the peer principal
    )
    await bridge.handle_inbound(client, inbound)

    # The author reaches the bus unchanged.
    msg = await asyncio.wait_for(bus.consume_inbound(), timeout=1.0)
    assert msg.sender_id == "ou_alice"
    assert msg.channel == "feishu"
    assert msg.chat_id == "oc_chat"

    # No spoof rejection — the client owns its channel.
    errors = [env for (_pid, env) in outbox.items if env.kind == KIND_ERROR]
    assert errors == [], errors

    await manager.stop()


@pytest.mark.asyncio
async def test_inbound_missing_sender_id_falls_back_to_principal() -> None:
    """A single-user transport may omit the author; the bridge fills
    it from the peer principal so downstream always sees a stable id.
    """
    bus = _new_bus()
    manager = ChannelManager({}, bus)
    outbox = _RecordedOutbox()
    bridge = WireBridge(
        bus=bus, manager=manager, outbox=outbox  # type: ignore[arg-type]
    )

    client = _peer("terminal-7f3a")
    env = Envelope(
        v=WIRE_VERSION,
        id="in-2",
        kind=KIND_INBOUND,
        ts=time.time(),
        body={"channel": "terminal", "chat_id": "c1", "content": "hi"},
    )
    await bridge.handle_inbound(client, env)

    msg = await asyncio.wait_for(bus.consume_inbound(), timeout=1.0)
    assert msg.sender_id == "terminal-7f3a"

    await manager.stop()


@pytest.mark.asyncio
async def test_channel_name_squatting_rejected() -> None:
    """Issue #2: once peer A claims ``channel_name=feishu``, a second
    peer B trying to use the same name must be rejected with a
    ``KIND_ERROR`` instead of silently failing inside ``_dispatch_safely``.
    """
    bus = _new_bus()
    manager = ChannelManager({}, bus)
    outbox = _RecordedOutbox()
    bridge = WireBridge(
        bus=bus, manager=manager, outbox=outbox  # type: ignore[arg-type]
    )

    peer_a = _peer("peer-a")
    peer_b = _peer("peer-b")

    # First claim wins — peer-a binds channel_name "feishu".
    await bridge.handle_inbound(
        peer_a,
        _inbound(
            env_id="a-1",
            channel="feishu",
            chat_id="chat-a",
            sender_id="peer-a",
        ),
    )
    # Drain the legitimate inbound so it doesn't confound the second
    # check.
    await asyncio.wait_for(bus.consume_inbound(), timeout=1.0)

    # Squatter peer-b tries the same name.
    await bridge.handle_inbound(
        peer_b,
        _inbound(
            env_id="b-1",
            channel="feishu",
            chat_id="chat-b",
            sender_id="peer-b",
        ),
    )

    # Squatter's inbound must NOT reach the bus.
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(bus.consume_inbound(), timeout=0.1)

    # The outbox should contain at least one error addressed to peer-b
    # carrying reason=channel_name_conflict.
    errors_to_b = [
        env
        for (pid, env) in outbox.items
        if pid == "peer-b"
        and env.kind == KIND_ERROR
        and isinstance(env.body, dict)
        and env.body.get("reason") == "channel_name_conflict"
    ]
    assert len(errors_to_b) == 1, outbox.items
    assert errors_to_b[0].body["channel_name"] == "feishu"

    await manager.stop()


def test_token_authenticator_uses_constant_time() -> None:
    """Issue #3: token comparison must use ``hmac.compare_digest``
    OR-accumulated across the full allow-list. We can't time-test in a
    unit test, so we assert the structural property that guarantees it:

    1. The helper :func:`_constant_time_membership` mentions
       ``compare_digest`` in its source.
    2. ``TokenAuthenticator`` does not perform a naive ``in`` check on a
       ``set`` of tokens (the regression we're guarding against).
    3. Behavioural smoke: wrong token rejected, right token accepted.
    """
    src = inspect.getsource(_constant_time_membership)
    assert "compare_digest" in src, (
        "constant-time membership must funnel through hmac.compare_digest"
    )

    auth_src = inspect.getsource(TokenAuthenticator)
    # The old vulnerable form was `token in self._allowed_tokens` —
    # naive ``in`` against a hashed container shortcircuits and leaks
    # timing. The fix routes through :func:`_constant_time_membership`.
    assert "_constant_time_membership" in auth_src, (
        "TokenAuthenticator must delegate the membership check to "
        "_constant_time_membership; a naive ``in`` leaks a timing oracle"
    )
    assert "token in self._allowed_tokens" not in auth_src, (
        "naive set membership reintroduces the timing oracle"
    )

    # Behavioural smoke — wrong token rejected, right one accepted.
    auth = TokenAuthenticator({"correct-horse-battery-staple"})

    async def _run() -> None:
        # transport is unused by token auth (see authenticate docstring)
        # so passing ``None`` is safe; the noqa-marked param accepts it.
        ok = await auth.authenticate(
            peer_kind="chat_client",
            peer_id="p",
            token="correct-horse-battery-staple",
            transport=None,  # type: ignore[arg-type]
        )
        assert ok is True
        bad = await auth.authenticate(
            peer_kind="chat_client",
            peer_id="p",
            token="wrong",
            transport=None,  # type: ignore[arg-type]
        )
        assert bad is False
        # Same prefix as a real token — would have leaked timing under
        # the naive ``in`` check.
        bad_prefix = await auth.authenticate(
            peer_kind="chat_client",
            peer_id="p",
            token="correct-horse-battery-stapl",
            transport=None,  # type: ignore[arg-type]
        )
        assert bad_prefix is False

    asyncio.run(_run())
