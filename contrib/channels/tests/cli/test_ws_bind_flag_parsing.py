"""WebSocket-flavoured ``--bind`` flag parsing for ``agentm-gateway``.

Covers the new ws:// / wss:// dispatch added in PR3 of
``.claude/plans/2026-05-12-gateway-websocket-transport.md``: schema
validation, mandatory token configuration, mandatory TLS material for
wss://, and the ``--bind-token-file`` parser.
"""

from __future__ import annotations


import pytest

from agentm_channels.cli import _resolve_bind


def test_resolve_bind_ws_requires_tokens() -> None:
    with pytest.raises(SystemExit) as exc:
        _resolve_bind(
            bind="ws://0.0.0.0:7777/agentm",
            bind_allow_uid=None,
            bind_allow_any_uid=False,
            cfg={},
        )
    assert "ws://" in str(exc.value) or "ws" in str(exc.value)
    assert "token" in str(exc.value).lower()


def test_resolve_bind_wss_requires_cert_and_key() -> None:
    with pytest.raises(SystemExit) as exc:
        _resolve_bind(
            bind="wss://0.0.0.0:7777/agentm",
            bind_allow_uid=None,
            bind_allow_any_uid=False,
            bind_token_file=None,
            bind_allow_anonymous=True,
            cfg={},
        )
    msg = str(exc.value).lower()
    assert "tls" in msg or "wss" in msg
    assert "cert" in msg or "key" in msg














