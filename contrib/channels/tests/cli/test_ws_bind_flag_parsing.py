"""WebSocket-flavoured ``--bind`` flag parsing for ``agentm-gateway``.

Covers the new ws:// / wss:// dispatch added in PR3 of
``.claude/plans/2026-05-12-gateway-websocket-transport.md``: schema
validation, mandatory token configuration, mandatory TLS material for
wss://, and the ``--bind-token-file`` parser.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from agentm_channels.cli import _load_tokens_file, _resolve_bind


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


def test_resolve_bind_ws_with_tls_rejected() -> None:
    with pytest.raises(SystemExit) as exc:
        _resolve_bind(
            bind="ws://0.0.0.0:7777/agentm",
            bind_allow_uid=None,
            bind_allow_any_uid=False,
            bind_allow_anonymous=True,
            tls_cert="/tmp/cert.pem",
            tls_key="/tmp/key.pem",
            cfg={},
        )
    assert "ws://" in str(exc.value)


def test_resolve_bind_unix_with_tls_rejected() -> None:
    with pytest.raises(SystemExit) as exc:
        _resolve_bind(
            bind="unix:///tmp/x.sock",
            bind_allow_uid=None,
            bind_allow_any_uid=False,
            tls_cert="/tmp/cert.pem",
            cfg={},
        )
    assert "unix" in str(exc.value).lower() or "tls" in str(exc.value).lower()


def test_resolve_bind_ws_with_allow_uid_rejected() -> None:
    # Peer-cred uid flags are unix-only.
    with pytest.raises(SystemExit) as exc:
        _resolve_bind(
            bind="ws://0.0.0.0:7777/agentm",
            bind_allow_uid=[1000],
            bind_allow_any_uid=False,
            cfg={},
        )
    assert "unix" in str(exc.value).lower()


def test_load_tokens_file_skips_comments_and_blanks() -> None:
    with tempfile.NamedTemporaryFile(
        "w", suffix=".tokens", delete=False, encoding="utf-8"
    ) as f:
        f.write(
            "# leading comment\n"
            "\n"
            "alpha\n"
            "  beta  \n"  # whitespace stripped
            "   # indented comment, still skipped\n"
            "\n"
            "gamma\n"
        )
        path = f.name
    try:
        tokens = _load_tokens_file(path)
    finally:
        os.unlink(path)
    assert tokens == {"alpha", "beta", "gamma"}


def test_resolve_bind_ws_with_token_file_resolves(tmp_path: Path) -> None:
    tokens_file = tmp_path / "tokens"
    tokens_file.write_text("# header\nfoo\nbar\n", encoding="utf-8")
    spec = _resolve_bind(
        bind="ws://0.0.0.0:7777/agentm",
        bind_allow_uid=None,
        bind_allow_any_uid=False,
        bind_token_file=str(tokens_file),
        cfg={},
    )
    assert spec.scheme == "ws"
    assert spec.host == "0.0.0.0"
    assert spec.port == 7777
    assert spec.ws_path == "/agentm"
    assert spec.tokens == frozenset({"foo", "bar"})


def test_check_ws_without_tokens_fails_fast() -> None:
    """End-to-end: --bind ws:// without tokens exits 2."""
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "agentm_channels.cli",
            "--bind",
            "ws://127.0.0.1:0/agentm",
            "--check",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "AGENTM_SKIP_DOTENV": "1"},
    )
    assert proc.returncode == 2, (proc.returncode, proc.stdout, proc.stderr)
    assert "token" in (proc.stderr + proc.stdout).lower()


def test_check_wss_without_tls_material_fails_fast(tmp_path: Path) -> None:
    """End-to-end: --bind wss:// without --tls-cert/--tls-key exits 2."""
    tokens_file = tmp_path / "tokens"
    tokens_file.write_text("alpha\n", encoding="utf-8")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "agentm_channels.cli",
            "--bind",
            "wss://127.0.0.1:0/agentm",
            "--bind-token-file",
            str(tokens_file),
            "--check",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "AGENTM_SKIP_DOTENV": "1"},
    )
    assert proc.returncode == 2, (proc.returncode, proc.stdout, proc.stderr)
    combined = (proc.stderr + proc.stdout).lower()
    assert "tls" in combined or "cert" in combined
