"""CLI smoke tests — no subprocess, no real socket."""

from __future__ import annotations

import io
import os

import pytest
from typer.testing import CliRunner

# Disable the import-time .env autoload so the repo's own .env doesn't
# leak provider keys into the test's environment.
os.environ.setdefault("AGENTM_SKIP_DOTENV", "1")

from agentm_terminal import cli as terminal_cli  # noqa: E402

runner = CliRunner()


def test_bad_connect_scheme_exits_two() -> None:
    result = runner.invoke(terminal_cli.app, ["--connect", "tcp://127.0.0.1:7000"])
    assert result.exit_code == 2
    assert "agentm-terminal: error" in result.stderr
    assert "unix://" in result.stderr


def test_connect_to_missing_socket_exits_seven(tmp_path) -> None:
    # --connect has a default (matches agentm-gateway). Aim at a tmp
    # path that doesn't exist so the connect path is exercised
    # regardless of whether a real gateway is running on the conventional
    # XDG socket.
    sock = tmp_path / "nope.sock"
    result = runner.invoke(terminal_cli.app, ["--connect", f"unix://{sock}"])
    assert result.exit_code == 7
    assert "connect-failed" in result.stderr


def test_resolve_format_defaults_text_for_tty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.stdout", _FakeStream(isatty=True))
    assert terminal_cli._resolve_format(None) == "text"


def test_resolve_format_defaults_json_for_pipe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.stdout", _FakeStream(isatty=False))
    assert terminal_cli._resolve_format(None) == "json"


class _FakeStream(io.StringIO):
    def __init__(self, *, isatty: bool) -> None:
        super().__init__()
        self._isatty = isatty

    def isatty(self) -> bool:  # type: ignore[override]
        return self._isatty
