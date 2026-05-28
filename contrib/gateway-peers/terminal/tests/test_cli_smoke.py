"""CLI smoke tests — no subprocess, no real socket."""

from __future__ import annotations

import io
import os

from typer.testing import CliRunner

# Disable the import-time .env autoload so the repo's own .env doesn't
# leak provider keys into the test's environment.
os.environ.setdefault("AGENTM_SKIP_DOTENV", "1")

from agentm_terminal import cli as terminal_cli  # noqa: E402

runner = CliRunner()




def test_connect_to_missing_socket_exits_seven(tmp_path) -> None:
    # --connect has a default (matches agentm-gateway). Aim at a tmp
    # path that doesn't exist so the connect path is exercised
    # regardless of whether a real gateway is running on the conventional
    # XDG socket.
    sock = tmp_path / "nope.sock"
    result = runner.invoke(terminal_cli.app, ["--connect", f"unix://{sock}"])
    assert result.exit_code == 7
    assert "connect-failed" in result.stderr






class _FakeStream(io.StringIO):
    def __init__(self, *, isatty: bool) -> None:
        super().__init__()
        self._isatty = isatty

    def isatty(self) -> bool:  # type: ignore[override]
        return self._isatty
