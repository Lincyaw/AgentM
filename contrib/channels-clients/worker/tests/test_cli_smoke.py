"""CLI smoke tests for ``agentm-worker``.

No subprocess, no real socket, no LLM.
"""

from __future__ import annotations

import os

from typer.testing import CliRunner

os.environ.setdefault("AGENTM_SKIP_DOTENV", "1")

from agentm_worker import cli as worker_cli  # noqa: E402

runner = CliRunner()


def test_help_exits_zero() -> None:
    result = runner.invoke(worker_cli.app, ["--help"])
    assert result.exit_code == 0
    assert "agentm-worker" in result.stdout
    assert "--connect" in result.stdout
    assert "--scenario" in result.stdout
    assert "--cwd" in result.stdout
    assert "--max-concurrency" in result.stdout
    # Examples section keeps the contract discoverable from --help alone.
    assert "Examples" in result.stdout


def test_version_prints_and_exits_zero() -> None:
    result = runner.invoke(worker_cli.app, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.stdout


def test_connect_to_missing_socket_exits_seven(tmp_path) -> None:
    sock = tmp_path / "nope.sock"
    result = runner.invoke(worker_cli.app, ["--connect", f"unix://{sock}"])
    assert result.exit_code == 7
    assert "connect-failed" in result.stderr


def test_bad_connect_scheme_exits_two() -> None:
    result = runner.invoke(worker_cli.app, ["--connect", "tcp://127.0.0.1:7000"])
    assert result.exit_code == 2
    assert "agentm-worker: error" in result.stderr
    assert "unix://" in result.stderr


def test_empty_connect_path_exits_two() -> None:
    # ``unix://`` with no netloc and no path → bad-argument.
    result = runner.invoke(worker_cli.app, ["--connect", "unix://"])
    assert result.exit_code == 2
