"""CLI smoke tests for ``agentm-feishu``.

No subprocess, no real Feishu, no real socket. ``--check-config`` is
used to drive the happy path through argument parsing and config
resolution without contacting the gateway or Feishu.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Disable autoload so the repo's own .env doesn't poison the fixture.
# Must happen BEFORE importing the cli module.
os.environ.setdefault("AGENTM_SKIP_DOTENV", "1")

from typer.testing import CliRunner  # noqa: E402

from agentm_feishu import cli as feishu_cli  # noqa: E402

runner = CliRunner()


@pytest.fixture(autouse=True)
def _scrub_lark_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test starts from a clean LARK_* environment."""
    monkeypatch.delenv("LARK_APP_ID", raising=False)
    monkeypatch.delenv("LARK_APP_SECRET", raising=False)


def test_bad_connect_scheme_exits_two() -> None:
    result = runner.invoke(
        feishu_cli.app,
        ["--connect", "tcp://127.0.0.1:7000", "--app-id", "cli_xxxx", "--check-config"],
    )
    assert result.exit_code == 2
    assert "agentm-feishu: error" in result.stderr
    assert "unix://" in result.stderr


def test_missing_app_id_exits_two() -> None:
    result = runner.invoke(
        feishu_cli.app, ["--connect", "unix:///tmp/gw.sock", "--check-config"]
    )
    assert result.exit_code == 2
    assert "missing --app-id" in result.stderr


def test_missing_app_secret_exits_two() -> None:
    result = runner.invoke(
        feishu_cli.app,
        [
            "--connect",
            "unix:///tmp/gw.sock",
            "--app-id",
            "cli_xxxx",
            "--check-config",
        ],
    )
    assert result.exit_code == 2
    assert "missing app secret" in result.stderr


def test_empty_secret_file_exits_two(tmp_path: Path) -> None:
    secret_file = tmp_path / "secret"
    secret_file.write_text("")
    result = runner.invoke(
        feishu_cli.app,
        [
            "--connect",
            "unix:///tmp/gw.sock",
            "--app-id",
            "cli_xxxx",
            "--app-secret",
            str(secret_file),
            "--check-config",
        ],
    )
    assert result.exit_code == 2
    assert "empty" in result.stderr


def test_check_config_success_with_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LARK_APP_ID", "cli_xxxx")
    monkeypatch.setenv("LARK_APP_SECRET", "secret_value")
    result = runner.invoke(
        feishu_cli.app, ["--connect", "unix:///tmp/gw.sock", "--check-config"]
    )
    assert result.exit_code == 0


def test_check_config_success_with_file(tmp_path: Path) -> None:
    secret_file = tmp_path / "secret"
    secret_file.write_text("secret_value\n")
    result = runner.invoke(
        feishu_cli.app,
        [
            "--connect",
            "unix:///tmp/gw.sock",
            "--app-id",
            "cli_xxxx",
            "--app-secret",
            str(secret_file),
            "--check-config",
        ],
    )
    assert result.exit_code == 0


def test_secret_file_wins_over_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When both are set the file wins, per CLI contract."""
    monkeypatch.setenv("LARK_APP_SECRET", "from_env")
    secret_file = tmp_path / "secret"
    secret_file.write_text("from_file")
    cfg = feishu_cli._resolve_config(
        app_id="cli_xxxx",
        app_secret_path=str(secret_file),
        allow_from=None,
        chat_id_prefix="feishu",
    )
    assert cfg.app_secret == "from_file"


def test_bad_secret_file_path_exits_two() -> None:
    result = runner.invoke(
        feishu_cli.app,
        [
            "--connect",
            "unix:///tmp/gw.sock",
            "--app-id",
            "cli_xxxx",
            "--app-secret",
            "/nonexistent/path/that/does/not/exist",
            "--check-config",
        ],
    )
    assert result.exit_code == 2
    assert "cannot read --app-secret file" in result.stderr


def test_default_allow_from_is_star(tmp_path: Path) -> None:
    secret_file = tmp_path / "secret"
    secret_file.write_text("x")
    cfg = feishu_cli._resolve_config(
        app_id="cli_xxxx",
        app_secret_path=str(secret_file),
        allow_from=None,
        chat_id_prefix="feishu",
    )
    assert cfg.allow_from == ["*"]


def test_repeated_allow_from(tmp_path: Path) -> None:
    secret_file = tmp_path / "secret"
    secret_file.write_text("x")
    cfg = feishu_cli._resolve_config(
        app_id="cli_xxxx",
        app_secret_path=str(secret_file),
        allow_from=["ou_alice", "ou_bob"],
        chat_id_prefix="feishu",
    )
    assert cfg.allow_from == ["ou_alice", "ou_bob"]


def test_env_path_does_not_leak_when_arg_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sanity: LARK_APP_ID from env is overridden by --app-id flag."""
    monkeypatch.setenv("LARK_APP_ID", "from_env")
    secret_file = tmp_path / "secret"
    secret_file.write_text("x")
    cfg = feishu_cli._resolve_config(
        app_id="explicit_arg",
        app_secret_path=str(secret_file),
        allow_from=None,
        chat_id_prefix="feishu",
    )
    assert cfg.app_id == "explicit_arg"
    # Belt-and-braces: env is still set (we test the override is at
    # resolve time, not via mutation of os.environ).
    assert os.environ["LARK_APP_ID"] == "from_env"
