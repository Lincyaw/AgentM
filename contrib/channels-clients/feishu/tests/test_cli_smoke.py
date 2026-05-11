"""argparse-level smoke tests for ``agentm-feishu``.

No subprocess, no real Feishu, no real socket. ``--check-config`` is
used to drive the happy path through argument parsing and config
resolution without contacting the gateway or Feishu.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agentm_feishu import cli as feishu_cli


@pytest.fixture(autouse=True)
def _scrub_lark_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every test start from a clean LARK_* environment.

    Also disable the ``.env`` autoload that ``cli.main()`` runs before
    arg parsing — otherwise the repo's own ``.env`` would put LARK_* back
    after we scrub them.
    """
    monkeypatch.delenv("LARK_APP_ID", raising=False)
    monkeypatch.delenv("LARK_APP_SECRET", raising=False)
    monkeypatch.setattr(feishu_cli, "load_dotenv_files", lambda _cwd: None)


def test_help_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        feishu_cli._build_parser().parse_args(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "agentm-feishu" in out
    assert "--connect" in out
    assert "--app-id" in out
    assert "--app-secret" in out
    # Examples section is mandated by the brief.
    assert "Examples" in out


def test_version_prints_and_exits_zero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc:
        feishu_cli._build_parser().parse_args(["--version"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "0.1.0" in out


def test_missing_connect_exits_two(capsys: pytest.CaptureFixture[str]) -> None:
    rc = feishu_cli.main([])
    assert rc == 2


def test_bad_connect_scheme_exits_two(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = feishu_cli.main(
        [
            "--connect",
            "tcp://127.0.0.1:7000",
            "--app-id",
            "cli_xxxx",
            "--check-config",
        ]
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "agentm-feishu: error" in err
    assert "unix://" in err


def test_missing_app_id_exits_two(capsys: pytest.CaptureFixture[str]) -> None:
    rc = feishu_cli.main(
        ["--connect", "unix:///tmp/gw.sock", "--check-config"]
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "missing --app-id" in err


def test_missing_app_secret_exits_two(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = feishu_cli.main(
        [
            "--connect",
            "unix:///tmp/gw.sock",
            "--app-id",
            "cli_xxxx",
            "--check-config",
        ]
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "missing app secret" in err


def test_empty_secret_file_exits_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    secret_file = tmp_path / "secret"
    secret_file.write_text("")
    rc = feishu_cli.main(
        [
            "--connect",
            "unix:///tmp/gw.sock",
            "--app-id",
            "cli_xxxx",
            "--app-secret",
            str(secret_file),
            "--check-config",
        ]
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "empty" in err


def test_check_config_success_with_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LARK_APP_ID", "cli_xxxx")
    monkeypatch.setenv("LARK_APP_SECRET", "secret_value")
    rc = feishu_cli.main(
        ["--connect", "unix:///tmp/gw.sock", "--check-config"]
    )
    assert rc == 0


def test_check_config_success_with_file(tmp_path: Path) -> None:
    secret_file = tmp_path / "secret"
    secret_file.write_text("secret_value\n")
    rc = feishu_cli.main(
        [
            "--connect",
            "unix:///tmp/gw.sock",
            "--app-id",
            "cli_xxxx",
            "--app-secret",
            str(secret_file),
            "--check-config",
        ]
    )
    assert rc == 0


def test_secret_file_wins_over_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When both are set the file wins, per CLI contract."""
    monkeypatch.setenv("LARK_APP_SECRET", "from_env")
    secret_file = tmp_path / "secret"
    secret_file.write_text("from_file")
    cfg = feishu_cli._resolve_config(
        feishu_cli._build_parser().parse_args(
            [
                "--connect",
                "unix:///tmp/gw.sock",
                "--app-id",
                "cli_xxxx",
                "--app-secret",
                str(secret_file),
            ]
        )
    )
    assert cfg.app_secret == "from_file"


def test_bad_secret_file_path_exits_two(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = feishu_cli.main(
        [
            "--connect",
            "unix:///tmp/gw.sock",
            "--app-id",
            "cli_xxxx",
            "--app-secret",
            "/nonexistent/path/that/does/not/exist",
            "--check-config",
        ]
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "cannot read --app-secret file" in err


def test_default_allow_from_is_star(tmp_path: Path) -> None:
    secret_file = tmp_path / "secret"
    secret_file.write_text("x")
    args = feishu_cli._build_parser().parse_args(
        [
            "--connect",
            "unix:///tmp/gw.sock",
            "--app-id",
            "cli_xxxx",
            "--app-secret",
            str(secret_file),
        ]
    )
    cfg = feishu_cli._resolve_config(args)
    assert cfg.allow_from == ["*"]


def test_repeated_allow_from(tmp_path: Path) -> None:
    secret_file = tmp_path / "secret"
    secret_file.write_text("x")
    args = feishu_cli._build_parser().parse_args(
        [
            "--connect",
            "unix:///tmp/gw.sock",
            "--app-id",
            "cli_xxxx",
            "--app-secret",
            str(secret_file),
            "--allow-from",
            "ou_alice",
            "--allow-from",
            "ou_bob",
        ]
    )
    cfg = feishu_cli._resolve_config(args)
    assert cfg.allow_from == ["ou_alice", "ou_bob"]


def test_env_path_does_not_leak_when_arg_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sanity: LARK_APP_ID from env is overridden by --app-id flag."""
    monkeypatch.setenv("LARK_APP_ID", "from_env")
    secret_file = tmp_path / "secret"
    secret_file.write_text("x")
    args = feishu_cli._build_parser().parse_args(
        [
            "--connect",
            "unix:///tmp/gw.sock",
            "--app-id",
            "explicit_arg",
            "--app-secret",
            str(secret_file),
        ]
    )
    cfg = feishu_cli._resolve_config(args)
    assert cfg.app_id == "explicit_arg"
    # Belt-and-braces: env is gone for the rest of the test.
    assert os.environ["LARK_APP_ID"] == "from_env"
