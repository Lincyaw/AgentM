"""Tests for the ``[feishu.bots.*]`` config.toml parser.

Fail-stop property: a valid multi-bot config produces the right number of
FeishuConfig objects with correct field mapping; an invalid entry (missing
app_id, missing secret) is silently skipped rather than crashing the process.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agentm_feishu.bot_config import load_bot_configs


def _write_config(tmp_path: Path, content: str) -> None:
    """Write a config.toml and point AGENTM_HOME at it."""
    (tmp_path / "config.toml").write_text(content, encoding="utf-8")


def test_valid_two_bots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTM_HOME", str(tmp_path))
    _write_config(
        tmp_path,
        """\
[feishu.bots.ops]
app_id = "cli_ops"
app_secret = "secret_ops"
scenario = "rca"
channel_name = "feishu-ops"
session_scope = "user"
allow_from = ["ou_admin", "ou_dev"]

[feishu.bots.chat]
app_id = "cli_chat"
app_secret = "secret_chat"
scenario = "chatbot"
""",
    )
    bots = load_bot_configs()
    assert len(bots) == 2

    names = [n for n, _ in bots]
    assert "ops" in names
    assert "chat" in names

    ops_cfg = next(c for n, c in bots if n == "ops")
    assert ops_cfg.app_id == "cli_ops"
    assert ops_cfg.app_secret == "secret_ops"
    assert ops_cfg.scenario == "rca"
    assert ops_cfg.channel_name == "feishu-ops"
    assert ops_cfg.session_scope == "user"
    assert ops_cfg.allow_from == ["ou_admin", "ou_dev"]

    chat_cfg = next(c for n, c in bots if n == "chat")
    assert chat_cfg.app_id == "cli_chat"
    assert chat_cfg.channel_name == "chat"  # defaults to bot key name
    assert chat_cfg.scenario == "chatbot"
    assert chat_cfg.session_scope == "chat"
    assert chat_cfg.allow_from == ["*"]


def test_missing_app_id_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENTM_HOME", str(tmp_path))
    _write_config(
        tmp_path,
        """\
[feishu.bots.bad]
app_secret = "s"

[feishu.bots.good]
app_id = "cli_ok"
app_secret = "s2"
""",
    )
    bots = load_bot_configs()
    assert len(bots) == 1
    assert bots[0][0] == "good"


def test_missing_secret_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENTM_HOME", str(tmp_path))
    # No app_secret, no app_secret_file, no env var
    monkeypatch.delenv("LARK_APP_SECRET_NOSECRET", raising=False)
    _write_config(
        tmp_path,
        """\
[feishu.bots.nosecret]
app_id = "cli_x"
""",
    )
    bots = load_bot_configs()
    assert bots == []


def test_secret_from_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENTM_HOME", str(tmp_path))
    secret_file = tmp_path / "my_secret"
    secret_file.write_text("  file_secret_value  \n", encoding="utf-8")
    _write_config(
        tmp_path,
        f"""\
[feishu.bots.fromfile]
app_id = "cli_f"
app_secret_file = "{secret_file}"
""",
    )
    bots = load_bot_configs()
    assert len(bots) == 1
    assert bots[0][1].app_secret == "file_secret_value"


def test_secret_from_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENTM_HOME", str(tmp_path))
    monkeypatch.setenv("LARK_APP_SECRET_ENVBOT", "env_secret_val")
    _write_config(
        tmp_path,
        """\
[feishu.bots.envbot]
app_id = "cli_e"
""",
    )
    bots = load_bot_configs()
    assert len(bots) == 1
    assert bots[0][1].app_secret == "env_secret_val"


def test_channel_name_defaults_to_bot_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENTM_HOME", str(tmp_path))
    _write_config(
        tmp_path,
        """\
[feishu.bots.mybot]
app_id = "cli_m"
app_secret = "s"
""",
    )
    bots = load_bot_configs()
    assert bots[0][1].channel_name == "mybot"


def test_empty_feishu_section(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENTM_HOME", str(tmp_path))
    _write_config(tmp_path, "[feishu]\n")
    assert load_bot_configs() == []


def test_no_config_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENTM_HOME", str(tmp_path))
    # No config.toml written
    assert load_bot_configs() == []


def test_invalid_session_scope_defaults_to_chat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENTM_HOME", str(tmp_path))
    _write_config(
        tmp_path,
        """\
[feishu.bots.b]
app_id = "cli_b"
app_secret = "s"
session_scope = "bogus"
""",
    )
    bots = load_bot_configs()
    assert len(bots) == 1
    assert bots[0][1].session_scope == "chat"


def test_secret_file_precedence_over_direct(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """app_secret_file wins over app_secret when both are present."""
    monkeypatch.setenv("AGENTM_HOME", str(tmp_path))
    secret_file = tmp_path / "secret_f"
    secret_file.write_text("from_file", encoding="utf-8")
    _write_config(
        tmp_path,
        f"""\
[feishu.bots.both]
app_id = "cli_both"
app_secret = "from_direct"
app_secret_file = "{secret_file}"
""",
    )
    bots = load_bot_configs()
    assert bots[0][1].app_secret == "from_file"
