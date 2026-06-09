"""Tests for per-channel workspace resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentm.gateway.workspace import WorkspaceResolver, load_gateway_config


class TestWorkspaceResolver:
    def test_no_root_returns_default(self) -> None:
        r = WorkspaceResolver("/default")
        assert r.resolve("any") == "/default"
        assert r.resolve("") == "/default"
        assert not r.active

    def test_root_resolves_by_channel(self, tmp_path: Path) -> None:
        r = WorkspaceResolver("/default", workspace_root=str(tmp_path))
        result = r.resolve("mychannel")
        assert result == str(tmp_path / "mychannel")
        assert (tmp_path / "mychannel").is_dir()
        assert r.active

    def test_override_wins_over_root(self, tmp_path: Path) -> None:
        override_dir = tmp_path / "custom"
        r = WorkspaceResolver(
            "/default",
            workspace_root=str(tmp_path / "root"),
            overrides={"special": str(override_dir)},
        )
        result = r.resolve("special")
        assert result == str(override_dir.resolve())
        assert override_dir.is_dir()

    def test_empty_channel_returns_default(self, tmp_path: Path) -> None:
        r = WorkspaceResolver("/default", workspace_root=str(tmp_path))
        assert r.resolve("") == "/default"

    def test_directory_created_only_once(self, tmp_path: Path) -> None:
        r = WorkspaceResolver("/default", workspace_root=str(tmp_path))
        r.resolve("ch")
        r.resolve("ch")
        assert (tmp_path / "ch").is_dir()

    def test_existing_directory_not_recreated(self, tmp_path: Path) -> None:
        (tmp_path / "existing").mkdir()
        marker = tmp_path / "existing" / "marker.txt"
        marker.write_text("keep")
        r = WorkspaceResolver("/default", workspace_root=str(tmp_path))
        r.resolve("existing")
        assert marker.read_text() == "keep"


class TestLoadGatewayConfig:
    def test_missing_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AGENTM_HOME", str(tmp_path))
        r = load_gateway_config("/default")
        assert not r.active
        assert r.resolve("x") == "/default"

    def test_valid_config(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AGENTM_HOME", str(tmp_path))
        config = tmp_path / "config.toml"
        config.write_text(
            "[gateway]\n"
            f'workspace_root = "{tmp_path / "ws"}"\n'
            "\n"
            "[gateway.workspaces]\n"
            f'special = "{tmp_path / "custom"}"\n'
        )
        r = load_gateway_config("/default")
        assert r.active
        assert r.resolve("normal") == str((tmp_path / "ws" / "normal").resolve())
        assert r.resolve("special") == str((tmp_path / "custom").resolve())

    def test_no_gateway_section(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AGENTM_HOME", str(tmp_path))
        (tmp_path / "config.toml").write_text('[models]\nfoo = "bar"\n')
        r = load_gateway_config("/default")
        assert not r.active
