"""Focused tests for CLI command parsing."""

from __future__ import annotations

from typer.testing import CliRunner

from agentm.cli.main import app

runner = CliRunner()


def test_no_command_shows_help_or_usage() -> None:
    result = runner.invoke(app, [])
    assert result.exit_code != 0
    assert "debug" in result.output.lower()


def test_debug_help_lists_core_options() -> None:
    result = runner.invoke(app, ["debug", "--help"])
    assert result.exit_code == 0
    assert "--summary" in result.output
    assert "--timeline" in result.output
    assert "--filter-agent" in result.output


def test_debug_requires_trajectory_file() -> None:
    result = runner.invoke(app, ["debug"])
    assert result.exit_code != 0
