"""Unit tests for CLI argument parsing."""

from __future__ import annotations

from typer.testing import CliRunner

from agentm.cli.main import app

runner = CliRunner()


def test_no_command_shows_help() -> None:
    """Bare `agentm` must show help, not crash."""
    result = runner.invoke(app, [])
    assert "run" in result.output
    assert "debug" in result.output


def test_run_help() -> None:
    """`agentm run --help` must list expected options."""
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "--data-dir" in result.output
    assert "--incident" in result.output
    assert "--debug" in result.output
    assert "--verbose" in result.output


def test_debug_help() -> None:
    """`agentm debug --help` must list expected options."""
    result = runner.invoke(app, ["debug", "--help"])
    assert result.exit_code == 0
    assert (
        "TRAJECTORY_FILE" in result.output.upper()
        or "trajectory" in result.output.lower()
    )
    assert "--summary" in result.output
    assert "--timeline" in result.output
    assert "--filter-agent" in result.output
    assert "--filter-type" in result.output


def test_run_requires_data_dir_and_incident() -> None:
    """`agentm run` without required options must fail with helpful error."""
    result = runner.invoke(app, ["run"])
    assert result.exit_code != 0


def test_debug_requires_trajectory_file() -> None:
    """`agentm debug` without a file must fail."""
    result = runner.invoke(app, ["debug"])
    assert result.exit_code != 0
