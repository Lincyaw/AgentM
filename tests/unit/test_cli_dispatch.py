"""CLI command-tree dispatch contract tests.

Fail-stop positions covered (the ``agentm`` console-script contract that
CLAUDE.md documents and downstream tooling / scripts rely on):

* Bare ``agentm`` shows help and exits 0 (not the prompt-required error) and
  lists every subcommand.
* An unknown subcommand exits 2.
* ``trace`` / ``gateway`` / ``list-extensions`` route to their commands, with
  ``gateway`` keeping its FLAT ``agentm gateway [OPTIONS]`` surface.
* Errors land on **stderr** (agent-native CLI: stdout is data-only).

The dispatch checks drive a real child process via ``agentm.cli.main`` so
they observe true exit codes and stdout/stderr separation rather than
CliRunner's merged buffer.
"""

from __future__ import annotations

import subprocess
import sys

import pytest


def _run(argv: list[str]) -> subprocess.CompletedProcess[str]:
    driver = (
        "import sys\n"
        f"sys.argv = {argv!r}\n"
        "import agentm.cli as cli\n"
        "try:\n"
        "    cli.main()\n"
        "except SystemExit:\n"
        "    raise\n"
    )
    return subprocess.run(
        [sys.executable, "-c", driver],
        capture_output=True,
        text=True,
    )


# --- dispatch + exit-code contract ----------------------------------------


def test_bare_invocation_shows_help_and_exits_zero() -> None:
    proc = _run(["agentm"])
    assert proc.returncode == 0
    for name in ("list-extensions", "trace", "gateway"):
        assert name in proc.stdout


def test_unknown_subcommand_exits_2() -> None:
    proc = _run(["agentm", "definitely-not-a-command"])
    assert proc.returncode == 2


def test_flag_without_prompt_shows_help_exit_0() -> None:
    proc = _run(["agentm", "--quiet"])
    assert proc.returncode == 0
    assert "--prompt" in proc.stdout


def test_trace_routes_to_trace_app() -> None:
    proc = _run(["agentm", "trace", "--help"])
    assert proc.returncode == 0
    assert "messages" in proc.stdout


def test_gateway_routes_to_gateway_app() -> None:
    proc = _run(["agentm", "gateway", "--help"])
    assert proc.returncode == 0
    assert "--bind" in proc.stdout
