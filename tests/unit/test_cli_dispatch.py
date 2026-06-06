"""CLI command-tree dispatch contract tests.

Fail-stop positions covered (the ``agentm`` console-script contract that
CLAUDE.md documents and downstream tooling / scripts rely on):

* Bare ``agentm`` shows help and exits 0 (not the prompt-required error) and
  lists every subcommand.
* An unknown subcommand exits 2.
* ``trace`` / ``gateway`` / ``list-extensions`` route to their commands, with
  ``gateway`` keeping its FLAT ``agentm gateway [OPTIONS]`` surface (mounting
  its single-command Typer app as a group would break ``agentm gateway
  --bind ...``).
* Errors land on **stderr** (agent-native CLI: stdout is data-only).
* The prompt hot path and ``agentm --help`` do NOT import the ``trace`` /
  ``gateway`` dependency closures (gateway pulls in the websockets server).
  This is the load-bearing property the lazy dispatch exists to deliver.

The dispatch / import-isolation / stream-discipline checks drive a real child
process via ``agentm.cli.main`` so they observe true ``sys.modules``, exit
codes, and stdout/stderr separation rather than CliRunner's merged buffer.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

import agentm.cli as cli

# A child process that sets argv, runs ``main()``, and on exit reports whether
# the heavy subcommand modules were imported. ``{argv}`` is substituted with a
# Python list literal.
_DRIVER = """
import sys, atexit
sys.argv = {argv}
import agentm.cli as cli

def _report():
    sys.stderr.write(
        "IMPORTS"
        " gateway=%s" % ("agentm.gateway.cli" in sys.modules)
        + " trace=%s" % ("agentm.cli_trace" in sys.modules)
        + " websockets=%s\\n" % any(k.split(".")[0] == "websockets" for k in sys.modules)
    )

atexit.register(_report)
try:
    cli.main()
except SystemExit as exc:
    raise
"""


def _run(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", _DRIVER.format(argv=argv)],
        capture_output=True,
        text=True,
    )


def _imports(stderr: str) -> dict[str, bool]:
    line = next(ln for ln in stderr.splitlines() if ln.startswith("IMPORTS"))
    return {
        kv.split("=")[0]: kv.split("=")[1] == "True"
        for kv in line.split()[1:]
    }


# --- dispatch + exit-code contract ----------------------------------------


def test_bare_invocation_shows_help_and_exits_zero() -> None:
    proc = _run(["agentm"])
    assert proc.returncode == 0
    for name in ("list-extensions", "trace", "gateway"):
        assert name in proc.stdout


def test_unknown_subcommand_exits_2() -> None:
    proc = _run(["agentm", "definitely-not-a-command"])
    assert proc.returncode == 2


def test_flag_without_prompt_errors_on_stderr_exit_2() -> None:
    # A flag but no prompt/resume/continue is the prompt-required fail-stop.
    # The error MUST be on stderr (stdout stays data-only) and exit 2.
    proc = _run(["agentm", "--quiet"])
    assert proc.returncode == 2
    assert "prompt is required" in proc.stderr
    assert "prompt is required" not in proc.stdout


def test_trace_routes_to_trace_app() -> None:
    # ``trace`` is a multi-command group: --help renders its COMMAND surface.
    proc = _run(["agentm", "trace", "--help"])
    assert proc.returncode == 0
    assert "COMMAND" in proc.stdout


def test_gateway_stays_a_flat_single_command() -> None:
    # The flat surface is the contract: ``agentm gateway --bind ...`` must
    # parse against the gateway options directly, not require a nested verb.
    proc = _run(["agentm", "gateway", "--help"])
    assert proc.returncode == 0
    assert "gateway [OPTIONS]" in proc.stdout
    assert "gateway [OPTIONS] COMMAND" not in proc.stdout


# --- lazy-import contract (the reason the dispatcher exists) ----------------


@pytest.mark.parametrize("argv", [["agentm", "--help"], ["agentm", "--quiet"]])
def test_prompt_and_help_paths_do_not_import_subcommand_closures(argv) -> None:
    imports = _imports(_run(argv).stderr)
    assert imports == {"gateway": False, "trace": False, "websockets": False}


def test_gateway_dispatch_does_not_import_trace() -> None:
    imports = _imports(_run(["agentm", "gateway", "--help"]).stderr)
    assert imports["gateway"] is True
    assert imports["trace"] is False


def test_trace_dispatch_does_not_import_gateway() -> None:
    imports = _imports(_run(["agentm", "trace", "--help"]).stderr)
    assert imports["trace"] is True
    assert imports["gateway"] is False
    assert imports["websockets"] is False


# --- in-process build sanity (no subprocess) -------------------------------


def test_build_command_is_idempotent() -> None:
    # Building twice must not double-register placeholders / mutate shared
    # module state in a way that changes the command set.
    first = sorted(cli._build_command().commands)
    second = sorted(cli._build_command().commands)
    assert first == second
    assert {"list-extensions", "trace", "gateway"} <= set(first)
