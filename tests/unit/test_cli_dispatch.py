"""CLI command-tree dispatch contract tests.

Fail-stop positions covered (the ``agentm`` console-script contract that
CLAUDE.md documents and downstream tooling / scripts rely on):

* Bare ``agentm`` shows help and exits 0 (not the prompt-required error) and
  lists every subcommand.
* An unknown subcommand exits 2.
* ``trace`` / ``gateway`` / ``list-extensions`` route to their commands.
* ``gateway`` keeps its FLAT ``agentm gateway [OPTIONS]`` surface — mounting
  its single-command Typer app via ``add_typer`` would silently turn it into
  ``agentm gateway COMMAND`` and break ``agentm gateway --bind ...``. This is
  the regression this refactor is most exposed to, so it is pinned here.

Driven through the real compiled command tree (Click ``CliRunner``) rather
than asserting private dispatch internals.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

import agentm.cli as cli


@pytest.fixture()
def root():
    return cli._build_command()


def test_help_lists_every_subcommand(root) -> None:
    result = CliRunner().invoke(root, ["--help"])
    assert result.exit_code == 0
    for name in ("list-extensions", "trace", "gateway"):
        assert name in result.output


def test_unknown_subcommand_exits_2(root) -> None:
    result = CliRunner().invoke(root, ["definitely-not-a-command"])
    assert result.exit_code == 2


def test_trace_routes_to_trace_app(root) -> None:
    # ``trace`` is a multi-command group: --help renders its COMMAND surface.
    result = CliRunner().invoke(root, ["trace", "--help"])
    assert result.exit_code == 0
    assert "COMMAND" in result.output


def test_gateway_stays_a_flat_single_command(root) -> None:
    # The flat surface is the contract: ``agentm gateway --bind ...`` must
    # parse against the gateway options directly, not require a nested verb.
    result = CliRunner().invoke(root, ["gateway", "--help"])
    assert result.exit_code == 0
    assert "gateway [OPTIONS]" in result.output
    assert "gateway [OPTIONS] COMMAND" not in result.output


def test_list_extensions_routes(root) -> None:
    result = CliRunner().invoke(root, ["list-extensions", "--source", "builtin"])
    assert result.exit_code == 0
    assert "# builtin" in result.output


def test_bare_invocation_shows_help_and_exits_zero(monkeypatch) -> None:
    # ``agentm`` with no args must show help (exit 0), not fall through to the
    # prompt-required error (exit 2).
    monkeypatch.setattr(cli.sys, "argv", ["agentm"])
    with pytest.raises(SystemExit) as excinfo:
        cli.main()
    assert excinfo.value.code == 0


def test_flag_without_prompt_exits_2(root) -> None:
    # A flag but no prompt/resume/continue is the prompt-required fail-stop.
    result = CliRunner().invoke(root, ["--quiet"])
    assert result.exit_code == 2
    assert "prompt is required" in result.output
