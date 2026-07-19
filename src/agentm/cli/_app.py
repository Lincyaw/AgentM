"""Root typer application — mounts all subcommands."""

from __future__ import annotations

import typer

from agentm.cli._chat import chat
from agentm.cli._run import run
from agentm.cli._config import config_app
from agentm.cli._scenario import scenario_app

app = typer.Typer(
    name="agentm",
    add_completion=False,
    no_args_is_help=True,
    help="AgentM SDK command-line interface.",
)

app.command("chat")(chat)
app.command("run")(run)
app.add_typer(config_app, name="config")
app.add_typer(scenario_app, name="scenario")

try:
    from agentm.extensions.builtin.policy.__main__ import app as _policy_app
    app.add_typer(_policy_app, name="policy", help="Policy engine queries and management.")
except ImportError:
    pass


def _version_callback(value: bool) -> None:
    if value:
        from importlib.metadata import version
        try:
            v = version("agentm")
        except Exception:
            v = "unknown"
        typer.echo(f"agentm {v}")
        raise typer.Exit()


@app.callback()
def _main(
    version: bool = typer.Option(
        False, "--version", "-V",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    pass


def main() -> None:
    app()
