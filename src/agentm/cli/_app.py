"""Root typer application — mounts all subcommands."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

import typer

from agentm.cli._chat import chat
from agentm.cli._run import run
from agentm.cli._config import config_app
from agentm.cli._scenario import scenario_app
from agentm.cli._session import session_app
from agentm.cli._trace import trace_app
from agentm.code_health import app as lint_app

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
app.add_typer(session_app, name="session")
app.add_typer(trace_app, name="trace")
app.add_typer(trace_app, name="t", hidden=True)
app.add_typer(lint_app, name="lint")


def _version_callback(value: bool) -> None:
    if value:
        try:
            v = version("agentm")
        except PackageNotFoundError:
            v = "unknown"
        typer.echo(f"agentm {v}")
        raise typer.Exit()


@app.callback()
def _main(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    pass


def main() -> None:
    app()
