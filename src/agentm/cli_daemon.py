"""``agentm daemon`` — local gateway daemon lifecycle commands."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated

import typer

from agentm.env import autoload_dotenv
from agentm.gateway_daemon import (
    GatewayDaemonConfig,
    GatewayDaemonError,
    default_daemon_connect_url,
    ensure_gateway_daemon,
    gateway_daemon_status,
    stop_gateway_daemon,
)


app = typer.Typer(
    name="daemon",
    help="Manage the local reloadable gateway daemon used by terminal clients.",
    no_args_is_help=True,
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


def _resolve_cwd(cwd: str | None) -> Path:
    return Path(cwd or os.environ.get("AGENTM_CWD") or os.getcwd())


def _print_status(*, json_output: bool) -> None:
    status = gateway_daemon_status()
    if json_output:
        typer.echo(json.dumps(status.as_dict(), sort_keys=True))
        return
    state = "running" if status.running else "stopped"
    typer.echo(state)
    typer.echo(f"socket: {status.connect_url}")
    if status.pid is not None:
        typer.echo(f"pid: {status.pid}")
    typer.echo(f"log: {status.log_path}")


@app.command(name="socket")
def socket_cmd() -> None:
    """Print the default local daemon socket URL."""

    typer.echo(default_daemon_connect_url())


@app.command(name="status")
def status_cmd(
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print a machine-readable status object."),
    ] = False,
) -> None:
    """Show whether the local gateway daemon is running."""

    _print_status(json_output=json_output)


@app.command(name="start")
def start_cmd(
    cwd: Annotated[
        str | None,
        typer.Option("--cwd", help="Working directory exposed to gateway sessions."),
    ] = None,
    scenario: Annotated[
        str | None,
        typer.Option(
            "--scenario",
            envvar="AGENTM_SCENARIO",
            help="Default scenario for new gateway sessions.",
        ),
    ] = "chatbot",
    state_dir: Annotated[
        Path | None,
        typer.Option(
            "--state-dir",
            envvar="AGENTM_STATE_DIR",
            help="Persistent gateway state dir. Default: $AGENTM_HOME/gateway.",
        ),
    ] = None,
    gateway_log: Annotated[
        Path | None,
        typer.Option(
            "--gateway-log",
            envvar="AGENTM_TERMINAL_GATEWAY_LOG",
            help="Gateway supervisor log file. Default: $AGENTM_HOME/logs/terminal-gateway.log.",
        ),
    ] = None,
    no_reload: Annotated[
        bool,
        typer.Option(
            "--no-reload",
            help="Disable supervisor source watching and worker restarts.",
        ),
    ] = False,
    startup_timeout: Annotated[
        float,
        typer.Option(
            "--startup-timeout",
            min=0.1,
            help="Seconds to wait for the daemon socket.",
        ),
    ] = 10.0,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print a machine-readable status object."),
    ] = False,
) -> None:
    """Start or reuse the local gateway daemon."""

    resolved_cwd = _resolve_cwd(cwd)
    autoload_dotenv(resolved_cwd)
    try:
        ensure_gateway_daemon(
            GatewayDaemonConfig(
                cwd=resolved_cwd,
                scenario=scenario,
                state_dir=state_dir,
                gateway_log=gateway_log,
                startup_timeout=startup_timeout,
                reload=not no_reload,
            )
        )
    except GatewayDaemonError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=7) from exc

    _print_status(json_output=json_output)


@app.command(name="stop")
def stop_cmd(
    timeout: Annotated[
        float,
        typer.Option("--timeout", min=0.1, help="Seconds to wait before SIGKILL."),
    ] = 5.0,
) -> None:
    """Stop the local gateway daemon if it is running."""

    stopped = stop_gateway_daemon(timeout_seconds=timeout)
    if stopped:
        typer.echo("stopped")
    else:
        typer.echo("already stopped")


@app.command(name="restart")
def restart_cmd(
    cwd: Annotated[
        str | None,
        typer.Option("--cwd", help="Working directory exposed to gateway sessions."),
    ] = None,
    scenario: Annotated[
        str | None,
        typer.Option(
            "--scenario",
            envvar="AGENTM_SCENARIO",
            help="Default scenario for new gateway sessions.",
        ),
    ] = "chatbot",
    state_dir: Annotated[
        Path | None,
        typer.Option(
            "--state-dir",
            envvar="AGENTM_STATE_DIR",
            help="Persistent gateway state dir. Default: $AGENTM_HOME/gateway.",
        ),
    ] = None,
    gateway_log: Annotated[
        Path | None,
        typer.Option(
            "--gateway-log",
            envvar="AGENTM_TERMINAL_GATEWAY_LOG",
            help="Gateway supervisor log file. Default: $AGENTM_HOME/logs/terminal-gateway.log.",
        ),
    ] = None,
    no_reload: Annotated[
        bool,
        typer.Option(
            "--no-reload",
            help="Disable supervisor source watching and worker restarts.",
        ),
    ] = False,
    startup_timeout: Annotated[
        float,
        typer.Option(
            "--startup-timeout",
            min=0.1,
            help="Seconds to wait for the daemon socket.",
        ),
    ] = 10.0,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print a machine-readable status object."),
    ] = False,
) -> None:
    """Restart the local gateway daemon."""

    stop_gateway_daemon()
    start_cmd(
        cwd=cwd,
        scenario=scenario,
        state_dir=state_dir,
        gateway_log=gateway_log,
        no_reload=no_reload,
        startup_timeout=startup_timeout,
        json_output=json_output,
    )
