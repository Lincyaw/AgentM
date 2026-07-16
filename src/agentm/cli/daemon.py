"""Local gateway lifecycle commands for ``agentm daemon``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from agentm.env import autoload_dotenv, resolve_cli_cwd
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
    help="Manage the local gateway daemon used by terminal clients.",
    no_args_is_help=True,
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


def _resolve_cwd(cwd: str | None) -> Path:
    return resolve_cli_cwd(cwd)


def _autoload_daemon_env(cwd: str | None) -> Path:
    resolved_cwd = _resolve_cwd(cwd)
    autoload_dotenv(resolved_cwd)
    return resolved_cwd


def _print_status(*, json_output: bool) -> None:
    status = gateway_daemon_status()
    if json_output:
        typer.echo(json.dumps(status.as_dict(), sort_keys=True))
        return
    state = "running" if status.running else "stopped"
    typer.echo(state)
    label = "socket" if status.connect_url.startswith("unix://") else "connect"
    typer.echo(f"{label}: {status.connect_url}")
    if status.auth_required and status.token_file is not None:
        typer.echo(f"token_file: {status.token_file}")
    elif status.connect_url.startswith(("ws://", "wss://")):
        typer.echo("auth: anonymous")
    typer.echo(f"reload: {'enabled' if status.reload else 'disabled'}")
    if status.pid is not None:
        typer.echo(f"pid: {status.pid}")
    typer.echo(f"log: {status.log_path}")


@app.command(name="socket")
def socket_cmd(
    cwd: Annotated[
        str | None,
        typer.Option("--cwd", help="Working directory used for .env discovery."),
    ] = None,
) -> None:
    """Print the active/default local daemon connect URL."""

    _autoload_daemon_env(cwd)
    status = gateway_daemon_status()
    if status.running:
        typer.echo(status.connect_url)
        return
    typer.echo(default_daemon_connect_url())


@app.command(name="status")
def status_cmd(
    cwd: Annotated[
        str | None,
        typer.Option("--cwd", help="Working directory used for .env discovery."),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print a machine-readable status object."),
    ] = False,
) -> None:
    """Show whether the local gateway daemon is running."""

    _autoload_daemon_env(cwd)
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
    bind: Annotated[
        str | None,
        typer.Option(
            "--bind",
            envvar="AGENTM_DAEMON_BIND",
            help=(
                "Gateway URL for the daemon worker. Default: per-user unix:// socket. "
                "Use ws://0.0.0.0:8765 for remote clients."
            ),
        ),
    ] = None,
    bind_token_file: Annotated[
        Path | None,
        typer.Option(
            "--bind-token-file",
            envvar="AGENTM_DAEMON_TOKEN_FILE",
            help=(
                "Token file for ws/wss daemon auth. Generated with mode 0600 "
                "when omitted."
            ),
        ),
    ] = None,
    bind_allow_anonymous: Annotated[
        bool,
        typer.Option(
            "--bind-allow-anonymous",
            help="Allow unauthenticated ws/wss daemon clients. Not recommended.",
        ),
    ] = False,
    tls_cert: Annotated[
        Path | None,
        typer.Option("--tls-cert", help="TLS certificate for wss:// daemon binds."),
    ] = None,
    tls_key: Annotated[
        Path | None,
        typer.Option("--tls-key", help="TLS private key for wss:// daemon binds."),
    ] = None,
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
    reload: Annotated[
        bool,
        typer.Option(
            "--reload/--no-reload",
            help=(
                "Enable supervisor source watching and worker restarts. "
                "Default: disabled."
            ),
        ),
    ] = False,
    startup_timeout: Annotated[
        float,
        typer.Option(
            "--startup-timeout",
            min=0.1,
            help="Seconds to wait for the daemon gateway endpoint.",
        ),
    ] = 10.0,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print a machine-readable status object."),
    ] = False,
) -> None:
    """Start or reuse the local gateway daemon."""

    resolved_cwd = _autoload_daemon_env(cwd)
    try:
        ensure_gateway_daemon(
            GatewayDaemonConfig(
                cwd=resolved_cwd,
                scenario=scenario,
                bind=bind,
                bind_token_file=bind_token_file,
                bind_allow_anonymous=bind_allow_anonymous,
                tls_cert=tls_cert,
                tls_key=tls_key,
                state_dir=state_dir,
                gateway_log=gateway_log,
                startup_timeout=startup_timeout,
                reload=reload,
            )
        )
    except GatewayDaemonError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=7) from exc

    _print_status(json_output=json_output)


@app.command(name="stop")
def stop_cmd(
    cwd: Annotated[
        str | None,
        typer.Option("--cwd", help="Working directory used for .env discovery."),
    ] = None,
    timeout: Annotated[
        float,
        typer.Option("--timeout", min=0.1, help="Seconds to wait before SIGKILL."),
    ] = 5.0,
) -> None:
    """Stop the local gateway daemon if it is running."""

    _autoload_daemon_env(cwd)
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
    bind: Annotated[
        str | None,
        typer.Option(
            "--bind",
            envvar="AGENTM_DAEMON_BIND",
            help=(
                "Gateway URL for the daemon worker. Default: per-user unix:// socket. "
                "Use ws://0.0.0.0:8765 for remote clients."
            ),
        ),
    ] = None,
    bind_token_file: Annotated[
        Path | None,
        typer.Option(
            "--bind-token-file",
            envvar="AGENTM_DAEMON_TOKEN_FILE",
            help=(
                "Token file for ws/wss daemon auth. Generated with mode 0600 "
                "when omitted."
            ),
        ),
    ] = None,
    bind_allow_anonymous: Annotated[
        bool,
        typer.Option(
            "--bind-allow-anonymous",
            help="Allow unauthenticated ws/wss daemon clients. Not recommended.",
        ),
    ] = False,
    tls_cert: Annotated[
        Path | None,
        typer.Option("--tls-cert", help="TLS certificate for wss:// daemon binds."),
    ] = None,
    tls_key: Annotated[
        Path | None,
        typer.Option("--tls-key", help="TLS private key for wss:// daemon binds."),
    ] = None,
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
    reload: Annotated[
        bool,
        typer.Option(
            "--reload/--no-reload",
            help=(
                "Enable supervisor source watching and worker restarts. "
                "Default: disabled."
            ),
        ),
    ] = False,
    startup_timeout: Annotated[
        float,
        typer.Option(
            "--startup-timeout",
            min=0.1,
            help="Seconds to wait for the daemon gateway endpoint.",
        ),
    ] = 10.0,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print a machine-readable status object."),
    ] = False,
) -> None:
    """Restart the local gateway daemon."""

    _autoload_daemon_env(cwd)
    stop_gateway_daemon()
    start_cmd(
        cwd=cwd,
        scenario=scenario,
        bind=bind,
        bind_token_file=bind_token_file,
        bind_allow_anonymous=bind_allow_anonymous,
        tls_cert=tls_cert,
        tls_key=tls_key,
        state_dir=state_dir,
        gateway_log=gateway_log,
        reload=reload,
        startup_timeout=startup_timeout,
        json_output=json_output,
    )
