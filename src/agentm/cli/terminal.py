"""Registration for the root ``agentm terminal`` command."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import typer

from agentm.env import autoload_dotenv, resolve_cli_cwd

CwdOpt = Annotated[
    str | None,
    typer.Option(
        "--cwd",
        help=(
            "Working directory exposed to extensions. Defaults to "
            "AGENTM_CWD when set, otherwise the process cwd at invocation time."
        ),
    ),
]


def register_terminal_command(
    app: typer.Typer,
    *,
    default_scenario: str,
) -> None:
    @app.command(
        name="terminal",
        context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    )
    def terminal_cmd(
        ctx: typer.Context,
        connect: Annotated[
            str | None,
            typer.Option(
                "--connect",
                help=(
                    "Existing gateway URL. Omit this to start or reuse the local "
                    "gateway daemon."
                ),
            ),
        ] = None,
        scenario: Annotated[
            str | None,
            typer.Option(
                "--scenario",
                help=(
                    "Scenario for the local gateway daemon or the first terminal "
                    "message. Defaults to chatbot in one-command mode."
                ),
            ),
        ] = None,
        cwd: CwdOpt = None,
        state_dir: Annotated[
            Path | None,
            typer.Option(
                "--state-dir",
                envvar="AGENTM_STATE_DIR",
                help=(
                    "Persistent gateway state dir. Only used when starting a "
                    "local gateway."
                ),
            ),
        ] = None,
        terminal_bin: Annotated[
            str,
            typer.Option(
                "--terminal-bin",
                envvar="AGENTM_TERMINAL_BIN",
                help="Terminal peer executable.",
            ),
        ] = "ag",
        terminal_log: Annotated[
            Path | None,
            typer.Option(
                "--terminal-log",
                help=(
                    "Terminal peer log file. Extra TUI flags may also be passed "
                    "after --."
                ),
            ),
        ] = None,
        session_id: Annotated[
            str | None,
            typer.Option(
                "--session-id",
                help=(
                    "Reconnect to a known terminal session id. Default: new "
                    "session per terminal."
                ),
            ),
        ] = None,
        simple: Annotated[
            bool,
            typer.Option(
                "--simple",
                help="Use the compact terminal layout.",
            ),
        ] = False,
        theme: Annotated[
            str | None,
            typer.Option(
                "--theme",
                help="Terminal theme: dark or light.",
            ),
        ] = None,
        gateway_log: Annotated[
            Path | None,
            typer.Option(
                "--gateway-log",
                envvar="AGENTM_TERMINAL_GATEWAY_LOG",
                help=(
                    "Gateway daemon/supervisor log file. Default: "
                    "$AGENTM_HOME/logs/terminal-gateway.log."
                ),
            ),
        ] = None,
        private_gateway: Annotated[
            bool,
            typer.Option(
                "--private-gateway",
                help=(
                    "Start a gateway only for this terminal and stop it when "
                    "the TUI exits."
                ),
            ),
        ] = False,
        reload: Annotated[
            bool,
            typer.Option(
                "--reload/--no-reload",
                help=(
                    "Enable daemon supervisor source watching and worker restarts. "
                    "Default: disabled."
                ),
            ),
        ] = False,
        gateway_command: Annotated[
            str,
            typer.Option(
                "--gateway-command",
                help="Command used to launch a --private-gateway worker.",
                hidden=True,
            ),
        ] = "agentm",
        startup_timeout: Annotated[
            float,
            typer.Option(
                "--startup-timeout",
                min=0.1,
                help="Seconds to wait for the local gateway endpoint.",
            ),
        ] = 10.0,
    ) -> None:
        """Open the terminal UI, starting/reusing the local gateway daemon by default."""

        from agentm.terminal_launcher import (
            TerminalLaunchConfig,
            TerminalLaunchError,
            run_terminal,
        )

        resolved_cwd = resolve_cli_cwd(cwd)
        autoload_dotenv(resolved_cwd)
        resolved_scenario = (
            scenario if scenario is not None else os.environ.get("AGENTM_SCENARIO")
        )
        if not connect and resolved_scenario is None:
            resolved_scenario = default_scenario

        if connect and (
            state_dir is not None
            or gateway_log is not None
            or private_gateway
            or reload
            or gateway_command != "agentm"
        ):
            raise typer.BadParameter(
                "--state-dir, --gateway-log, --private-gateway, --reload, and "
                "--gateway-command only apply when agentm terminal starts the "
                "local gateway"
            )

        try:
            rc = run_terminal(
                TerminalLaunchConfig(
                    cwd=resolved_cwd,
                    connect=connect,
                    scenario=resolved_scenario,
                    state_dir=state_dir,
                    terminal_bin=terminal_bin,
                    terminal_log=terminal_log,
                    session_id=session_id,
                    simple=simple,
                    theme=theme,
                    gateway_log=gateway_log,
                    gateway_command=gateway_command,
                    terminal_args=list(ctx.args),
                    startup_timeout=startup_timeout,
                    use_daemon=not private_gateway,
                    reload=reload,
                )
            )
        except TerminalLaunchError as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(code=7) from exc
        except KeyboardInterrupt:
            raise typer.Exit(code=130) from None
        raise typer.Exit(code=rc)
