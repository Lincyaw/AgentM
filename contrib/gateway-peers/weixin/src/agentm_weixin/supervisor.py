"""Process supervisor using supervisord.

``agentm-weixin serve`` generates a supervisord.conf on the fly and
launches supervisord in foreground (``nodaemon=true``).  Two programs:

  gateway       agentm gateway --bind unix://…
  weixin        agentm-weixin run --connect unix://…

Both auto-restart on crash. The adapter has built-in reconnect logic so
it tolerates the gateway coming up a few seconds later.

supervisord gives us for free:
  - Restart policy with backoff
  - Log capture + rotation
  - ``supervisorctl`` for live management (status / stop / restart)
  - Clean process-group shutdown on SIGINT/SIGTERM
"""

from __future__ import annotations

import os
import shutil
import sys
import textwrap
from pathlib import Path

from agentm.core.lib import agentm_home_dir
from loguru import logger


def _state_dir() -> Path:
    return agentm_home_dir() / "weixin"


def _expand_path(path: str) -> Path:
    return Path(os.path.expandvars(path)).expanduser()


def _log_dir() -> Path:
    d = _state_dir() / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _run_dir() -> Path:
    runtime = os.environ.get("XDG_RUNTIME_DIR", "")
    if runtime:
        d = _expand_path(runtime) / "agentm-weixin"
    else:
        d = _state_dir() / "run"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _find_executable(name: str) -> str:
    """Find executable in PATH or the current venv."""
    # Check venv bin first (uv-installed scripts)
    venv_bin = Path(sys.prefix) / "bin" / name
    if venv_bin.exists():
        return str(venv_bin)
    found = shutil.which(name)
    if found:
        return found
    return name


def generate_config(
    *,
    bind_url: str,
    model: str | None = None,
    gateway_scenario: str | None = None,
    gateway_extra_args: list[str] | None = None,
    account_id: str,
    channel_name: str = "weixin",
    adapter_scenario: str | None = "chatbot",
    session_scope: str = "user",
) -> str:
    """Generate a supervisord.conf string for the gateway + adapter pair."""

    log_dir = _log_dir()
    run_dir = _run_dir()

    # Build gateway command
    agentm_bin = _find_executable("agentm")
    gw_cmd_parts = [agentm_bin, "gateway", "--bind", bind_url]
    if model:
        gw_cmd_parts.extend(["--model", model])
    if gateway_scenario:
        gw_cmd_parts.extend(["--scenario", gateway_scenario])
    if gateway_extra_args:
        gw_cmd_parts.extend(gateway_extra_args)
    gw_cmd = " ".join(gw_cmd_parts)

    # Build adapter command
    weixin_bin = _find_executable("agentm-weixin")
    adapter_cmd_parts = [
        weixin_bin, "run",
        "--connect", bind_url,
        "--account-id", account_id,
        "--channel-name", channel_name,
        "--session-scope", session_scope,
        "-v",
    ]
    if adapter_scenario:
        adapter_cmd_parts.extend(["--scenario", adapter_scenario])
    adapter_cmd = " ".join(adapter_cmd_parts)

    return textwrap.dedent(f"""\
        [supervisord]
        nodaemon=true
        logfile={log_dir}/supervisord.log
        logfile_maxbytes=10MB
        logfile_backups=3
        pidfile={run_dir}/supervisord.pid
        childlogdir={log_dir}

        [unix_http_server]
        file={run_dir}/supervisor.sock

        [rpcinterface:supervisor]
        supervisor.rpcinterface_factory=supervisor.rpcinterface:make_main_rpcinterface

        [supervisorctl]
        serverurl=unix://{run_dir}/supervisor.sock

        [program:gateway]
        command={gw_cmd}
        autorestart=true
        startretries=999
        startsecs=2
        redirect_stderr=true
        stdout_logfile={log_dir}/gateway.log
        stdout_logfile_maxbytes=10MB
        stdout_logfile_backups=3
        priority=10

        [program:weixin]
        command={adapter_cmd}
        autorestart=true
        startretries=999
        startsecs=3
        redirect_stderr=true
        stdout_logfile={log_dir}/weixin.log
        stdout_logfile_maxbytes=10MB
        stdout_logfile_backups=3
        priority=20
    """)


def write_config(config_text: str) -> str:
    """Write config to the state dir, return the path."""
    path = _state_dir() / "supervisord.conf"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(config_text, "utf-8")
    return str(path)


def launch_supervisord(config_path: str) -> int:
    """Launch supervisord in foreground. Returns its exit code."""
    supervisord_bin = _find_executable("supervisord")
    logger.info(f"[supervisor] config: {config_path}")
    logger.info(f"[supervisor] logs:   {_log_dir()}")
    logger.info(f"[supervisor] ctl:    supervisorctl -c {config_path} status")

    os.execvp(supervisord_bin, [supervisord_bin, "-c", config_path])
    return 1  # unreachable, execvp replaces the process


def supervisorctl_cmd(config_path: str) -> str:
    """Return the supervisorctl command users can run."""
    return f"supervisorctl -c {config_path} status"
