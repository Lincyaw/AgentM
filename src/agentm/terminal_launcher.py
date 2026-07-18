"""One-command launcher for the gateway-backed terminal peer."""

from __future__ import annotations

import os
import shlex
import shutil
import signal
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

from agentm.core.lib.paths import expand_path as _expand_path
from agentm.gateway import load_token_file
from agentm.gateway_daemon import (
    GatewayDaemonConfig,
    GatewayDaemonError,
    default_gateway_log,
    ensure_gateway_daemon,
    gateway_accepts_connections,
    gateway_daemon_status,
)


class TerminalLaunchError(RuntimeError):
    """Raised when the local gateway or terminal peer cannot be launched."""


DEFAULT_TERMINAL_BIN = "ag"


@dataclass(slots=True)
class TerminalLaunchConfig:
    """Resolved configuration for ``agentm terminal``."""

    cwd: Path
    connect: str | None = None
    scenario: str | None = None
    state_dir: Path | None = None
    terminal_bin: str = DEFAULT_TERMINAL_BIN
    terminal_bin_fallbacks: tuple[str, ...] = ("agentm-terminal",)
    terminal_log: Path | None = None
    session_id: str | None = None
    simple: bool = False
    theme: str | None = None
    gateway_log: Path | None = None
    gateway_command: str = "agentm"
    terminal_args: list[str] = field(default_factory=list)
    startup_timeout: float = 10.0
    use_daemon: bool = True
    reload: bool = False

    def __post_init__(self) -> None:
        self.cwd = _expand_path(self.cwd)
        for field_name in ("state_dir", "terminal_log", "gateway_log"):
            path = getattr(self, field_name)
            if path is not None:
                setattr(self, field_name, _expand_path(path))


@dataclass(slots=True)
class _GatewayProcess:
    process: subprocess.Popen[Any]
    log_file: IO[bytes]
    temp_dir: tempfile.TemporaryDirectory[str] | None
    connect_url: str
    log_path: Path


def run_terminal(config: TerminalLaunchConfig) -> int:
    """Run the terminal peer, ensuring a gateway is available when needed."""

    cwd = config.cwd
    if not cwd.is_dir():
        raise TerminalLaunchError(f"working directory does not exist: {cwd}")

    if config.connect:
        return _run_terminal_peer(config, connect_url=config.connect)

    if config.scenario:
        _validate_local_scenario(config.scenario)

    if config.use_daemon:
        connect_url = _ensure_daemon_gateway(config)
        return _run_terminal_peer(
            config,
            connect_url=connect_url,
            token_file=_daemon_token_file_for_connect(connect_url),
        )

    gateway = _start_gateway(config)
    try:
        connect_url = _gateway_connect_url(gateway)
        _wait_for_gateway(connect_url, gateway, config.startup_timeout)
        return _run_terminal_peer(config, connect_url=connect_url)
    finally:
        _stop_gateway(gateway)


def _start_gateway(config: TerminalLaunchConfig) -> _GatewayProcess:
    temp_dir = tempfile.TemporaryDirectory(prefix="agentm-gw-", dir="/tmp")
    bind_url = f"unix://{Path(temp_dir.name) / 'gateway.sock'}"
    log_path = config.gateway_log or default_gateway_log()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("ab")

    try:
        command = _split_command(config.gateway_command)
        executable = _find_executable(command[0])
        args = [
            executable,
            *command[1:],
            "gateway",
            "--cwd",
            str(config.cwd),
            "--bind",
            bind_url,
        ]
        if config.state_dir is not None:
            args.extend(["--state-dir", str(config.state_dir)])
        if config.scenario:
            args.extend(["--scenario", config.scenario])

        process = subprocess.Popen(
            args,
            cwd=str(config.cwd),
            env=os.environ.copy(),
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=log_file,
            start_new_session=True,
        )
    except Exception:
        log_file.close()
        temp_dir.cleanup()
        raise
    return _GatewayProcess(
        process=process,
        log_file=log_file,
        temp_dir=temp_dir,
        connect_url=bind_url,
        log_path=log_path,
    )


def _run_terminal_peer(
    config: TerminalLaunchConfig,
    *,
    connect_url: str,
    token_file: Path | None = None,
) -> int:
    executable = _find_terminal_executable(config)
    args = [executable, "--connect", connect_url]
    terminal_args = config.terminal_args
    if token_file is not None and not _has_token_arg(terminal_args):
        args.extend(["--token-file", str(token_file)])
    if config.session_id:
        args.extend(["--session-id", config.session_id])
    elif not _has_session_id_arg(terminal_args):
        args.extend(["--session-id", _default_terminal_session_id(config.cwd)])
    if config.scenario:
        args.extend(["--scenario", config.scenario])
    if config.simple:
        args.append("--simple")
    if config.theme:
        args.extend(["--theme", config.theme])
    if config.terminal_log is not None:
        args.extend(["--log", str(config.terminal_log)])
    args.extend(terminal_args)
    return subprocess.call(args, cwd=str(config.cwd), env=os.environ.copy())


def _ensure_daemon_gateway(config: TerminalLaunchConfig) -> str:
    try:
        return ensure_gateway_daemon(
            GatewayDaemonConfig(
                cwd=config.cwd,
                scenario=config.scenario,
                state_dir=config.state_dir,
                gateway_log=config.gateway_log,
                startup_timeout=config.startup_timeout,
                reload=config.reload,
            )
        )
    except GatewayDaemonError as exc:
        raise TerminalLaunchError(str(exc)) from exc


def _daemon_token_file_for_connect(connect_url: str) -> Path | None:
    status = gateway_daemon_status()
    if status.connect_url != connect_url or not status.auth_required:
        return None
    if status.token_file is None:
        raise TerminalLaunchError("daemon requires auth but has no token file")
    try:
        load_token_file(str(status.token_file))
    except ValueError as exc:
        raise TerminalLaunchError(str(exc)) from exc
    return status.token_file


def _validate_local_scenario(scenario: str) -> None:
    from agentm.extensions.loader import ScenarioLoadError, validate_scenario

    try:
        validate_scenario(scenario)
    except ScenarioLoadError as exc:
        raise TerminalLaunchError(f"--scenario {scenario!r}: {exc}") from exc


def _wait_for_gateway(
    connect_url: str,
    gateway: _GatewayProcess,
    timeout_seconds: float,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        rc = gateway.process.poll()
        if rc is not None:
            raise TerminalLaunchError(
                f"gateway exited before it was ready (exit {rc})."
                f"{_gateway_log_hint(gateway)}"
            )
        if gateway_accepts_connections(connect_url):
            return
        time.sleep(0.1)
    raise TerminalLaunchError(
        f"gateway socket was not ready after {timeout_seconds:.1f}s."
        f"{_gateway_log_hint(gateway)}"
    )


def _stop_gateway(gateway: _GatewayProcess) -> None:
    try:
        if gateway.process.poll() is None:
            _terminate_process_group(gateway.process, signal.SIGTERM)
            try:
                gateway.process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                _terminate_process_group(gateway.process, signal.SIGKILL)
                gateway.process.wait(timeout=5.0)
    finally:
        gateway.log_file.close()
        if gateway.temp_dir is not None:
            gateway.temp_dir.cleanup()


def _terminate_process_group(process: subprocess.Popen[Any], sig: signal.Signals) -> None:
    if os.name == "posix":
        try:
            os.killpg(process.pid, sig)
            return
        except ProcessLookupError:
            return
    if sig == signal.SIGTERM:
        process.terminate()
    else:
        process.kill()


def _gateway_connect_url(gateway: _GatewayProcess) -> str:
    if not gateway.connect_url:
        raise TerminalLaunchError("gateway connect URL was not initialized")
    return gateway.connect_url


def _gateway_log_path(gateway: _GatewayProcess) -> Path:
    return gateway.log_path


def _gateway_log_hint(gateway: _GatewayProcess) -> str:
    path = _gateway_log_path(gateway)
    try:
        content = path.read_bytes()
    except OSError:
        return f" Log: {path}"
    if not content:
        return f" Log: {path}"
    tail = content[-4000:].decode("utf-8", errors="replace").strip()
    return f" Log: {path}\n{tail}"


def _split_command(value: str) -> list[str]:
    parts = shlex.split(value)
    if not parts:
        raise TerminalLaunchError("gateway command is empty")
    return parts


def _find_executable(value: str) -> str:
    if os.sep in value or (os.altsep and os.altsep in value):
        path = _expand_path(value)
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
        raise TerminalLaunchError(
            f"executable not found or not executable: {value} ({path})"
        )
    found = shutil.which(value)
    if found:
        return found
    raise TerminalLaunchError(f"executable not found on PATH: {value}")


def _find_terminal_executable(config: TerminalLaunchConfig) -> str:
    try:
        return _find_executable(config.terminal_bin)
    except TerminalLaunchError as primary_error:
        # Legacy-binary fallbacks apply only when the caller did not override
        # the default binary name.
        if config.terminal_bin != DEFAULT_TERMINAL_BIN:
            raise
        for fallback in config.terminal_bin_fallbacks:
            try:
                return _find_executable(fallback)
            except TerminalLaunchError:
                continue
        fallback_text = ", ".join(config.terminal_bin_fallbacks)
        if fallback_text:
            raise TerminalLaunchError(
                f"{primary_error}; also tried fallback terminal binaries: "
                f"{fallback_text}"
            ) from primary_error
        raise


def _has_session_id_arg(args: list[str]) -> bool:
    for arg in args:
        if arg in {"-session-id", "--session-id"}:
            return True
        if arg.startswith("-session-id=") or arg.startswith("--session-id="):
            return True
    return False


def _has_token_arg(args: list[str]) -> bool:
    for arg in args:
        if arg in {"-token", "--token", "-token-file", "--token-file"}:
            return True
        if (
            arg.startswith("-token=")
            or arg.startswith("--token=")
            or arg.startswith("-token-file=")
            or arg.startswith("--token-file=")
        ):
            return True
    return False


def _default_terminal_session_id(cwd: Path) -> str:
    base = cwd.name or "terminal"
    return f"{base}-{uuid.uuid4().hex[:12]}"
