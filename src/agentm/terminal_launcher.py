"""One-command launcher for the gateway-backed terminal peer."""

from __future__ import annotations

import os
import hashlib
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any


class TerminalLaunchError(RuntimeError):
    """Raised when the local gateway or terminal peer cannot be launched."""


@dataclass(slots=True)
class TerminalLaunchConfig:
    """Resolved configuration for ``agentm terminal``."""

    cwd: Path
    connect: str | None = None
    scenario: str | None = None
    state_dir: Path | None = None
    terminal_bin: str = "agentm-terminal"
    terminal_log: Path | None = None
    gateway_log: Path | None = None
    gateway_command: str = "agentm"
    terminal_args: list[str] = field(default_factory=list)
    startup_timeout: float = 10.0
    use_daemon: bool = True
    reload: bool = True


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

    if config.use_daemon:
        connect_url = _ensure_daemon_gateway(config)
        return _run_terminal_peer(config, connect_url=connect_url)

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
    log_path = config.gateway_log or _default_gateway_log()
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


def _run_terminal_peer(config: TerminalLaunchConfig, *, connect_url: str) -> int:
    executable = _find_executable(config.terminal_bin)
    args = [executable, "--connect", connect_url]
    if config.scenario:
        args.extend(["--scenario", config.scenario])
    if config.terminal_log is not None:
        args.extend(["--log", str(config.terminal_log)])
    args.extend(config.terminal_args)
    return subprocess.call(args, cwd=str(config.cwd), env=os.environ.copy())


def _ensure_daemon_gateway(config: TerminalLaunchConfig) -> str:
    connect_url = _default_daemon_connect_url()
    if _gateway_accepts_connections(connect_url):
        return connect_url

    runtime_dir = _daemon_runtime_dir()
    pid_file = runtime_dir / "gateway-supervisor.pid"
    if _pid_file_is_live(pid_file):
        _wait_for_gateway(connect_url, _pid_only_gateway(pid_file), config.startup_timeout)
        return connect_url

    log_path = config.gateway_log or _default_gateway_log()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("ab")
    args = [
        sys.executable,
        "-m",
        "agentm.gateway_supervisor",
        "--cwd",
        str(config.cwd),
        "--bind",
        connect_url,
        "--state-dir",
        str(config.state_dir or _default_daemon_state_dir()),
        "--pid-file",
        str(pid_file),
    ]
    if config.scenario:
        args.extend(["--scenario", config.scenario])
    if not config.reload:
        args.append("--no-reload")
    try:
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
        raise
    log_file.close()
    gateway = _GatewayProcess(
        process=process,
        log_file=open(os.devnull, "ab"),
        temp_dir=None,
        connect_url=connect_url,
        log_path=log_path,
    )
    try:
        _wait_for_gateway(connect_url, gateway, config.startup_timeout)
    except Exception:
        if process.poll() is None:
            _terminate_process_group(process, signal.SIGTERM)
        raise
    finally:
        gateway.log_file.close()
    return connect_url


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
        if _gateway_accepts_connections(connect_url):
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


def _unix_socket_path(connect_url: str) -> Path:
    prefix = "unix://"
    if not connect_url.startswith(prefix):
        raise TerminalLaunchError(
            f"local terminal launcher only creates unix:// sockets, got {connect_url!r}"
        )
    raw_path = connect_url.removeprefix(prefix)
    if not raw_path:
        raise TerminalLaunchError("local gateway socket path is empty")
    return Path(raw_path)


def _default_gateway_log() -> Path:
    home = Path(os.environ.get("AGENTM_HOME", str(Path.home() / ".agentm")))
    return home / "logs" / "terminal-gateway.log"


def _default_daemon_state_dir() -> Path:
    home = Path(os.environ.get("AGENTM_HOME", str(Path.home() / ".agentm")))
    return home / "gateway"


def _daemon_runtime_dir() -> Path:
    root = os.environ.get("AGENTM_RUNTIME_DIR")
    if root:
        path = Path(root)
    else:
        uid = os.getuid() if hasattr(os, "getuid") else os.getpid()
        home = os.environ.get("AGENTM_HOME", str(Path.home() / ".agentm"))
        digest = hashlib.sha1(home.encode("utf-8")).hexdigest()[:8]
        path = Path(tempfile.gettempdir()) / f"agentm-{uid}-{digest}"
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        path.chmod(0o700)
    except OSError:
        pass
    return path


def _default_daemon_connect_url() -> str:
    return f"unix://{_daemon_runtime_dir() / 'gateway.sock'}"


def _gateway_accepts_connections(connect_url: str) -> bool:
    try:
        socket_path = _unix_socket_path(connect_url)
    except TerminalLaunchError:
        return False
    if not socket_path.exists() or socket_path.is_dir():
        return False
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.2)
            probe.connect(str(socket_path))
            return True
    except OSError:
        return False


def _pid_file_is_live(pid_file: Path) -> bool:
    try:
        raw = pid_file.read_text(encoding="utf-8").strip()
        pid = int(raw)
    except (OSError, ValueError):
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _pid_only_gateway(pid_file: Path) -> _GatewayProcess:
    class _PidProcess:
        def poll(self) -> int | None:
            return None if _pid_file_is_live(pid_file) else 1

    return _GatewayProcess(
        process=_PidProcess(),  # type: ignore[arg-type]
        log_file=open(os.devnull, "ab"),
        temp_dir=None,
        connect_url=_default_daemon_connect_url(),
        log_path=_default_gateway_log(),
    )


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
        path = Path(value)
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
        raise TerminalLaunchError(f"executable not found or not executable: {value}")
    found = shutil.which(value)
    if found:
        return found
    raise TerminalLaunchError(f"executable not found on PATH: {value}")
