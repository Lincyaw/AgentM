"""Local gateway daemon management.

This module owns the small single-host supervisor lifecycle used by
``agentm daemon`` and the convenience ``agentm terminal`` command.  It does
not host AgentSession objects; the supervisor starts the normal
``agentm gateway`` worker and restarts it when watched files change.
"""

from __future__ import annotations

import hashlib
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class GatewayDaemonError(RuntimeError):
    """Raised when the local gateway daemon cannot be managed."""


@dataclass(frozen=True, slots=True)
class GatewayDaemonConfig:
    """Configuration for starting or reusing the local gateway daemon."""

    cwd: Path
    scenario: str | None = None
    state_dir: Path | None = None
    gateway_log: Path | None = None
    startup_timeout: float = 10.0
    reload: bool = True
    poll_interval: float = 1.0


@dataclass(frozen=True, slots=True)
class GatewayDaemonStatus:
    """Status snapshot for the local gateway daemon."""

    connect_url: str
    runtime_dir: Path
    pid_file: Path
    state_dir: Path
    log_path: Path
    pid: int | None
    pid_alive: bool
    socket_ready: bool

    @property
    def running(self) -> bool:
        return self.pid_alive and self.socket_ready

    def as_dict(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "socket_ready": self.socket_ready,
            "pid_alive": self.pid_alive,
            "pid": self.pid,
            "connect_url": self.connect_url,
            "runtime_dir": str(self.runtime_dir),
            "pid_file": str(self.pid_file),
            "state_dir": str(self.state_dir),
            "log_path": str(self.log_path),
        }


def default_agentm_home() -> Path:
    return Path(os.environ.get("AGENTM_HOME", str(Path.home() / ".agentm")))


def default_gateway_log() -> Path:
    return default_agentm_home() / "logs" / "terminal-gateway.log"


def default_daemon_state_dir() -> Path:
    return default_agentm_home() / "gateway"


def default_daemon_runtime_dir(*, create: bool = False) -> Path:
    root = os.environ.get("AGENTM_RUNTIME_DIR")
    if root:
        path = Path(root)
    else:
        uid = os.getuid() if hasattr(os, "getuid") else os.getpid()
        home = str(default_agentm_home())
        digest = hashlib.sha1(home.encode("utf-8")).hexdigest()[:8]
        path = Path(tempfile.gettempdir()) / f"agentm-{uid}-{digest}"
    if create:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            path.chmod(0o700)
        except OSError:
            pass
    return path


def default_daemon_connect_url(*, create_runtime_dir: bool = False) -> str:
    return f"unix://{default_daemon_runtime_dir(create=create_runtime_dir) / 'gateway.sock'}"


def default_daemon_pid_file(*, create_runtime_dir: bool = False) -> Path:
    return default_daemon_runtime_dir(create=create_runtime_dir) / "gateway-supervisor.pid"


def unix_socket_path(connect_url: str) -> Path:
    prefix = "unix://"
    if not connect_url.startswith(prefix):
        raise GatewayDaemonError(
            f"local gateway daemon only supports unix:// sockets, got {connect_url!r}"
        )
    raw_path = connect_url.removeprefix(prefix)
    if not raw_path:
        raise GatewayDaemonError("local gateway socket path is empty")
    return Path(raw_path)


def gateway_accepts_connections(connect_url: str) -> bool:
    try:
        socket_path = unix_socket_path(connect_url)
    except GatewayDaemonError:
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


def read_pid_file(pid_file: Path) -> int | None:
    try:
        return int(pid_file.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def pid_is_live(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def pid_file_is_live(pid_file: Path) -> bool:
    pid = read_pid_file(pid_file)
    return (
        pid is not None
        and pid_is_live(pid)
        and _pid_looks_like_gateway_supervisor(pid)
    )


def gateway_daemon_status() -> GatewayDaemonStatus:
    connect_url = default_daemon_connect_url()
    pid_file = default_daemon_pid_file()
    pid = read_pid_file(pid_file)
    return GatewayDaemonStatus(
        connect_url=connect_url,
        runtime_dir=default_daemon_runtime_dir(),
        pid_file=pid_file,
        state_dir=default_daemon_state_dir(),
        log_path=default_gateway_log(),
        pid=pid,
        pid_alive=pid is not None and pid_file_is_live(pid_file),
        socket_ready=gateway_accepts_connections(connect_url),
    )


def ensure_gateway_daemon(config: GatewayDaemonConfig) -> str:
    """Start the local daemon if needed and return its connect URL."""

    cwd = config.cwd
    if not cwd.is_dir():
        raise GatewayDaemonError(f"working directory does not exist: {cwd}")

    connect_url = default_daemon_connect_url(create_runtime_dir=True)
    if gateway_accepts_connections(connect_url):
        return connect_url

    pid_file = default_daemon_pid_file(create_runtime_dir=True)
    log_path = config.gateway_log or default_gateway_log()
    if pid_file_is_live(pid_file):
        _wait_for_gateway(
            connect_url,
            timeout_seconds=config.startup_timeout,
            pid_file=pid_file,
            log_path=log_path,
        )
        return connect_url

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("ab")
    state_dir = config.state_dir or default_daemon_state_dir()
    args = [
        sys.executable,
        "-m",
        "agentm.gateway_supervisor",
        "--cwd",
        str(cwd),
        "--bind",
        connect_url,
        "--state-dir",
        str(state_dir),
        "--pid-file",
        str(pid_file),
        "--poll-interval",
        str(max(config.poll_interval, 0.2)),
    ]
    if config.scenario:
        args.extend(["--scenario", config.scenario])
    if not config.reload:
        args.append("--no-reload")

    try:
        process = subprocess.Popen(
            args,
            cwd=str(cwd),
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

    try:
        _wait_for_gateway(
            connect_url,
            timeout_seconds=config.startup_timeout,
            process=process,
            log_path=log_path,
        )
    except Exception:
        if process.poll() is None:
            _terminate_process_group(process.pid, signal.SIGTERM)
        raise
    return connect_url


def stop_gateway_daemon(*, timeout_seconds: float = 5.0) -> bool:
    """Stop the local daemon. Returns True when a live pid was signalled."""

    pid_file = default_daemon_pid_file()
    pid = read_pid_file(pid_file)
    if pid is None or not pid_file_is_live(pid_file):
        _unlink_stale_pid_file(pid_file)
        return False

    _terminate_process_group(pid, signal.SIGTERM)
    if not _wait_for_pid_exit(pid, timeout_seconds):
        _terminate_process_group(pid, signal.SIGKILL)
        _wait_for_pid_exit(pid, 2.0)

    _unlink_stale_pid_file(pid_file)
    return True


def _wait_for_gateway(
    connect_url: str,
    *,
    timeout_seconds: float,
    log_path: Path,
    process: subprocess.Popen[Any] | None = None,
    pid_file: Path | None = None,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process is not None:
            rc = process.poll()
            if rc is not None:
                raise GatewayDaemonError(
                    f"gateway daemon exited before it was ready (exit {rc})."
                    f"{_gateway_log_hint(log_path)}"
                )
        if pid_file is not None and not pid_file_is_live(pid_file):
            raise GatewayDaemonError(
                "gateway daemon pid disappeared before the socket was ready."
                f"{_gateway_log_hint(log_path)}"
            )
        if gateway_accepts_connections(connect_url):
            return
        time.sleep(0.1)
    raise GatewayDaemonError(
        f"gateway socket was not ready after {timeout_seconds:.1f}s."
        f"{_gateway_log_hint(log_path)}"
    )


def _wait_for_pid_exit(pid: int, timeout_seconds: float) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not pid_is_live(pid):
            return True
        time.sleep(0.1)
    return not pid_is_live(pid)


def _terminate_process_group(pid: int, sig: signal.Signals) -> None:
    if os.name == "posix":
        try:
            os.killpg(pid, sig)
            return
        except ProcessLookupError:
            return
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        return


def _pid_looks_like_gateway_supervisor(pid: int) -> bool:
    if os.name != "posix":
        return True
    try:
        result = subprocess.run(  # noqa: S603, S607
            ["ps", "-p", str(pid), "-o", "command="],
            check=False,
            capture_output=True,
            text=True,
            timeout=1.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return True
    command = result.stdout.strip()
    return "agentm.gateway_supervisor" in command


def _unlink_stale_pid_file(pid_file: Path) -> None:
    try:
        pid_file.unlink()
    except OSError:
        pass


def _gateway_log_hint(path: Path) -> str:
    try:
        content = path.read_bytes()
    except OSError:
        return f" Log: {path}"
    if not content:
        return f" Log: {path}"
    tail = content[-4000:].decode("utf-8", errors="replace").strip()
    return f" Log: {path}\n{tail}"
