"""Local gateway daemon management.

This module owns the small single-host supervisor lifecycle used by
``agentm daemon`` and the convenience ``agentm terminal`` command.  It does
not host AgentSession objects; the supervisor starts the normal
``agentm gateway`` worker and can restart it when watched files change if
explicitly asked to do so.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import secrets
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from agentm.core.lib.user_config import agentm_home_dir


def _expand_path(path: Path | str) -> Path:
    return Path(os.path.expandvars(str(path))).expanduser()


class GatewayDaemonError(RuntimeError):
    """Raised when the local gateway daemon cannot be managed."""


@dataclass(frozen=True, slots=True)
class GatewayDaemonConfig:
    """Configuration for starting or reusing the local gateway daemon."""

    cwd: Path
    scenario: str | None = None
    bind: str | None = None
    bind_token_file: Path | None = None
    bind_allow_anonymous: bool = False
    tls_cert: Path | None = None
    tls_key: Path | None = None
    state_dir: Path | None = None
    gateway_log: Path | None = None
    startup_timeout: float = 10.0
    reload: bool = False
    poll_interval: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "cwd", _expand_path(self.cwd))
        for field_name in (
            "bind_token_file",
            "tls_cert",
            "tls_key",
            "state_dir",
            "gateway_log",
        ):
            path = getattr(self, field_name)
            if path is not None:
                object.__setattr__(self, field_name, _expand_path(path))


@dataclass(frozen=True, slots=True)
class GatewayDaemonStatus:
    """Status snapshot for the local gateway daemon."""

    connect_url: str
    runtime_dir: Path
    pid_file: Path
    state_dir: Path
    log_path: Path
    token_file: Path | None
    auth_required: bool
    reload: bool
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
            "token_file": str(self.token_file) if self.token_file else None,
            "auth_required": self.auth_required,
            "reload": self.reload,
        }


def default_agentm_home() -> Path:
    return agentm_home_dir()


def default_gateway_log() -> Path:
    return default_agentm_home() / "logs" / "terminal-gateway.log"


def default_daemon_state_dir() -> Path:
    return default_agentm_home() / "gateway"


def default_daemon_token_file() -> Path:
    return default_daemon_state_dir() / "token"


def default_daemon_runtime_dir(*, create: bool = False) -> Path:
    root = os.environ.get("AGENTM_RUNTIME_DIR")
    if root:
        path = _expand_path(root)
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


def default_daemon_metadata_file(*, create_runtime_dir: bool = False) -> Path:
    return default_daemon_runtime_dir(create=create_runtime_dir) / "gateway-daemon.json"


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
    parsed = urlparse(connect_url)
    if parsed.scheme in {"ws", "wss"}:
        return _websocket_accepts_connections(connect_url)
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
    metadata = _read_daemon_metadata()
    connect_url = str(metadata.get("connect_url") or default_daemon_connect_url())
    pid_file = default_daemon_pid_file()
    pid = read_pid_file(pid_file)
    token_file_raw = metadata.get("token_file")
    token_file = Path(str(token_file_raw)) if token_file_raw else None
    state_dir_raw = metadata.get("state_dir")
    log_path_raw = metadata.get("log_path")
    return GatewayDaemonStatus(
        connect_url=connect_url,
        runtime_dir=default_daemon_runtime_dir(),
        pid_file=pid_file,
        state_dir=Path(str(state_dir_raw)) if state_dir_raw else default_daemon_state_dir(),
        log_path=Path(str(log_path_raw)) if log_path_raw else default_gateway_log(),
        token_file=token_file,
        auth_required=bool(metadata.get("auth_required", False)),
        reload=bool(metadata.get("reload", False)),
        pid=pid,
        pid_alive=pid is not None and pid_file_is_live(pid_file),
        socket_ready=gateway_accepts_connections(connect_url),
    )


def ensure_gateway_daemon(config: GatewayDaemonConfig) -> str:
    """Start the local daemon if needed and return its connect URL."""

    cwd = config.cwd
    if not cwd.is_dir():
        raise GatewayDaemonError(f"working directory does not exist: {cwd}")

    pid_file = default_daemon_pid_file(create_runtime_dir=True)
    active_metadata = _read_daemon_metadata()
    active_url = active_metadata.get("connect_url")
    active_reload = bool(active_metadata.get("reload", False))
    if pid_file_is_live(pid_file):
        connect_url = str(config.bind or active_url or default_daemon_connect_url())
        if active_url and str(active_url) != connect_url:
            raise GatewayDaemonError(
                f"gateway daemon is already running at {active_url}; "
                "stop or restart it to change --bind"
            )
        if config.reload and not active_reload:
            raise GatewayDaemonError(
                "gateway daemon is already running without supervisor reload; "
                "use `agentm daemon restart --reload` to enable it"
            )
        log_path = config.gateway_log or default_gateway_log()
        if gateway_accepts_connections(connect_url):
            return connect_url
        _wait_for_gateway(
            connect_url,
            timeout_seconds=config.startup_timeout,
            pid_file=pid_file,
            log_path=log_path,
        )
        return connect_url

    connect_url = config.bind or default_daemon_connect_url(create_runtime_dir=True)
    token_file = _resolve_auth_token_file(
        connect_url,
        token_file=config.bind_token_file,
        allow_anonymous=config.bind_allow_anonymous,
        tls_cert=config.tls_cert,
        tls_key=config.tls_key,
    )
    log_path = config.gateway_log or default_gateway_log()
    state_dir = config.state_dir or default_daemon_state_dir()
    metadata = {
        "connect_url": connect_url,
        "state_dir": str(state_dir),
        "log_path": str(log_path),
        "token_file": str(token_file) if token_file else None,
        "auth_required": token_file is not None,
        "reload": config.reload,
    }
    if gateway_accepts_connections(connect_url):
        _write_daemon_metadata(metadata)
        return connect_url

    _write_daemon_metadata(metadata)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("ab")
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
    if token_file is not None:
        args.extend(["--bind-token-file", str(token_file)])
    if config.bind_allow_anonymous:
        args.append("--bind-allow-anonymous")
    if config.tls_cert is not None:
        args.extend(["--tls-cert", str(config.tls_cert)])
    if config.tls_key is not None:
        args.extend(["--tls-key", str(config.tls_key)])
    if config.scenario:
        args.extend(["--scenario", config.scenario])
    if config.reload:
        args.append("--reload")

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


def _resolve_auth_token_file(
    connect_url: str,
    *,
    token_file: Path | None,
    allow_anonymous: bool,
    tls_cert: Path | None,
    tls_key: Path | None,
) -> Path | None:
    parsed = urlparse(connect_url)
    scheme = parsed.scheme
    if scheme == "unix":
        if token_file is not None or allow_anonymous or tls_cert is not None or tls_key is not None:
            raise GatewayDaemonError(
                "--bind-token-file, --bind-allow-anonymous, --tls-cert, and "
                "--tls-key are only valid with ws:// or wss:// daemon binds"
            )
        return None
    if scheme not in {"ws", "wss"}:
        raise GatewayDaemonError(
            f"daemon --bind scheme {scheme!r} is not supported; use unix://, ws://, or wss://"
        )
    if scheme == "wss" and (tls_cert is None or tls_key is None):
        raise GatewayDaemonError("wss:// daemon bind requires --tls-cert and --tls-key")
    if scheme == "ws" and (tls_cert is not None or tls_key is not None):
        raise GatewayDaemonError("ws:// daemon bind cannot use TLS; use wss://")
    if allow_anonymous:
        if token_file is not None:
            raise GatewayDaemonError(
                "--bind-token-file and --bind-allow-anonymous are mutually exclusive"
            )
        return None
    return _ensure_token_file(token_file or default_daemon_token_file())


def _ensure_token_file(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _read_nonempty_token(path)
        _chmod_private(path)
        return path
    token = "agm_" + secrets.token_urlsafe(32)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        fd = os.open(str(path), flags, 0o600)
    except FileExistsError:
        _read_nonempty_token(path)
        _chmod_private(path)
        return path
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(token + "\n")
    _chmod_private(path)
    return path


def _read_nonempty_token(path: Path) -> str:
    try:
        token = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise GatewayDaemonError(f"cannot read daemon token file {path}: {exc}") from exc
    if not token:
        raise GatewayDaemonError(f"daemon token file is empty: {path}")
    return token


def _chmod_private(path: Path) -> None:
    try:
        path.chmod(0o600)
    except OSError:
        pass


def _read_daemon_metadata() -> dict[str, Any]:
    path = default_daemon_metadata_file()
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _write_daemon_metadata(data: dict[str, Any]) -> None:
    path = default_daemon_metadata_file(create_runtime_dir=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, sort_keys=True) + "\n", encoding="utf-8")
    try:
        tmp.chmod(0o600)
    except OSError:
        pass
    tmp.replace(path)


def _websocket_accepts_connections(connect_url: str) -> bool:
    try:
        from websockets.sync.client import connect as ws_connect
    except Exception:
        return False
    probe_url = _websocket_probe_url(connect_url)
    kwargs: dict[str, Any] = {"open_timeout": 0.2, "close_timeout": 0.2}
    if probe_url.startswith("wss://"):
        import ssl

        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        kwargs["ssl"] = context
    try:
        with ws_connect(probe_url, **kwargs):
            return True
    except Exception:
        return False


def _websocket_probe_url(connect_url: str) -> str:
    parsed = urlparse(connect_url)
    host = parsed.hostname or "127.0.0.1"
    if host in {"0.0.0.0", "::"}:
        host = "127.0.0.1" if host == "0.0.0.0" else "::1"
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    netloc = host
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    return parsed._replace(netloc=netloc, path=parsed.path or "/").geturl()


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
