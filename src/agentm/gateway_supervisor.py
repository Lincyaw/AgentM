"""Development supervisor for the gateway worker.

The supervisor intentionally does not host AgentSession objects.  It keeps a
stable daemon process around and can watch source/config files when requested.
When reload is enabled, it restarts a fresh ``agentm gateway`` worker after
changes so the worker imports current code on every restart.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentm.core.lib import agentm_home_dir, expand_path


_WATCH_SUFFIXES = {".py", ".yaml", ".yml", ".toml", ".json", ".md"}
_WATCH_NAMES = {".env", "AGENTS.md", "CLAUDE.md"}
_SKIP_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "node_modules",
}


@dataclass(slots=True)
class SupervisorConfig:
    cwd: Path
    bind: str
    state_dir: Path
    scenario: str | None = None
    bind_token_file: Path | None = None
    bind_allow_anonymous: bool = False
    tls_cert: Path | None = None
    tls_key: Path | None = None
    reload: bool = False
    poll_interval: float = 1.0
    pid_file: Path | None = None
    watch_paths: list[Path] = field(default_factory=list)


class GatewaySupervisor:
    """Owns a restartable ``agentm gateway`` worker process."""

    def __init__(self, config: SupervisorConfig) -> None:
        self._config = config
        self._worker: subprocess.Popen[Any] | None = None
        self._stopping = False

    def run(self) -> int:
        self._install_signal_handlers()
        self._write_pid_file()
        fingerprint = self._snapshot() if self._config.reload else {}
        try:
            self._start_worker()
            while not self._stopping:
                worker = self._worker
                if worker is not None and worker.poll() is not None:
                    self._log(f"gateway worker exited with {worker.returncode}; restarting")
                    self._start_worker()
                    if self._config.reload:
                        fingerprint = self._snapshot()

                if self._config.reload:
                    current = self._snapshot()
                    if current != fingerprint:
                        self._log("source change detected; restarting gateway worker")
                        fingerprint = current
                        self._restart_worker()

                time.sleep(self._config.poll_interval)
        finally:
            self._stop_worker()
            self._remove_pid_file()
        return 0

    def _install_signal_handlers(self) -> None:
        def _handle(_signum: int, _frame: Any) -> None:
            self._stopping = True

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, _handle)
            except (ValueError, OSError):
                pass

    def _start_worker(self) -> None:
        self._stop_worker()
        self._config.state_dir.mkdir(parents=True, exist_ok=True)
        args = [
            sys.executable,
            "-m",
            "agentm.gateway.cli",
            "--cwd",
            str(self._config.cwd),
            "--bind",
            self._config.bind,
            "--state-dir",
            str(self._config.state_dir),
        ]
        if self._config.bind_token_file is not None:
            args.extend(["--bind-token-file", str(self._config.bind_token_file)])
        if self._config.bind_allow_anonymous:
            args.append("--bind-allow-anonymous")
        if self._config.tls_cert is not None:
            args.extend(["--tls-cert", str(self._config.tls_cert)])
        if self._config.tls_key is not None:
            args.extend(["--tls-key", str(self._config.tls_key)])
        if self._config.scenario:
            args.extend(["--scenario", self._config.scenario])
        self._log("starting gateway worker: " + " ".join(args))
        self._worker = subprocess.Popen(args, start_new_session=True)

    def _restart_worker(self) -> None:
        self._stop_worker()
        if not self._stopping:
            self._start_worker()

    def _stop_worker(self) -> None:
        worker = self._worker
        if worker is None:
            return
        self._worker = None
        if worker.poll() is not None:
            return
        self._terminate_process_group(worker, signal.SIGTERM)
        try:
            worker.wait(timeout=20.0)
        except subprocess.TimeoutExpired:
            self._log("gateway worker did not stop cleanly; killing")
            self._terminate_process_group(worker, signal.SIGKILL)
            worker.wait(timeout=5.0)

    def _terminate_process_group(
        self, process: subprocess.Popen[Any], sig: signal.Signals
    ) -> None:
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

    def _snapshot(self) -> dict[str, tuple[int, int]]:
        snapshot: dict[str, tuple[int, int]] = {}
        for path in self._iter_watch_files():
            try:
                stat = path.stat()
            except OSError:
                continue
            snapshot[str(path)] = (stat.st_mtime_ns, stat.st_size)
        return snapshot

    def _iter_watch_files(self) -> list[Path]:
        files: list[Path] = []
        for root in self._config.watch_paths:
            if root.is_file():
                if _is_watch_file(root):
                    files.append(root)
                continue
            if not root.is_dir():
                continue
            for path in root.rglob("*"):
                if any(part in _SKIP_DIRS for part in path.parts):
                    continue
                if path.is_file() and _is_watch_file(path):
                    files.append(path)
        return files

    def _write_pid_file(self) -> None:
        if self._config.pid_file is None:
            return
        self._config.pid_file.parent.mkdir(parents=True, exist_ok=True)
        self._config.pid_file.write_text(f"{os.getpid()}\n", encoding="utf-8")

    def _remove_pid_file(self) -> None:
        if self._config.pid_file is None:
            return
        try:
            if self._config.pid_file.read_text(encoding="utf-8").strip() == str(os.getpid()):
                self._config.pid_file.unlink()
        except OSError:
            pass

    def _log(self, message: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"{ts} supervisor: {message}", flush=True)


def default_watch_paths(cwd: Path) -> list[Path]:
    package_dir = Path(__file__).parent
    paths: list[Path] = [package_dir]

    src_dir = package_dir.parent
    repo_dir = src_dir.parent
    if (repo_dir / "pyproject.toml").is_file() and (repo_dir / "src" / "agentm").is_dir():
        paths.extend(
            [
                repo_dir / "pyproject.toml",
                repo_dir / "uv.lock",
                repo_dir / "contrib" / "scenarios",
                repo_dir / "contrib" / "extensions",
            ]
        )

    paths.extend(
        [
            cwd / ".env",
            cwd / ".agentm" / "atoms",
            agentm_home_dir() / "config.toml",
        ]
    )
    return paths


def _is_watch_file(path: Path) -> bool:
    return path.name in _WATCH_NAMES or path.suffix in _WATCH_SUFFIXES


def _optional_path(value: str | None) -> Path | None:
    return expand_path(value) if value else None


def _parse_args(argv: list[str] | None) -> SupervisorConfig:
    parser = argparse.ArgumentParser(prog="python -m agentm.gateway_supervisor")
    parser.add_argument("--cwd", required=True)
    parser.add_argument("--bind", required=True)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--bind-token-file")
    parser.add_argument("--bind-allow-anonymous", action="store_true")
    parser.add_argument("--tls-cert")
    parser.add_argument("--tls-key")
    parser.add_argument("--scenario")
    parser.add_argument("--pid-file")
    parser.add_argument(
        "--reload",
        dest="reload",
        action="store_true",
        help="watch source/config files and restart the gateway worker on changes",
    )
    parser.add_argument(
        "--no-reload",
        dest="reload",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(reload=False)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--watch", action="append", default=[])
    ns = parser.parse_args(argv)
    cwd = expand_path(ns.cwd)
    watch_paths = default_watch_paths(cwd)
    watch_paths.extend(expand_path(p) for p in ns.watch)
    return SupervisorConfig(
        cwd=cwd,
        bind=ns.bind,
        state_dir=expand_path(ns.state_dir),
        scenario=ns.scenario,
        bind_token_file=_optional_path(ns.bind_token_file),
        bind_allow_anonymous=bool(ns.bind_allow_anonymous),
        tls_cert=_optional_path(ns.tls_cert),
        tls_key=_optional_path(ns.tls_key),
        reload=bool(ns.reload),
        poll_interval=max(float(ns.poll_interval), 0.2),
        pid_file=_optional_path(ns.pid_file),
        watch_paths=watch_paths,
    )


def main(argv: list[str] | None = None) -> int:
    return GatewaySupervisor(_parse_args(argv)).run()


if __name__ == "__main__":
    raise SystemExit(main())
