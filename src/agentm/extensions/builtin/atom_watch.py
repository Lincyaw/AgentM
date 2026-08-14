"""Reload atoms from a watched directory while the session runs.

Compose this atom to get the development loop: edit an atom's source on disk
and the running session picks it up, without restarting and without losing the
conversation so far. Compose without it and the session's atoms are fixed at
start, which is what a recorded run wants.

That is the whole of the mode distinction. There is no dev flag in the runtime;
the difference between a development session and a normal one is which atoms
were composed, the same way a package's dev script differs from its start
script.

Watching is deliberately coarse: a poll on size and mtime, no dependency on a
filesystem notification library. The interval is the upper bound on how stale a
change can be, and a development loop does not need better.
"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, Field

from agentm.core.abi import (
    AtomAPI,
    DiagnosticEvent,
    ExtensionManifest,
    ExtensionSource,
    ExtensionSpec,
    SessionReadyEvent,
    SessionShutdownEvent,
)


class AtomWatchConfig(BaseModel):
    """Which directory to watch, and how often."""

    directory: str = Field(
        default=".agentm/authored_tools",
        description="Directory, relative to cwd, watched for atom source.",
    )
    interval_seconds: float = Field(
        default=1.0,
        gt=0.0,
        description="Seconds between scans. The upper bound on staleness.",
    )
    install_new: bool = Field(
        default=True,
        description="Install files that appear after the session has started.",
    )


MANIFEST = ExtensionManifest(
    name="atom_watch",
    description="Reload atoms from a watched directory while the session runs.",
    registers=("event:session_ready", "event:session_shutdown"),
    config_schema=AtomWatchConfig,
)


class _Watcher:
    """Polls a directory and reinstalls atoms whose source changed."""

    def __init__(self, api: AtomAPI, config: AtomWatchConfig) -> None:
        self._api = api
        self._directory = Path(api.ctx.cwd) / config.directory
        self._interval = config.interval_seconds
        self._install_new = config.install_new
        self._task: asyncio.Task[None] | None = None
        self._seen: dict[Path, tuple[int, float]] = {}

    def on_session_ready(self, _event: SessionReadyEvent) -> None:
        """Take the current directory as the baseline and start polling."""

        self._seen = self._scan()
        self._task = asyncio.create_task(self._loop(), name="agentm-atom-watch")
        logger.info(
            "watching {} for atom changes every {}s",
            self._directory,
            self._interval,
        )

    async def on_session_shutdown(self, _event: SessionShutdownEvent) -> None:
        task = self._task
        self._task = None
        if task is None:
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    def _scan(self) -> dict[Path, tuple[int, float]]:
        found: dict[Path, tuple[int, float]] = {}
        try:
            entries = sorted(self._directory.glob("*.py"))
        except OSError as exc:
            logger.warning("atom watch cannot read {}: {}", self._directory, exc)
            return found
        for path in entries:
            try:
                stat = path.stat()
            except OSError:
                continue
            found[path] = (stat.st_size, stat.st_mtime)
        return found

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(self._interval)
            try:
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - a dev loop must not die
                logger.warning("atom watch scan failed: {}", exc)

    async def _tick(self) -> None:
        current = self._scan()
        changed = [
            path
            for path, stamp in current.items()
            if self._seen.get(path) != stamp
            and (self._install_new or path in self._seen)
        ]
        # Record every path scanned, including ones skipped as new, so a file
        # is reported once rather than on every tick.
        self._seen = current
        for path in changed:
            await self._reload(path)

    async def _reload(self, path: Path) -> None:
        try:
            content = path.read_bytes()
        except OSError as exc:
            logger.warning("atom watch cannot read {}: {}", path, exc)
            return
        digest = "sha256:" + hashlib.sha256(content).hexdigest()
        spec = ExtensionSpec(
            source=ExtensionSource(
                kind="file",
                location=str(path.resolve()),
                digest=digest,
            ),
            config={},
        )
        # Hold the work bracket only across the install, so a reload in flight
        # is not cut short, and the session can still reach idle between scans.
        with self._api.track_background():
            try:
                await self._api.install_extension(
                    spec,
                    trigger="atom_watch",
                    replace=True,
                )
            except Exception as exc:  # noqa: BLE001 - surfaced, not swallowed
                logger.warning("atom watch could not load {}: {}", path.name, exc)
                await self._api.bus.emit(
                    DiagnosticEvent.CHANNEL,
                    DiagnosticEvent(
                        level="warning",
                        source="atom_watch",
                        message=f"{path.name} failed to reload: {exc}",
                    ),
                )
                return
        logger.info("reloaded atom from {}", path.name)
        await self._api.bus.emit(
            DiagnosticEvent.CHANNEL,
            DiagnosticEvent(
                level="info",
                source="atom_watch",
                message=f"{path.name} reloaded; its tools apply from the next turn",
            ),
        )


def install(api: AtomAPI, config: AtomWatchConfig) -> None:
    """Start watching once the session is ready."""

    watcher = _Watcher(api, config)
    api.on(SessionReadyEvent.CHANNEL, watcher.on_session_ready)
    api.on(SessionShutdownEvent.CHANNEL, watcher.on_session_shutdown)
