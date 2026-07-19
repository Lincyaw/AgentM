"""Host-side trajectory backend resolution and store construction."""

from __future__ import annotations

import os
import tomllib
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Protocol

from agentm.core.abi.store import TrajectoryStore
from agentm.storage.trajectory.jsonl import JsonlTrajectoryStore
from agentm.storage.trajectory.postgres import (
    PostgresConnection,
    PostgresTrajectoryStore,
)

_CONFIG_KEYS = frozenset({"dir", "dsn", "schema"})


@dataclass(frozen=True, slots=True)
class _JsonlLocation:
    directory: Path


@dataclass(frozen=True, slots=True)
class _PostgresLocation:
    dsn: str
    schema: str = "public"


_TrajectoryLocation = _JsonlLocation | _PostgresLocation


class _OwnedPostgresConnection(PostgresConnection, Protocol):
    def close(self) -> None: ...


@dataclass(slots=True)
class ResolvedTrajectoryStore:
    """A selected trajectory store plus shared resolver-owned resources."""

    store: TrajectoryStore
    _closers: tuple[Callable[[], None], ...] = field(
        default=(),
        repr=False,
    )
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _leases: int = field(default=0, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def acquire(self) -> None:
        """Keep the backend open for one session sharing this selection."""

        with self._lock:
            if self._closed:
                raise RuntimeError("trajectory storage is already closed")
            self._leases += 1

    def release(self) -> None:
        """Release one session lease and close after the last session."""

        with self._lock:
            if self._leases <= 0:
                raise RuntimeError("trajectory storage has no active lease")
            self._leases -= 1
            should_close = self._leases == 0
            if should_close:
                self._closed = True
        if should_close:
            self._close_resources()

    def close(self) -> None:
        """Close an unleased resolver result exactly once."""

        with self._lock:
            if self._closed:
                return
            if self._leases:
                raise RuntimeError(
                    "cannot close trajectory storage while sessions still use it"
                )
            self._closed = True
        self._close_resources()

    def _close_resources(self) -> None:
        errors: list[Exception] = []
        for closer in reversed(self._closers):
            try:
                closer()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise ExceptionGroup(
                "trajectory storage close failed",
                errors,
            )


def resolve_trajectory_store(
    cwd: str | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> ResolvedTrajectoryStore | None:
    """Resolve and open one configured trajectory store."""

    location = _resolve_location(cwd, env=env, create=False)
    if location is None:
        return None
    return _open_store(location)


def resolve_trajectory_store_or_create(
    cwd: str | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> ResolvedTrajectoryStore:
    """Resolve a configured backend or create the project-local JSONL store."""

    location = _resolve_location(cwd, env=env, create=True)
    if location is None:
        raise RuntimeError("trajectory location resolution returned no backend")
    return _open_store(location)


def _resolve_location(
    cwd: str | None,
    *,
    env: Mapping[str, str] | None,
    create: bool,
) -> _TrajectoryLocation | None:
    source_env = os.environ if env is None else env
    environment_location = _environment_location(source_env)
    if environment_location is not None:
        return environment_location

    base = Path(cwd).expanduser() if cwd else Path.cwd()
    user_config = _user_config_path(source_env)
    for config_path in _unique_paths((base / "agentm.toml", user_config)):
        configured = _config_location(config_path)
        if configured is not None:
            return configured

    project_local = base / ".agentm" / "trajectory"
    if create or project_local.is_dir():
        return _JsonlLocation(project_local)
    return None


def _environment_location(
    env: Mapping[str, str],
) -> _TrajectoryLocation | None:
    dsn = _optional_nonempty(env.get("AGENTM_TRAJECTORY_DSN"), "trajectory DSN")
    directory = _optional_nonempty(
        env.get("AGENTM_TRAJECTORY_DIR"),
        "trajectory directory",
    )
    schema = _optional_nonempty(
        env.get("AGENTM_TRAJECTORY_SCHEMA"),
        "trajectory schema",
    )
    if dsn is not None and directory is not None:
        raise ValueError(
            "AGENTM_TRAJECTORY_DSN and AGENTM_TRAJECTORY_DIR are mutually exclusive"
        )
    if directory is not None:
        if schema is not None:
            raise ValueError(
                "AGENTM_TRAJECTORY_SCHEMA requires AGENTM_TRAJECTORY_DSN"
            )
        return _JsonlLocation(Path(directory).expanduser())
    if dsn is not None:
        return _PostgresLocation(dsn=dsn, schema=schema or "public")
    if schema is not None:
        raise ValueError(
            "AGENTM_TRAJECTORY_SCHEMA requires AGENTM_TRAJECTORY_DSN"
        )
    return None


def _config_location(path: Path) -> _TrajectoryLocation | None:
    if not path.exists():
        return None
    with path.open("rb") as handle:
        loaded: object = tomllib.load(handle)
    if not isinstance(loaded, Mapping):
        raise TypeError(f"trajectory config root must be an object: {path}")
    section = loaded.get("trajectory")
    if section is None:
        return None
    if not isinstance(section, Mapping):
        raise TypeError(f"[trajectory] must be a table: {path}")
    if not all(isinstance(key, str) for key in section):
        raise TypeError(f"[trajectory] keys must be strings: {path}")
    keys = {key for key in section if isinstance(key, str)}
    unknown = keys - _CONFIG_KEYS
    if unknown:
        rendered = ", ".join(sorted(unknown))
        raise ValueError(f"unknown [trajectory] keys in {path}: {rendered}")

    dsn = _config_string(section, "dsn", path)
    directory = _config_string(section, "dir", path)
    schema = _config_string(section, "schema", path)
    if dsn is not None and directory is not None:
        raise ValueError(f"[trajectory] dsn and dir are mutually exclusive: {path}")
    if directory is not None:
        if schema is not None:
            raise ValueError(f"[trajectory] schema requires dsn: {path}")
        configured_path = Path(directory).expanduser()
        if not configured_path.is_absolute():
            configured_path = path.parent / configured_path
        return _JsonlLocation(configured_path)
    if dsn is not None:
        return _PostgresLocation(dsn=dsn, schema=schema or "public")
    if schema is not None:
        raise ValueError(f"[trajectory] schema requires dsn: {path}")
    return None


def _config_string(
    section: Mapping[object, object],
    key: str,
    path: Path,
) -> str | None:
    value = section.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"[trajectory].{key} must be a non-empty string: {path}")
    return value


def _optional_nonempty(value: str | None, label: str) -> str | None:
    if value is None:
        return None
    if not value.strip():
        raise ValueError(f"{label} cannot be empty")
    return value


def _user_config_path(env: Mapping[str, str]) -> Path:
    home = env.get("AGENTM_HOME")
    if home:
        return Path(home).expanduser() / "config.toml"
    return Path.home() / ".agentm" / "config.toml"


def _unique_paths(paths: tuple[Path, ...]) -> tuple[Path, ...]:
    unique: list[Path] = []
    for path in paths:
        if path not in unique:
            unique.append(path)
    return tuple(unique)


def _open_store(
    location: _TrajectoryLocation,
) -> ResolvedTrajectoryStore:
    if isinstance(location, _JsonlLocation):
        return ResolvedTrajectoryStore(
            store=JsonlTrajectoryStore(location.directory)
        )
    return _open_postgres_store(location)


def _open_postgres_store(
    location: _PostgresLocation,
) -> ResolvedTrajectoryStore:
    connect = _postgres_connect()
    with ExitStack() as cleanup:
        connection = connect(location.dsn)
        cleanup.callback(connection.close)
        store = PostgresTrajectoryStore(
            connection,
            schema=location.schema,
            create_schema=True,
        )
        cleanup.pop_all()
    return ResolvedTrajectoryStore(
        store=store,
        _closers=(connection.close,),
    )


def _postgres_connect() -> Callable[[str], _OwnedPostgresConnection]:
    try:
        from agentm.storage.trajectory.psycopg import (
            connect,
        )
    except ModuleNotFoundError as exc:
        if exc.name not in {"psycopg", "psycopg_binary"}:
            raise
        raise RuntimeError(
            "Postgres trajectory storage requires the "
            "'agentm[storage-postgres]' extra (psycopg >= 3.2)"
        ) from exc

    typed_connect: Callable[[str], _OwnedPostgresConnection] = connect
    return typed_connect


__all__ = [
    "ResolvedTrajectoryStore",
    "resolve_trajectory_store",
    "resolve_trajectory_store_or_create",
]
