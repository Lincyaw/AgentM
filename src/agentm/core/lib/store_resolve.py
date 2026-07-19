"""Auto-resolve a trajectory store from environment / project / config.

Priority (first match wins):
1. AGENTM_TRAJECTORY_DSN env var   → Postgres
2. AGENTM_TRAJECTORY_DIR env var   → JSONL directory
3. agentm.toml [trajectory] dsn    → Postgres
4. agentm.toml [trajectory] dir    → JSONL directory
5. $cwd/.agentm/trajectory/ exists → JSONL (project-local)
6. None — no configured persistence

``resolve_or_create`` adds step 7: auto-create the project-local JSONL
directory so every session gets persistence by default.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

from loguru import logger

from agentm.core.abi.store import TrajectoryStore


def _load_trajectory_config(cwd: str | None) -> dict[str, str]:
    """Read [trajectory] section from agentm.toml if present."""
    candidates: list[Path] = []
    if cwd:
        candidates.append(Path(cwd) / "agentm.toml")
    candidates.append(Path.cwd() / "agentm.toml")
    home = os.environ.get("AGENTM_HOME")
    if home:
        candidates.append(Path(home).expanduser() / "config.toml")
    else:
        candidates.append(Path.home() / ".agentm" / "config.toml")

    for path in candidates:
        if path.exists():
            try:
                with path.open("rb") as f:
                    data = tomllib.load(f)
                section = data.get("trajectory")
                if isinstance(section, dict):
                    return {k: str(v) for k, v in section.items() if v is not None}
            except Exception:
                continue
    return {}


def resolve_trajectory_store(cwd: str | None = None) -> TrajectoryStore | None:
    """Return a store if one can be discovered, else None."""

    dsn = os.environ.get("AGENTM_TRAJECTORY_DSN")
    if dsn:
        return _postgres_store(dsn)

    explicit_dir = os.environ.get("AGENTM_TRAJECTORY_DIR")
    if explicit_dir:
        return _jsonl_store(Path(explicit_dir).expanduser())

    toml_cfg = _load_trajectory_config(cwd)
    if "dsn" in toml_cfg:
        return _postgres_store(toml_cfg["dsn"])
    if "dir" in toml_cfg:
        return _jsonl_store(Path(toml_cfg["dir"]).expanduser())

    base = Path(cwd) if cwd else Path.cwd()
    project_local = base / ".agentm" / "trajectory"
    if project_local.is_dir():
        return _jsonl_store(project_local)

    return None


def resolve_trajectory_store_or_create(
    cwd: str | None = None,
) -> TrajectoryStore:
    """Like resolve, but auto-creates project-local JSONL directory."""

    store = resolve_trajectory_store(cwd)
    if store is not None:
        return store

    base = Path(cwd) if cwd else Path.cwd()
    project_local = base / ".agentm" / "trajectory"
    project_local.mkdir(parents=True, exist_ok=True)
    logger.debug("trajectory store: auto-created {}", project_local)
    return _jsonl_store(project_local)


def _postgres_store(dsn: str) -> TrajectoryStore:
    try:
        from agentm.storage.trajectory.psycopg import connect
    except ModuleNotFoundError as exc:
        if exc.name not in {"psycopg", "psycopg_binary"}:
            raise
        raise RuntimeError(
            "Postgres trajectory storage requires the "
            "'agentm[storage-postgres]' extra (psycopg >= 3.2)"
        ) from exc

    from agentm.storage.trajectory.postgres_turns import PostgresTrajectoryStore

    conn = connect(dsn)
    return PostgresTrajectoryStore(conn, create_schema=True)


def _jsonl_store(directory: Path) -> TrajectoryStore:
    from agentm.core.runtime.stores.jsonl import JsonlTrajectoryStore

    directory.mkdir(parents=True, exist_ok=True)
    return JsonlTrajectoryStore(directory)


__all__ = [
    "resolve_trajectory_store",
    "resolve_trajectory_store_or_create",
]
