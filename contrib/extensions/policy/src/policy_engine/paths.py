"""Filesystem locations shared by the policy runtime and its CLI."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path


def default_policy_db_path() -> Path:
    """Return the policy DB selected by the AgentM home contract."""

    agentm_home = os.environ.get("AGENTM_HOME", str(Path.home() / ".agentm"))
    return Path(agentm_home).expanduser() / "policy_state" / "policy.db"


def resolve_policy_db_path(
    *,
    session_id: str | None = None,
    cwd: Path | None = None,
) -> Path:
    """Locate a repo-local policy DB containing ``session_id`` when possible.

    The runtime has an explicit ``AGENTM_HOME``. An independently started viewer
    does not, so it discovers run-specific homes below the project's ``.agentm``
    directory and identifies the right database by persisted session rows.
    """

    if os.environ.get("AGENTM_HOME"):
        default_path = default_policy_db_path()
        candidates = _agentm_home_policy_db_candidates(default_path)
        if session_id:
            for candidate in candidates:
                if _database_has_session(candidate, session_id):
                    return candidate
        if default_path.is_file() or not candidates:
            return default_path
        return max(candidates, key=_modified_at)

    candidates = _project_policy_db_candidates(cwd or Path.cwd())
    if session_id:
        for candidate in candidates:
            if _database_has_session(candidate, session_id):
                return candidate
    if candidates:
        return max(candidates, key=_modified_at)
    return default_policy_db_path()


def _project_policy_db_candidates(cwd: Path) -> tuple[Path, ...]:
    project_root = _find_project_root(cwd.expanduser().resolve())
    agentm_dir = project_root / ".agentm"
    paths = {
        path.resolve()
        for pattern in (
            "policy_state/policy.db",
            "policy_state/sessions/*.db",
            "*/policy_state/policy.db",
            "*/policy_state/sessions/*.db",
        )
        for path in agentm_dir.glob(pattern)
        if path.is_file()
    }
    return tuple(sorted(paths))


def _agentm_home_policy_db_candidates(default_path: Path) -> tuple[Path, ...]:
    paths = {path.resolve() for path in default_path.parent.glob("sessions/*.db")}
    if default_path.is_file():
        paths.add(default_path.resolve())
    return tuple(sorted(paths))


def _find_project_root(path: Path) -> Path:
    current = path if path.is_dir() else path.parent
    for candidate in (current, *current.parents):
        if (candidate / "agentm.toml").is_file():
            return candidate
    return current


def _database_has_session(path: Path, session_id: str) -> bool:
    try:
        connection = sqlite3.connect(
            f"{path.as_uri()}?mode=ro",
            uri=True,
            timeout=0.2,
        )
        try:
            tables = {
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type = 'table' AND name IN "
                    "('policy_tool_events', 'policy_session_summary', "
                    "'ifg_session_summary')"
                )
            }
            for table in (
                "policy_tool_events",
                "policy_session_summary",
                "ifg_session_summary",
            ):
                if table not in tables:
                    continue
                row = connection.execute(
                    f"SELECT 1 FROM {table} WHERE session_id = ? LIMIT 1",  # noqa: S608
                    (session_id,),
                ).fetchone()
                if row is not None:
                    return True
        finally:
            connection.close()
    except sqlite3.Error:
        return False
    return False


def _modified_at(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


__all__ = ["default_policy_db_path", "resolve_policy_db_path"]
