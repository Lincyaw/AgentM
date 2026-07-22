# code-health: ignore-file[AM025] -- validates ast-grep JSON at the process boundary
"""Remote repository-outline indexing for sandbox-backed AgentM sessions.

The worker intentionally uses only the Python standard library plus the
``ast-grep`` executable.  It runs next to the repository, keeps the complete
outline in a sandbox-local SQLite file, and returns only a path manifest and
the small set of documents requested by the host policy runtime.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path


_SCHEMA = """
CREATE TABLE IF NOT EXISTS repository_documents (
    path TEXT PRIMARY KEY,
    document_json TEXT NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS repository_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""
_ITEM_KEYS = (
    "role",
    "symbolType",
    "name",
    "range",
    "signature",
    "astKind",
    "isImport",
    "isExported",
    "isPublic",
)
_MAX_RETURNED_DOCUMENTS = 64


class RepositoryIndexWorkerError(RuntimeError):
    """Raised when the remote outline command cannot produce a valid index."""


def update_repository_index(
    *,
    root: str,
    target: str,
    db_path: str,
    replace: bool,
    include_documents: bool,
) -> dict[str, object]:
    """Scan ``target`` and update the sandbox-local repository index."""

    normalized_root = _normalize(root)
    normalized_target = _normalize_target(normalized_root, target)
    documents = _scan_target(normalized_root, normalized_target)
    database = Path(db_path)
    database.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.executescript(_SCHEMA)
        removed_paths = _scope_paths(connection, normalized_target)
        now = time.time()
        with connection:
            if replace:
                connection.execute("DELETE FROM repository_documents")
            else:
                _delete_scope(connection, normalized_target)
            connection.executemany(
                """
                INSERT INTO repository_documents(path, document_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    document_json = excluded.document_json,
                    updated_at = excluded.updated_at
                """,
                (
                    (
                        path,
                        json.dumps(document, ensure_ascii=False, separators=(",", ":")),
                        now,
                    )
                    for path, document in documents.items()
                ),
            )
            connection.execute(
                """
                INSERT INTO repository_metadata(key, value) VALUES ('root', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (normalized_root,),
            )

        all_paths = _all_paths(connection)
        returned_documents = (
            tuple(documents.values())[:_MAX_RETURNED_DOCUMENTS]
            if include_documents
            else ()
        )

    return {
        "version": 1,
        "ok": True,
        "root": normalized_root,
        "target": normalized_target,
        "replace": replace,
        "files": len(all_paths),
        "paths": list(all_paths if replace else documents),
        "removed_paths": list(removed_paths),
        "documents": list(returned_documents),
        "documents_truncated": include_documents
        and len(documents) > _MAX_RETURNED_DOCUMENTS,
    }


def load_repository_documents(
    *,
    root: str,
    target: str,
    db_path: str,
) -> dict[str, object]:
    """Load cached outline documents without rescanning repository files."""

    normalized_root = _normalize(root)
    normalized_target = _normalize_target(normalized_root, target)
    database = Path(db_path)
    if not database.is_file():
        raise RepositoryIndexWorkerError(f"repository index not found: {database}")

    with sqlite3.connect(database) as connection:
        stored_root = connection.execute(
            "SELECT value FROM repository_metadata WHERE key = 'root'"
        ).fetchone()
        if stored_root is None or _normalize(str(stored_root[0])) != normalized_root:
            raise RepositoryIndexWorkerError(
                "repository index root does not match the requested repository"
            )
        prefix = f"{normalized_target.rstrip('/')}/*"
        rows = connection.execute(
            """
            SELECT path, document_json
            FROM repository_documents
            WHERE path = ? OR path GLOB ?
            ORDER BY path
            LIMIT ?
            """,
            (normalized_target, prefix, _MAX_RETURNED_DOCUMENTS + 1),
        ).fetchall()
        file_count = connection.execute(
            "SELECT COUNT(*) FROM repository_documents"
        ).fetchone()

    documents: list[Mapping[str, object]] = []
    for path, raw_document in rows[:_MAX_RETURNED_DOCUMENTS]:
        try:
            document = json.loads(str(raw_document))
        except json.JSONDecodeError as exc:
            raise RepositoryIndexWorkerError(
                f"repository index contains invalid JSON for {path}: {exc}"
            ) from exc
        if isinstance(document, Mapping):
            documents.append(document)

    return {
        "version": 1,
        "ok": True,
        "root": normalized_root,
        "target": normalized_target,
        "replace": False,
        "files": int(file_count[0]) if file_count is not None else 0,
        "paths": [str(path) for path, _document in rows],
        "removed_paths": [],
        "documents": documents,
        "documents_truncated": len(rows) > _MAX_RETURNED_DOCUMENTS,
        "cache": "hit",
    }


def _scan_target(root: str, target: str) -> dict[str, Mapping[str, object]]:
    if not os.path.exists(target):
        return {}
    try:
        completed = subprocess.run(
            [
                "ast-grep",
                "outline",
                target,
                "--json=compact",
                "--items",
                "all",
                "--view",
                "expanded",
                "--threads",
                "1",
            ],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise RepositoryIndexWorkerError(
            f"outline execution failed: {type(exc).__name__}: {exc}"
        ) from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip()[:500] or f"exit {completed.returncode}"
        raise RepositoryIndexWorkerError(f"outline failed: {detail}")
    try:
        payload = json.loads(completed.stdout or "[]")
    except json.JSONDecodeError as exc:
        raise RepositoryIndexWorkerError(
            f"outline returned invalid JSON: {exc}"
        ) from exc
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        raise RepositoryIndexWorkerError("outline returned a non-array payload")

    documents: dict[str, Mapping[str, object]] = {}
    for raw_document in payload:
        if not isinstance(raw_document, Mapping):
            continue
        raw_path = raw_document.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            continue
        path = _normalize_target(root, raw_path)
        documents[path] = _compact_document(raw_document, path=path)
    return documents


def _compact_document(
    document: Mapping[object, object],
    *,
    path: str,
) -> Mapping[str, object]:
    compact: dict[str, object] = {"path": path}
    language = document.get("language")
    if isinstance(language, str) and language:
        compact["language"] = language
    compact["items"] = _compact_items(document.get("items"))
    return compact


def _compact_items(raw_items: object) -> list[Mapping[str, object]]:
    if not isinstance(raw_items, Sequence) or isinstance(raw_items, (str, bytes)):
        return []
    items: list[Mapping[str, object]] = []
    for raw_item in raw_items:
        if not isinstance(raw_item, Mapping):
            continue
        item: dict[str, object] = {}
        for key in _ITEM_KEYS:
            value = raw_item.get(key)
            if value is not None:
                item[key] = value
        members = _compact_items(raw_item.get("members"))
        if members:
            item["members"] = members
        if item.get("name"):
            items.append(item)
    return items


def _scope_paths(connection: sqlite3.Connection, target: str) -> tuple[str, ...]:
    prefix = f"{target.rstrip('/')}/*"
    rows = connection.execute(
        "SELECT path FROM repository_documents WHERE path = ? OR path GLOB ?",
        (target, prefix),
    )
    return tuple(str(row[0]) for row in rows)


def _delete_scope(connection: sqlite3.Connection, target: str) -> None:
    prefix = f"{target.rstrip('/')}/*"
    connection.execute(
        "DELETE FROM repository_documents WHERE path = ? OR path GLOB ?",
        (target, prefix),
    )


def _all_paths(connection: sqlite3.Connection) -> tuple[str, ...]:
    rows = connection.execute("SELECT path FROM repository_documents ORDER BY path")
    return tuple(str(row[0]) for row in rows)


def _normalize(path: str) -> str:
    return os.path.normpath(path or ".")


def _normalize_target(root: str, target: str) -> str:
    if os.path.isabs(target):
        return _normalize(target)
    return _normalize(os.path.join(root, target))


__all__ = [
    "RepositoryIndexWorkerError",
    "load_repository_documents",
    "update_repository_index",
]
