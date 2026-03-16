"""MarkdownVault -- CRUD operations with atomic writes and transactional index."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import struct
import threading
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import yaml

from agentm.tools.vault.parser import (
    append_to_section,
    extract_title,
    extract_wikilinks,
    parse_note,
    replace_section,
    replace_string,
    serialize_note,
)
from agentm.tools.vault.schema import create_schema

_REQUIRED_FRONTMATTER = {"type", "confidence"}
_VALID_TYPES = {"skill", "episodic", "concept", "failure-pattern", "system-knowledge"}
_VALID_CONFIDENCE = {"fact", "pattern", "heuristic"}
_VALID_STATUS = {"active", "superseded", "archived"}


class MarkdownVault:
    """Markdown + YAML frontmatter knowledge store backed by SQLite index.

    All write operations are atomic (tmp file + ``os.replace``) and protected
    by a threading lock.  Index updates happen inside a single SQLite
    transaction.
    """

    def __init__(
        self,
        vault_dir: str | Path,
        embedding_model: str | None = None,
        embed_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._vault_dir = Path(vault_dir)
        self._vault_dir.mkdir(parents=True, exist_ok=True)
        self._embedding_model = embedding_model
        self._embed_fn = embed_fn
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        # Eagerly create schema so the DB file exists immediately.
        create_schema(self._get_conn())

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            db_path = self._vault_dir / ".vault.db"
            self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    @staticmethod
    def _compute_body_hash(body: str) -> str:
        return hashlib.sha256(body.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Shared index helper
    # ------------------------------------------------------------------

    def _index_note(
        self, conn: sqlite3.Connection, path: str, frontmatter: dict, body: str
    ) -> None:
        """UPSERT notes, DELETE+INSERT links/tags/fts in an open transaction."""
        title = extract_title(body)
        body_hash = self._compute_body_hash(body)
        now = datetime.now(timezone.utc).isoformat()

        fm_type = frontmatter.get("type", "")
        confidence = frontmatter.get("confidence", "")
        status = frontmatter.get("status", "active")
        created_at = frontmatter.get("created", now)
        updated_at = frontmatter.get("updated", now)
        fm_json = json.dumps(frontmatter, ensure_ascii=False)

        conn.execute(
            "INSERT INTO notes (path, type, title, confidence, status, created_at, updated_at, frontmatter, body_hash) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(path) DO UPDATE SET "
            "type=excluded.type, title=excluded.title, confidence=excluded.confidence, "
            "status=excluded.status, updated_at=excluded.updated_at, "
            "frontmatter=excluded.frontmatter, body_hash=excluded.body_hash",
            (path, fm_type, title, confidence, status, created_at, updated_at, fm_json, body_hash),
        )

        # Links
        conn.execute("DELETE FROM links WHERE source = ?", (path,))
        wikilinks = extract_wikilinks(body)
        for target in wikilinks:
            conn.execute(
                "INSERT OR IGNORE INTO links (source, target) VALUES (?, ?)",
                (path, target),
            )

        # Tags
        conn.execute("DELETE FROM tags WHERE path = ?", (path,))
        for tag in frontmatter.get("tags", []):
            conn.execute(
                "INSERT OR IGNORE INTO tags (path, tag) VALUES (?, ?)",
                (path, tag),
            )

        # FTS
        conn.execute("DELETE FROM notes_fts WHERE path = ?", (path,))
        tags_str = " ".join(frontmatter.get("tags", []))
        conn.execute(
            "INSERT INTO notes_fts (path, title, body, tags) VALUES (?, ?, ?, ?)",
            (path, title, body, tags_str),
        )

        # Embedding (if embed_fn configured)
        if self._embed_fn is not None:
            self._upsert_embedding(conn, path, body)

    def _remove_index(self, conn: sqlite3.Connection, path: str) -> None:
        """Remove all index entries for *path*."""
        conn.execute("DELETE FROM notes WHERE path = ?", (path,))
        conn.execute("DELETE FROM links WHERE source = ?", (path,))
        conn.execute("DELETE FROM tags WHERE path = ?", (path,))
        conn.execute("DELETE FROM notes_fts WHERE path = ?", (path,))
        self._remove_embedding(conn, path)

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def _upsert_embedding(
        self, conn: sqlite3.Connection, path: str, body: str
    ) -> None:
        """Compute and store embedding for a note."""
        assert self._embed_fn is not None
        vec = self._embed_fn(body)
        dim = len(vec)
        blob = struct.pack(f"{dim}f", *vec)

        # Try notes_vec (sqlite-vec) first — create lazily with correct dim
        from agentm.tools.vault.schema import has_vec_support

        if has_vec_support(conn):
            has_vec = conn.execute(
                "SELECT name FROM sqlite_master WHERE name='notes_vec'"
            ).fetchone()
            if not has_vec:
                conn.executescript(
                    f"CREATE VIRTUAL TABLE IF NOT EXISTS notes_vec USING vec0("
                    f"path TEXT PRIMARY KEY, embedding FLOAT[{dim}]);"
                )
            self._remove_embedding(conn, path)
            conn.execute(
                "INSERT INTO notes_vec (path, embedding) VALUES (?, ?)",
                (path, blob),
            )
        else:
            # Fallback: note_embeddings table
            conn.execute(
                "CREATE TABLE IF NOT EXISTS note_embeddings "
                "(path TEXT PRIMARY KEY, embedding BLOB)"
            )
            conn.execute(
                "INSERT OR REPLACE INTO note_embeddings (path, embedding) VALUES (?, ?)",
                (path, blob),
            )

    def _remove_embedding(self, conn: sqlite3.Connection, path: str) -> None:
        """Remove stored embedding for a note."""
        has_vec = conn.execute(
            "SELECT name FROM sqlite_master WHERE name='notes_vec'"
        ).fetchone()
        if has_vec:
            conn.execute("DELETE FROM notes_vec WHERE path = ?", (path,))
        has_emb = conn.execute(
            "SELECT name FROM sqlite_master WHERE name='note_embeddings'"
        ).fetchone()
        if has_emb:
            conn.execute("DELETE FROM note_embeddings WHERE path = ?", (path,))

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_note(
        self, path: str, frontmatter: dict, body: str
    ) -> list[str]:
        """Check a note for structural issues. Returns a list of warnings."""
        warnings: list[str] = []

        # Required frontmatter fields
        for field in _REQUIRED_FRONTMATTER:
            if field not in frontmatter:
                warnings.append(f"Missing required frontmatter field: '{field}'")

        # Validate field values
        fm_type = frontmatter.get("type", "")
        if fm_type and fm_type not in _VALID_TYPES:
            warnings.append(
                f"Unknown type '{fm_type}'. Expected one of: {', '.join(sorted(_VALID_TYPES))}"
            )

        confidence = frontmatter.get("confidence", "")
        if confidence and confidence not in _VALID_CONFIDENCE:
            warnings.append(
                f"Unknown confidence '{confidence}'. Expected one of: {', '.join(sorted(_VALID_CONFIDENCE))}"
            )

        status = frontmatter.get("status", "")
        if status and status not in _VALID_STATUS:
            warnings.append(
                f"Unknown status '{status}'. Expected one of: {', '.join(sorted(_VALID_STATUS))}"
            )

        # Check title exists
        title = extract_title(body)
        if not title:
            warnings.append("Body is missing an H1 heading (# Title)")

        # Check for dead wikilinks
        conn = self._get_conn()
        wikilinks = extract_wikilinks(body)
        for target in wikilinks:
            row = conn.execute(
                "SELECT 1 FROM notes WHERE path = ?", (target,)
            ).fetchone()
            if row is None:
                # Also check if file exists on disk (might be a new note not yet indexed)
                if not self._resolve(target).exists():
                    warnings.append(f"Dead link: [[{target}]] (target does not exist)")

        return warnings

    # ------------------------------------------------------------------
    # File helpers
    # ------------------------------------------------------------------

    def _resolve(self, path: str) -> Path:
        """Convert vault path to absolute filesystem path."""
        return self._vault_dir / (path + ".md")

    def _atomic_write(self, filepath: Path, content: str) -> None:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        tmp = filepath.with_suffix(".md.tmp")
        tmp.write_text(content, encoding="utf-8")
        os.replace(str(tmp), str(filepath))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write(self, path: str, frontmatter: dict, body: str) -> list[str]:
        """Create or overwrite a note. Returns list of validation warnings."""
        content = serialize_note(frontmatter, body)
        with self._lock:
            self._atomic_write(self._resolve(path), content)
            conn = self._get_conn()
            self._index_note(conn, path, frontmatter, body)
            conn.commit()
            return self._validate_note(path, frontmatter, body)

    def write_batch(self, entries: list[dict]) -> list[str]:
        """Write multiple notes in a single atomic transaction. Returns warnings."""
        if not entries:
            return []
        all_warnings: list[str] = []
        with self._lock:
            # Write all files first
            for entry in entries:
                content = serialize_note(entry["frontmatter"], entry["body"])
                self._atomic_write(self._resolve(entry["path"]), content)
            # Index all in one transaction
            conn = self._get_conn()
            for entry in entries:
                self._index_note(conn, entry["path"], entry["frontmatter"], entry["body"])
            conn.commit()
            # Validate after all are indexed (so cross-references can resolve)
            for entry in entries:
                ws = self._validate_note(entry["path"], entry["frontmatter"], entry["body"])
                for w in ws:
                    all_warnings.append(f"{entry['path']}: {w}")
        return all_warnings

    def read(self, path: str) -> dict | None:
        """Read a note, returning ``{"path", "frontmatter", "body"}`` or None."""
        filepath = self._resolve(path)
        if not filepath.exists():
            return None
        content = filepath.read_text(encoding="utf-8")
        frontmatter, body = parse_note(content)
        return {"path": path, "frontmatter": frontmatter, "body": body}

    def edit(self, path: str, operation: str, params: dict) -> list[str]:
        """Apply an incremental edit and re-index. Returns validation warnings."""
        filepath = self._resolve(path)
        if not filepath.exists():
            msg = f"Note not found: {path}"
            raise FileNotFoundError(msg)

        content = filepath.read_text(encoding="utf-8")
        frontmatter, body = parse_note(content)

        if operation == "replace_string":
            body = replace_string(body, params["old"], params["new"])
        elif operation == "set_frontmatter":
            frontmatter = {**frontmatter, **params}
        elif operation == "replace_section":
            body = replace_section(body, params["heading"], params["body"])
        elif operation == "append_section":
            body = append_to_section(body, params["heading"], params["content"])
        else:
            msg = f"Unknown edit operation: {operation!r}"
            raise ValueError(msg)

        new_content = serialize_note(frontmatter, body)
        with self._lock:
            self._atomic_write(filepath, new_content)
            conn = self._get_conn()
            self._index_note(conn, path, frontmatter, body)
            conn.commit()
            return self._validate_note(path, frontmatter, body)

    def delete(self, path: str) -> None:
        """Delete a note file and all its index entries."""
        filepath = self._resolve(path)
        if not filepath.exists():
            msg = f"Note not found: {path}"
            raise FileNotFoundError(msg)

        with self._lock:
            filepath.unlink()
            conn = self._get_conn()
            self._remove_index(conn, path)
            conn.commit()

    def rename(self, old_path: str, new_path: str) -> None:
        """Move a note and rewrite ``[[old_path]]`` backlinks in all referencing files."""
        old_file = self._resolve(old_path)
        if not old_file.exists():
            msg = f"Note not found: {old_path}"
            raise FileNotFoundError(msg)

        with self._lock:
            conn = self._get_conn()

            # 1. Find all files that reference old_path
            refs = conn.execute(
                "SELECT source FROM links WHERE target = ?", (old_path,)
            ).fetchall()

            # 2. Rewrite backlinks in referencing files
            for (source,) in refs:
                if source == old_path:
                    continue  # will be handled when we move the file itself
                ref_file = self._resolve(source)
                if ref_file.exists():
                    ref_content = ref_file.read_text(encoding="utf-8")
                    updated = ref_content.replace(f"[[{old_path}]]", f"[[{new_path}]]")
                    self._atomic_write(ref_file, updated)
                    # Re-index the referencing file
                    fm, bd = parse_note(updated)
                    self._index_note(conn, source, fm, bd)

            # 3. Move the file itself
            new_file = self._resolve(new_path)
            new_file.parent.mkdir(parents=True, exist_ok=True)
            # Also rewrite self-references in the moved file
            content = old_file.read_text(encoding="utf-8")
            content = content.replace(f"[[{old_path}]]", f"[[{new_path}]]")
            self._atomic_write(new_file, content)
            old_file.unlink()

            # 4. Update index: remove old, add new
            self._remove_index(conn, old_path)
            fm, bd = parse_note(content)
            self._index_note(conn, new_path, fm, bd)

            conn.commit()

    def list_notes(
        self, path: str = "", depth: int = 1, type_filter: str | None = None
    ) -> list[dict]:
        """List notes under *path* up to *depth* levels deep."""
        conn = self._get_conn()
        query = "SELECT path, type, title FROM notes"
        conditions: list[str] = []
        params: list[str | int] = []

        if path:
            conditions.append("path LIKE ?")
            params.append(f"{path}/%")
        if type_filter:
            conditions.append("type = ?")
            params.append(type_filter)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        rows = conn.execute(query, params).fetchall()

        results = []
        for row_path, row_type, row_title in rows:
            # Compute depth relative to the prefix
            if path:
                relative = row_path[len(path) + 1 :]  # strip prefix + "/"
            else:
                relative = row_path

            # Count slashes to determine depth
            parts = relative.split("/")
            if len(parts) <= depth:
                results.append({"path": row_path, "type": row_type, "title": row_title})

        return results

    def rebuild_index(self) -> int:
        """Scan all ``.md`` files and rebuild the entire index. Returns count."""
        with self._lock:
            conn = self._get_conn()
            # Clear all data tables
            for table in ("notes", "notes_fts", "links", "tags"):
                conn.execute(f"DELETE FROM {table}")  # noqa: S608

            count = 0
            for md_file in self._vault_dir.rglob("*.md"):
                # Skip non-vault files
                rel = md_file.relative_to(self._vault_dir)
                path = str(rel.with_suffix(""))
                content = md_file.read_text(encoding="utf-8")
                fm, body = parse_note(content)
                self._index_note(conn, path, fm, body)
                count += 1

            conn.commit()
        return count
