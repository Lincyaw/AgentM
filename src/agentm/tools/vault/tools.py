"""Vault tool functions -- factory creating 10 closures over a MarkdownVault."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from agentm.tools.vault.graph import get_backlinks, lint, traverse
from agentm.tools.vault.search import hybrid_search, keyword_search, semantic_search
from agentm.tools.vault.store import MarkdownVault


def _ok(**kwargs: Any) -> str:
    return json.dumps({"status": "ok", **kwargs}, ensure_ascii=False)


def _err(msg: str) -> str:
    return json.dumps({"error": msg}, ensure_ascii=False)


def create_vault_tools(vault: MarkdownVault) -> dict[str, Any]:
    """Create vault tool functions with closure over vault instance.

    Returns a dict mapping tool name to callable.  Every callable returns
    a JSON string and never raises -- errors are encoded as ``{"error": ...}``.
    """

    # ------------------------------------------------------------------
    # 1. vault_write
    # ------------------------------------------------------------------

    def vault_write(
        path: str | None = None,
        frontmatter: dict | None = None,
        body: str | None = None,
        entries: list[dict] | None = None,
    ) -> str:
        """Create or overwrite one or more notes in the vault.

        Single mode: provide path, frontmatter, body.
        Batch mode: provide entries (list of dicts with path/frontmatter/body keys).
        Returns warnings if the note has structural issues (missing fields, dead links).
        """
        try:
            if entries is not None:
                warnings = vault.write_batch(entries)
                return _ok(count=len(entries), warnings=warnings)
            if path is not None:
                warnings = vault.write(path, frontmatter or {}, body or "")
                return _ok(path=path, warnings=warnings)
            return _err("Provide (path, frontmatter, body) or entries=list")
        except Exception as exc:
            return _err(str(exc))

    # ------------------------------------------------------------------
    # 2. vault_read
    # ------------------------------------------------------------------

    def vault_read(path: str) -> str:
        """Read a note from the vault. Returns frontmatter and body."""
        try:
            result = vault.read(path)
            if result is None:
                return _err(f"Note not found: {path}")
            return json.dumps(result, ensure_ascii=False)
        except Exception as exc:
            return _err(str(exc))

    # ------------------------------------------------------------------
    # 3. vault_edit
    # ------------------------------------------------------------------

    def vault_edit(path: str, operation: str, params: dict) -> str:
        """Apply an incremental edit to an existing note.

        Operations: replace_string, set_frontmatter, replace_section, append_section.
        Returns warnings if the edit introduces structural issues.
        """
        try:
            warnings = vault.edit(path, operation, params)
            return _ok(path=path, operation=operation, warnings=warnings)
        except Exception as exc:
            return _err(str(exc))

    # ------------------------------------------------------------------
    # 4. vault_delete
    # ------------------------------------------------------------------

    def vault_delete(path: str) -> str:
        """Delete a note and remove it from the index."""
        try:
            vault.delete(path)
            return _ok(path=path)
        except Exception as exc:
            return _err(str(exc))

    # ------------------------------------------------------------------
    # 5. vault_rename
    # ------------------------------------------------------------------

    def vault_rename(old_path: str, new_path: str) -> str:
        """Rename or move a note. Backlinks in other notes are rewritten automatically."""
        try:
            vault.rename(old_path, new_path)
            return _ok(old_path=old_path, new_path=new_path)
        except Exception as exc:
            return _err(str(exc))

    # ------------------------------------------------------------------
    # 6. vault_list
    # ------------------------------------------------------------------

    def vault_list(
        path: str,
        depth: int = 1,
        type_filter: str | None = None,
    ) -> str:
        """Browse notes under a path. Use path="" for root."""
        try:
            notes = vault.list_notes(path, depth=depth, type_filter=type_filter)
            return json.dumps({"notes": notes}, ensure_ascii=False)
        except Exception as exc:
            return _err(str(exc))

    # ------------------------------------------------------------------
    # 7. vault_search
    # ------------------------------------------------------------------

    def vault_search(
        query: str,
        filters: dict | None = None,
        mode: str = "hybrid",
        limit: int = 10,
    ) -> str:
        """Search the vault. Modes: keyword, semantic, hybrid.

        Filters: {type, confidence, status, tags} narrow results.
        """
        try:
            conn = vault._get_conn()
            f = filters or {}

            if mode == "keyword":
                hits = keyword_search(conn, query, f, limit)
            elif mode == "semantic":
                if vault._embedding_model is None:
                    return _err("No embedding model configured; use mode='keyword'")
                # Build embedding function from model name
                hits = semantic_search(conn, query, _get_embed_fn(vault), f, limit)
            elif mode == "hybrid":
                if vault._embedding_model is None:
                    # Fallback to keyword-only when no embedding
                    hits = keyword_search(conn, query, f, limit)
                else:
                    hits = hybrid_search(conn, query, _get_embed_fn(vault), f, limit)
            else:
                return _err(f"Unknown search mode: {mode}")

            results = [asdict(h) for h in hits]
            return json.dumps({"results": results}, ensure_ascii=False)
        except Exception as exc:
            return _err(str(exc))

    # ------------------------------------------------------------------
    # 8. vault_backlinks
    # ------------------------------------------------------------------

    def vault_backlinks(path: str) -> str:
        """Find all notes that link to the given path."""
        try:
            conn = vault._get_conn()
            links = get_backlinks(conn, path)
            return json.dumps({"backlinks": links}, ensure_ascii=False)
        except Exception as exc:
            return _err(str(exc))

    # ------------------------------------------------------------------
    # 9. vault_traverse
    # ------------------------------------------------------------------

    def vault_traverse(
        start: str,
        depth: int = 2,
        direction: str = "both",
    ) -> str:
        """Explore the link graph around a note via BFS.

        Direction: forward (outgoing), backward (incoming), both.
        """
        try:
            conn = vault._get_conn()
            result = traverse(conn, start, depth, direction)
            return json.dumps(
                {
                    "start": result.start,
                    "nodes": [asdict(n) for n in result.nodes],
                    "edges": [asdict(e) for e in result.edges],
                },
                ensure_ascii=False,
            )
        except Exception as exc:
            return _err(str(exc))

    # ------------------------------------------------------------------
    # 10. vault_lint
    # ------------------------------------------------------------------

    def vault_lint(request: str) -> str:
        """Check vault health: find dead links and orphan notes.

        Pass any value for request (e.g. "check").
        """
        try:
            conn = vault._get_conn()
            result = lint(conn)
            return json.dumps(
                {
                    "dead_links": [list(pair) for pair in result.dead_links],
                    "orphan_notes": result.orphan_notes,
                },
                ensure_ascii=False,
            )
        except Exception as exc:
            return _err(str(exc))

    return {
        "vault_write": vault_write,
        "vault_read": vault_read,
        "vault_edit": vault_edit,
        "vault_delete": vault_delete,
        "vault_rename": vault_rename,
        "vault_list": vault_list,
        "vault_search": vault_search,
        "vault_backlinks": vault_backlinks,
        "vault_traverse": vault_traverse,
        "vault_lint": vault_lint,
    }


def _get_embed_fn(vault: MarkdownVault):
    """Lazily build an embedding function from the vault's model name."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(vault._embedding_model)
    return lambda text: model.encode(text).tolist()
