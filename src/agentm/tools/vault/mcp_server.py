"""Memory Vault MCP Server — expose vault tools over MCP protocol."""

import json
import os
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass
from typing import Any

from mcp.server.fastmcp import Context, FastMCP

from agentm.tools.vault.graph import get_backlinks, lint, traverse
from agentm.tools.vault.search import hybrid_search, keyword_search, semantic_search
from agentm.tools.vault.store import MarkdownVault


@dataclass
class VaultContext:
    """Holds the long-lived MarkdownVault instance."""

    vault: MarkdownVault
    embed_fn: Any = None


@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[VaultContext]:
    """Initialize vault on startup, close on shutdown."""
    vault_dir = os.environ.get("VAULT_DIR", "./vault")
    embedding_model = os.environ.get("VAULT_EMBEDDING_MODEL") or None
    vault = MarkdownVault(vault_dir, embedding_model=embedding_model)

    embed_fn = None
    if embedding_model:
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(embedding_model)
            embed_fn = lambda text: model.encode(text).tolist()  # noqa: E731
        except Exception:
            pass

    yield VaultContext(vault=vault, embed_fn=embed_fn)


mcp = FastMCP("Memory Vault", lifespan=lifespan)


def _vc(ctx: Context) -> VaultContext:
    return ctx.request_context.lifespan_context


# ---------------------------------------------------------------------------
# CRUD Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def vault_write(
    path: str,
    frontmatter: dict,
    body: str,
    ctx: Context = None,
) -> str:
    """Create or overwrite a note in the vault.

    Args:
        path: Note path relative to vault root (e.g. "skill/timeout-diagnosis")
        frontmatter: YAML metadata dict. Required fields: type, confidence.
        body: Markdown body content. Should start with "# Title".

    Returns warnings if the note has structural issues.
    """
    vc = _vc(ctx)
    warnings = vc.vault.write(path, frontmatter, body)
    return json.dumps({"status": "ok", "path": path, "warnings": warnings}, ensure_ascii=False)


@mcp.tool()
def vault_write_batch(
    entries: list[dict],
    ctx: Context = None,
) -> str:
    """Write multiple notes atomically in a single transaction.

    Args:
        entries: List of dicts, each with "path", "frontmatter", "body" keys.

    All notes are written together — if one fails, none are committed.
    """
    vc = _vc(ctx)
    warnings = vc.vault.write_batch(entries)
    return json.dumps(
        {"status": "ok", "count": len(entries), "warnings": warnings},
        ensure_ascii=False,
    )


@mcp.tool()
def vault_read(path: str, ctx: Context = None) -> str:
    """Read a note from the vault. Returns its frontmatter and body.

    Args:
        path: Note path (e.g. "concept/connection-pooling")
    """
    vc = _vc(ctx)
    result = vc.vault.read(path)
    if result is None:
        return json.dumps({"error": f"Note not found: {path}"})
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def vault_edit(
    path: str,
    operation: str,
    params: dict,
    ctx: Context = None,
) -> str:
    """Apply an incremental edit to an existing note without rewriting the full content.

    Args:
        path: Note path to edit.
        operation: One of "replace_string", "set_frontmatter", "replace_section", "append_section".
        params: Operation-specific parameters:
            - replace_string: {"old": "...", "new": "..."}
            - set_frontmatter: {"field": "value", ...} (merge update)
            - replace_section: {"heading": "## Section", "body": "new content"}
            - append_section: {"heading": "## Section", "content": "appended text"}

    Returns warnings if the edit introduces structural issues.
    """
    vc = _vc(ctx)
    try:
        warnings = vc.vault.edit(path, operation, params)
        return json.dumps(
            {"status": "ok", "path": path, "operation": operation, "warnings": warnings},
            ensure_ascii=False,
        )
    except (FileNotFoundError, ValueError) as exc:
        return json.dumps({"error": str(exc)}, ensure_ascii=False)


@mcp.tool()
def vault_delete(path: str, ctx: Context = None) -> str:
    """Delete a note and remove it from all indexes.

    Args:
        path: Note path to delete.
    """
    vc = _vc(ctx)
    try:
        vc.vault.delete(path)
        return json.dumps({"status": "ok", "path": path}, ensure_ascii=False)
    except FileNotFoundError as exc:
        return json.dumps({"error": str(exc)}, ensure_ascii=False)


@mcp.tool()
def vault_rename(old_path: str, new_path: str, ctx: Context = None) -> str:
    """Rename or move a note. All [[backlinks]] in other notes are rewritten automatically.

    Args:
        old_path: Current note path.
        new_path: New note path.
    """
    vc = _vc(ctx)
    try:
        vc.vault.rename(old_path, new_path)
        return json.dumps(
            {"status": "ok", "old_path": old_path, "new_path": new_path},
            ensure_ascii=False,
        )
    except FileNotFoundError as exc:
        return json.dumps({"error": str(exc)}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Browse & Search Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def vault_list(
    path: str,
    depth: int = 1,
    type_filter: str | None = None,
    ctx: Context = None,
) -> str:
    """Browse notes under a path.

    Args:
        path: Directory path (e.g. "skill"). Use "" for vault root.
        depth: How many levels deep to list (default 1).
        type_filter: Only return notes of this type (e.g. "skill", "concept").
    """
    vc = _vc(ctx)
    notes = vc.vault.list_notes(path, depth=depth, type_filter=type_filter)
    return json.dumps({"notes": notes}, ensure_ascii=False)


@mcp.tool()
def vault_search(
    query: str,
    filters: dict | None = None,
    mode: str = "hybrid",
    limit: int = 10,
    ctx: Context = None,
) -> str:
    """Search the vault by keyword, semantic similarity, or both.

    Args:
        query: Search query text.
        filters: Optional filters: {"type": "skill", "confidence": "fact", "tags": ["db"]}.
        mode: "keyword" (FTS5), "semantic" (vector), or "hybrid" (both, default).
        limit: Max results (default 10).

    Returns rich results with path, score, title, type, confidence, tags, and snippet.
    """
    vc = _vc(ctx)
    conn = vc.vault._get_conn()
    f = filters or {}

    if mode == "keyword":
        hits = keyword_search(conn, query, f, limit)
    elif mode == "semantic":
        if vc.embed_fn is None:
            return json.dumps({"error": "No embedding model configured; use mode='keyword'"})
        hits = semantic_search(conn, query, vc.embed_fn, f, limit)
    elif mode == "hybrid":
        if vc.embed_fn is None:
            hits = keyword_search(conn, query, f, limit)
        else:
            hits = hybrid_search(conn, query, vc.embed_fn, f, limit)
    else:
        return json.dumps({"error": f"Unknown search mode: {mode}"})

    results = [asdict(h) for h in hits]
    return json.dumps({"results": results}, ensure_ascii=False)


@mcp.tool()
def vault_backlinks(path: str, ctx: Context = None) -> str:
    """Find all notes that contain a [[wikilink]] to the given path.

    Args:
        path: Target note path.
    """
    vc = _vc(ctx)
    conn = vc.vault._get_conn()
    links = get_backlinks(conn, path)
    return json.dumps({"backlinks": links}, ensure_ascii=False)


@mcp.tool()
def vault_traverse(
    start: str,
    depth: int = 2,
    direction: str = "both",
    ctx: Context = None,
) -> str:
    """Explore the link graph around a note via BFS traversal.

    Args:
        start: Starting note path.
        depth: How many hops to follow (default 2).
        direction: "forward" (outgoing links), "backward" (incoming), or "both".

    Returns nodes with depth and directed edges (source → target).
    """
    vc = _vc(ctx)
    conn = vc.vault._get_conn()
    result = traverse(conn, start, depth, direction)
    return json.dumps(
        {
            "start": result.start,
            "nodes": [asdict(n) for n in result.nodes],
            "edges": [asdict(e) for e in result.edges],
        },
        ensure_ascii=False,
    )


@mcp.tool()
def vault_lint(ctx: Context = None) -> str:
    """Check vault health: find dead [[wikilinks]] and orphan notes with no connections."""
    vc = _vc(ctx)
    conn = vc.vault._get_conn()
    result = lint(conn)
    return json.dumps(
        {
            "dead_links": [list(pair) for pair in result.dead_links],
            "orphan_notes": result.orphan_notes,
        },
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
