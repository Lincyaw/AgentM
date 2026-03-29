"""Memory Vault MCP Server — expose vault tools over MCP protocol.

Thin MCP wrappers that delegate to the shared tool implementations
in ``tools.py`` via ``create_vault_tools()``.
"""

import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from mcp.server.fastmcp import Context, FastMCP

from agentm.tools.vault.store import MarkdownVault
from agentm.tools.vault.tools import create_vault_tools


@dataclass
class VaultContext:
    """Holds the long-lived MarkdownVault instance and pre-built tool closures."""

    vault: MarkdownVault
    tools: dict[str, Any] = field(default_factory=dict)


@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[VaultContext]:
    """Initialize vault on startup, close on shutdown."""
    vault_dir = os.environ.get("VAULT_DIR", "./vault")
    embedding_model = os.environ.get("VAULT_EMBEDDING_MODEL") or None

    embed_fn = None
    if embedding_model:
        try:
            # Suppress stderr during model loading to avoid interfering
            # with MCP stdio transport (progress bars, warnings, etc.)
            _stderr = sys.stderr
            sys.stderr = open(os.devnull, "w")  # noqa: SIM115, PTH123
            try:
                from sentence_transformers import SentenceTransformer

                model = SentenceTransformer(embedding_model)
                embed_fn = lambda text: model.encode(text).tolist()  # noqa: E731
            finally:
                sys.stderr.close()
                sys.stderr = _stderr
        except Exception:
            pass

    vault = MarkdownVault(vault_dir, embedding_model=embedding_model, embed_fn=embed_fn)
    tools = create_vault_tools(vault)

    yield VaultContext(vault=vault, tools=tools)


mcp = FastMCP("Memory Vault", lifespan=lifespan)


def _tools(ctx: Context | None) -> dict[str, Any]:
    """Extract the pre-built tools dict from the MCP lifespan context."""
    if ctx is None:
        raise ValueError("MCP context is required but was not provided")
    return ctx.request_context.lifespan_context.tools


# ---------------------------------------------------------------------------
# CRUD Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def vault_write(
    path: str,
    frontmatter: dict,
    body: str,
    ctx: Context | None = None,
) -> str:
    """Create or overwrite a note in the vault.

    Args:
        path: Note path relative to vault root (e.g. "skill/timeout-diagnosis")
        frontmatter: YAML metadata dict. Required fields: type, confidence.
        body: Markdown body content. Should start with "# Title".

    Returns warnings if the note has structural issues.
    """
    return _tools(ctx)["vault_write"](
        [{"path": path, "frontmatter": frontmatter, "body": body}]
    )


@mcp.tool()
def vault_write_batch(
    entries: list[dict],
    ctx: Context | None = None,
) -> str:
    """Write multiple notes atomically in a single transaction.

    Args:
        entries: List of dicts, each with "path", "frontmatter", "body" keys.

    All notes are written together — if one fails, none are committed.
    """
    return _tools(ctx)["vault_write"](entries)


@mcp.tool()
def vault_read(path: str, ctx: Context | None = None) -> str:
    """Read a note from the vault. Returns its frontmatter and body.

    Args:
        path: Note path (e.g. "concept/connection-pooling")
    """
    return _tools(ctx)["vault_read"](path)


@mcp.tool()
def vault_edit(
    path: str,
    operation: str,
    params: dict,
    ctx: Context | None = None,
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
    return _tools(ctx)["vault_edit"](path, operation, params)


@mcp.tool()
def vault_delete(path: str, ctx: Context | None = None) -> str:
    """Delete a note and remove it from all indexes.

    Args:
        path: Note path to delete.
    """
    return _tools(ctx)["vault_delete"](path)


@mcp.tool()
def vault_rename(old_path: str, new_path: str, ctx: Context | None = None) -> str:
    """Rename or move a note. All [[backlinks]] in other notes are rewritten automatically.

    Args:
        old_path: Current note path.
        new_path: New note path.
    """
    return _tools(ctx)["vault_rename"](old_path, new_path)


# ---------------------------------------------------------------------------
# Browse & Search Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def vault_list(
    path: str,
    depth: int = 1,
    type_filter: str | None = None,
    ctx: Context | None = None,
) -> str:
    """Browse notes under a path.

    Args:
        path: Directory path (e.g. "skill"). Use "" for vault root.
        depth: How many levels deep to list (default 1).
        type_filter: Only return notes of this type (e.g. "skill", "concept").
    """
    return _tools(ctx)["vault_list"](path, depth=depth, type_filter=type_filter or "")


@mcp.tool()
def vault_search(
    query: str,
    filters: dict | None = None,
    mode: str = "hybrid",
    limit: int = 10,
    ctx: Context | None = None,
) -> str:
    """Search the vault by keyword, semantic similarity, or both.

    Args:
        query: Search query text.
        filters: Optional filters: {"type": "skill", "confidence": "fact", "tags": ["db"]}.
        mode: "keyword" (FTS5), "semantic" (vector), or "hybrid" (both, default).
        limit: Max results (default 10).

    Returns rich results with path, score, title, type, confidence, tags, and snippet.
    """
    return _tools(ctx)["vault_search"](query, filters=filters or {}, mode=mode, limit=limit)


@mcp.tool()
def vault_backlinks(path: str, ctx: Context | None = None) -> str:
    """Find all notes that contain a [[wikilink]] to the given path.

    Args:
        path: Target note path.
    """
    return _tools(ctx)["vault_backlinks"](path)


@mcp.tool()
def vault_traverse(
    start: str,
    depth: int = 2,
    direction: str = "both",
    ctx: Context | None = None,
) -> str:
    """Explore the link graph around a note via BFS traversal.

    Args:
        start: Starting note path.
        depth: How many hops to follow (default 2).
        direction: "forward" (outgoing links), "backward" (incoming), or "both".

    Returns nodes with depth and directed edges (source -> target).
    """
    return _tools(ctx)["vault_traverse"](start, depth=depth, direction=direction)


@mcp.tool()
def vault_lint(request: str, ctx: Context | None = None) -> str:
    """Check vault health: find dead [[wikilinks]] and orphan notes with no connections.

    Args:
        request: A short description of what you want to check (e.g. "check all links").
    """
    return _tools(ctx)["vault_lint"](request)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
