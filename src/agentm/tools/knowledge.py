"""Knowledge Store — file-system backend with inverted index + hybrid search.

Layout:
    <base_dir>/
      <category>/
        <slug>.json      # entry content
        <slug>.emb        # embedding cache (JSON array of floats)

Path mapping:
    "/failure-patterns/db-connection-exhaustion"
    -> <base_dir>/failure-patterns/db-connection-exhaustion.json
"""

from __future__ import annotations

import builtins
import json
import math
import re
import threading
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Module-level shared state (embedding model is a singleton resource)
# ---------------------------------------------------------------------------

_model: Any = None
_model_load_failed: bool = False


def _get_model() -> Any | None:
    """Lazy-load the sentence-transformers model.  Returns None on failure."""
    global _model, _model_load_failed
    if _model is not None:
        return _model
    if _model_load_failed:
        return None
    try:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer("jinaai/jina-embeddings-v5-text-nano")
        return _model
    except Exception:
        _model_load_failed = True
        return None


# ---------------------------------------------------------------------------
# Pure helpers (no state)
# ---------------------------------------------------------------------------

_SPLIT_RE = re.compile(r"[\s\-_/.,;:!?()]+")


def _tokenize(text: str) -> set[str]:
    """Lowercase split on whitespace / punctuation."""
    return {t for t in _SPLIT_RE.split(text.lower()) if t}


def _cosine_similarity(a: builtins.list[float], b: builtins.list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _path_to_fs(path: str) -> tuple[str, str]:
    """Split ``/category/slug`` into ``("category", "slug")``.

    Raises ``ValueError`` if the path does not contain at least two segments.
    """
    parts = [p for p in path.strip("/").split("/") if p]
    if len(parts) < 2:
        raise ValueError(
            f"Path must have at least two segments (/<category>/<slug>), got: {path!r}"
        )
    category = "/".join(parts[:-1])
    slug = parts[-1]
    return category, slug


def _indexable_tokens(entry: dict) -> set[str]:
    """Tokens from title + description + tags."""
    parts: builtins.list[str] = []
    if "title" in entry:
        parts.append(str(entry["title"]))
    if "description" in entry:
        parts.append(str(entry["description"]))
    for tag in entry.get("tags", []):
        parts.append(str(tag))
    return _tokenize(" ".join(parts))


def _entry_text(entry: dict) -> str:
    return (entry.get("title", "") + " " + entry.get("description", "")).strip()


def _build_inverted_index(
    entries: dict[str, dict],
) -> dict[str, set[str]]:
    index: dict[str, set[str]] = {}
    for path, entry in entries.items():
        tokens = _indexable_tokens(entry)
        for tok in tokens:
            index.setdefault(tok, set()).add(path)
    return index


# ---------------------------------------------------------------------------
# KnowledgeStore class
# ---------------------------------------------------------------------------


class KnowledgeStore:
    """Encapsulates all knowledge-store state and operations."""

    def __init__(self, base_dir: str = "./knowledge") -> None:
        """Initialise the knowledge store rooted at *base_dir*.

        * Creates the directory if it does not exist.
        * Scans for existing ``<category>/<slug>.json`` files and loads them
          into the in-memory cache.
        * Builds the inverted index.
        * Loads cached embeddings from ``.emb`` sidecar files.
        * Migrates ``knowledge.json`` (legacy flat-file format) if present.
        """
        self._base_dir: Path = Path(base_dir)
        self._entries: dict[str, dict] = {}
        self._inv_index: dict[str, set[str]] = {}
        self._embeddings: dict[str, builtins.list[float]] = {}
        self._lock: threading.Lock = threading.Lock()

        self._base_dir.mkdir(parents=True, exist_ok=True)

        # Migrate legacy knowledge.json if present and store is otherwise empty
        self._migrate_legacy_json()

        # Scan existing files
        for json_file in self._base_dir.rglob("*.json"):
            rel = json_file.relative_to(self._base_dir)
            parts = rel.with_suffix("").parts  # ("category", "slug")
            if len(parts) < 1:
                continue
            path = "/" + "/".join(parts)
            try:
                entry = json.loads(json_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            self._entries[path] = entry

            # Load cached embedding
            emb_file = json_file.with_suffix(".emb")
            if emb_file.exists():
                try:
                    self._embeddings[path] = json.loads(
                        emb_file.read_text(encoding="utf-8")
                    )
                except (json.JSONDecodeError, OSError):
                    pass

        self._inv_index = _build_inverted_index(self._entries)

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def write(self, path: str, entry: dict[str, Any]) -> str:
        """Create or overwrite a knowledge entry at *path*."""
        category, slug = _path_to_fs(path)
        fs_path = self._base_dir / category / f"{slug}.json"

        with self._lock:
            fs_path.parent.mkdir(parents=True, exist_ok=True)
            fs_path.write_text(
                json.dumps(entry, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            self._entries[path] = entry
            self._update_inverted_index(path, entry)
            self._update_embedding(path, entry)

        return f"Written: {path}"

    def read(self, path: str) -> str:
        """Read a specific knowledge entry by its full path."""
        entry = self._entries.get(path)
        if entry is None:
            return json.dumps({"error": f"Not found: {path}"})
        return json.dumps(entry, default=str)

    def delete(self, path: str) -> str:
        """Delete a knowledge entry."""
        category, slug = _path_to_fs(path)
        fs_path = self._base_dir / category / f"{slug}.json"
        emb_path = fs_path.with_suffix(".emb")

        with self._lock:
            if fs_path.exists():
                fs_path.unlink()
            if emb_path.exists():
                emb_path.unlink()
            self._entries.pop(path, None)
            self._embeddings.pop(path, None)
            # Rebuild inverted index (simpler than surgical removal)
            self._rebuild_inverted_index()

        return f"Deleted: {path}"

    def list(  # noqa: A003
        self, request: str, path: str = "/", depth: int = 1
    ) -> str:
        """List the structure of the knowledge base at *path*.

        *depth* controls how many levels to expand:
        - ``1`` (default): immediate children only
        - ``0``: unlimited depth
        - ``N``: up to N levels deep
        """
        _ = request
        prefix = path.rstrip("/")
        if prefix == "":
            prefix = ""

        sub_dirs: set[str] = set()
        entries_at_path: builtins.list[dict[str, Any]] = []

        for entry_path, entry in self._entries.items():
            if prefix and not entry_path.startswith(prefix + "/"):
                continue
            if not prefix:
                # root listing -- everything matches
                pass

            # Calculate relative depth
            if prefix:
                relative = entry_path[len(prefix) + 1 :]  # strip prefix + "/"
            else:
                relative = entry_path.lstrip("/")

            parts = relative.split("/")
            entry_depth = len(parts)

            if depth > 0 and entry_depth > depth:
                # Only record the sub-directory at the allowed depth
                sub_dir = prefix + "/" + "/".join(parts[:depth])
                sub_dirs.add(sub_dir)
                continue

            # Collect sub-directories from intermediary path segments
            if len(parts) > 1:
                sub_dir = prefix + "/" + parts[0]
                sub_dirs.add(sub_dir)

            # Direct entries at depth boundary or within range
            if depth == 0 or entry_depth <= depth:
                entries_at_path.append(
                    {"path": entry_path, "title": entry.get("title", "")}
                )

        result = {
            "path": path,
            "sub_paths": sorted(sub_dirs),
            "entries": entries_at_path,
        }
        return json.dumps(result, default=str)

    def search(
        self,
        query: str,
        path: str = "/",
        limit: int = 5,
        mode: str = "hybrid",
    ) -> str:
        """Search the knowledge base.

        *mode*: ``"keyword"`` | ``"semantic"`` | ``"hybrid"`` (default).
        """
        if mode == "keyword":
            results = self._keyword_search(query, path, limit)
        elif mode == "semantic":
            results = self._semantic_search(query, path, limit)
        else:
            results = self._hybrid_search(query, path, limit)

        if not results:
            return "No results found."

        entries: builtins.list[dict[str, Any]] = []
        for entry_path, score in results:
            entry = self._entries.get(entry_path, {})
            entries.append(
                {"path": entry_path, "value": entry, "score": round(score, 4)}
            )
        return json.dumps(entries, default=str)

    # -------------------------------------------------------------------
    # Inverted index (private)
    # -------------------------------------------------------------------

    def _update_inverted_index(self, path: str, entry: dict) -> None:
        """Add/replace tokens for *path* in the inverted index."""
        # Remove old tokens
        for tok, paths in builtins.list(self._inv_index.items()):
            paths.discard(path)
            if not paths:
                del self._inv_index[tok]
        # Add new tokens
        for tok in _indexable_tokens(entry):
            self._inv_index.setdefault(tok, set()).add(path)

    def _rebuild_inverted_index(self) -> None:
        self._inv_index = _build_inverted_index(self._entries)

    # -------------------------------------------------------------------
    # Keyword search (private)
    # -------------------------------------------------------------------

    def _keyword_search(
        self, query: str, path: str, limit: int
    ) -> builtins.list[tuple[str, float]]:
        """AND-semantics keyword search with prefix/substring matching."""
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        prefix = path.rstrip("/")

        candidates: dict[str, int] = {}
        for qt in query_tokens:
            # Find all index tokens that contain qt as prefix or substring
            matched_paths: set[str] = set()
            for idx_token, paths in self._inv_index.items():
                if qt in idx_token:
                    matched_paths |= paths
            for p in matched_paths:
                if prefix and not p.startswith(prefix + "/") and p != prefix:
                    continue
                if not prefix or p.startswith(prefix + "/") or prefix == "":
                    candidates[p] = candidates.get(p, 0) + 1

        total = len(query_tokens)
        # AND semantics: all query tokens must match
        scored = [
            (p, count / total) for p, count in candidates.items() if count == total
        ]
        scored.sort(key=lambda x: -x[1])
        return scored[:limit]

    # -------------------------------------------------------------------
    # Embedding / semantic search (private)
    # -------------------------------------------------------------------

    def _compute_embedding(self, text: str) -> builtins.list[float] | None:
        model = _get_model()
        if model is None:
            return None
        vec = model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def _update_embedding(self, path: str, entry: dict) -> None:
        """Compute and cache embedding for an entry."""
        text = _entry_text(entry)
        if not text:
            return
        vec = self._compute_embedding(text)
        if vec is None:
            return
        self._embeddings[path] = vec
        # Write sidecar
        category, slug = _path_to_fs(path)
        emb_path = self._base_dir / category / f"{slug}.emb"
        try:
            emb_path.write_text(json.dumps(vec), encoding="utf-8")
        except OSError:
            pass

    def _semantic_search(
        self, query: str, path: str, limit: int
    ) -> builtins.list[tuple[str, float]]:
        """Cosine similarity search using cached embeddings."""
        q_vec = self._compute_embedding(query)
        if q_vec is None:
            # Model unavailable -- fall back to keyword
            return self._keyword_search(query, path, limit)

        prefix = path.rstrip("/")
        scored: builtins.list[tuple[str, float]] = []
        for p, vec in self._embeddings.items():
            if prefix and not p.startswith(prefix + "/") and p != prefix:
                continue
            sim = _cosine_similarity(q_vec, vec)
            scored.append((p, sim))

        scored.sort(key=lambda x: -x[1])
        return scored[:limit]

    # -------------------------------------------------------------------
    # Hybrid search (private)
    # -------------------------------------------------------------------

    def _hybrid_search(
        self, query: str, path: str, limit: int
    ) -> builtins.list[tuple[str, float]]:
        """Reciprocal Rank Fusion of keyword + semantic results."""
        kw_results = self._keyword_search(query, path, limit * 2)
        sem_results = self._semantic_search(query, path, limit * 2)

        k = 60  # RRF constant
        rrf_scores: dict[str, float] = {}

        for rank, (p, _score) in enumerate(kw_results):
            rrf_scores[p] = rrf_scores.get(p, 0.0) + 1.0 / (k + rank)

        for rank, (p, _score) in enumerate(sem_results):
            rrf_scores[p] = rrf_scores.get(p, 0.0) + 1.0 / (k + rank)

        ranked = sorted(rrf_scores.items(), key=lambda x: -x[1])
        return ranked[:limit]

    # -------------------------------------------------------------------
    # Legacy migration (private)
    # -------------------------------------------------------------------

    def _migrate_legacy_json(self) -> None:
        """One-time migration from flat ``knowledge.json`` to per-entry files."""
        legacy = self._base_dir / "knowledge.json"
        if not legacy.exists():
            return

        # Only migrate if directory is otherwise empty (no json files yet)
        existing_jsons = builtins.list(self._base_dir.rglob("*.json"))
        if len(existing_jsons) > 1:
            # Other json files exist besides knowledge.json -- skip migration
            return

        try:
            data = json.loads(legacy.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return

        if not isinstance(data, builtins.list):
            return

        for item in data:
            if not isinstance(item, dict):
                continue
            path = item.get("path", "")
            if not path:
                continue
            try:
                category, slug = _path_to_fs(path)
            except ValueError:
                continue
            fs_path = self._base_dir / category / f"{slug}.json"
            fs_path.parent.mkdir(parents=True, exist_ok=True)
            # Strip the 'path' field from the stored entry
            entry = {k: v for k, v in item.items() if k != "path"}
            fs_path.write_text(
                json.dumps(entry, indent=2, ensure_ascii=False), encoding="utf-8"
            )

        # Rename to mark as migrated
        legacy.rename(self._base_dir / "knowledge.json.migrated")


# ---------------------------------------------------------------------------
# Module-level backward-compatible API
# ---------------------------------------------------------------------------

_default_store: KnowledgeStore | None = None

# Legacy module-level attributes -- kept so that external code and tests
# that directly read/write ``knowledge_module._base_dir`` etc. continue
# to work.  Setting ``_base_dir = None`` (the test teardown pattern)
# signals "not initialized".
_base_dir: Path | None = None
_entries: dict[str, dict] = {}
_inv_index: dict[str, set[str]] = {}
_embeddings: dict[str, builtins.list[float]] = {}


def init(base_dir: str = "./knowledge") -> None:
    """Initialize the default module-level knowledge store."""
    global _default_store, _base_dir, _entries, _inv_index, _embeddings
    _default_store = KnowledgeStore(base_dir)
    # Mirror instance state onto module-level names for backward compat
    _base_dir = _default_store._base_dir
    _entries = _default_store._entries
    _inv_index = _default_store._inv_index
    _embeddings = _default_store._embeddings


def _ensure_init() -> None:
    if _base_dir is None:
        raise RuntimeError(
            "Knowledge store not initialized \u2014 call init() first"
        )


def _get_default() -> KnowledgeStore:
    _ensure_init()
    assert _default_store is not None  # guaranteed by _ensure_init
    return _default_store


def knowledge_write(path: str, entry: dict[str, Any]) -> str:
    """Create or overwrite a knowledge entry at *path*."""
    return _get_default().write(path, entry)


def knowledge_read(path: str) -> str:
    """Read a specific knowledge entry by its full path."""
    return _get_default().read(path)


def knowledge_delete(path: str) -> str:
    """Delete a knowledge entry."""
    return _get_default().delete(path)


def knowledge_list(request: str, path: str = "/", depth: int = 1) -> str:
    """List the structure of the knowledge base at *path*."""
    return _get_default().list(request, path, depth)


def knowledge_search(
    query: str,
    path: str = "/",
    limit: int = 5,
    mode: str = "hybrid",
) -> str:
    """Search the knowledge base."""
    return _get_default().search(query, path, limit, mode)
