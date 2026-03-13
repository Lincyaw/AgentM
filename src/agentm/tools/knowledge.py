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

import json
import math
import re
import threading
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_base_dir: Path | None = None
_entries: dict[str, dict] = {}  # path -> entry dict (in-memory cache)
_inv_index: dict[str, set[str]] = {}  # token -> set of paths
_embeddings: dict[str, list[float]] = {}  # path -> embedding vector
_model: Any = None  # lazy-loaded SentenceTransformer
_model_load_failed: bool = False
_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def init(base_dir: str = "./knowledge") -> None:
    """Initialise the knowledge store rooted at *base_dir*.

    * Creates the directory if it does not exist.
    * Scans for existing ``<category>/<slug>.json`` files and loads them
      into the in-memory cache.
    * Builds the inverted index.
    * Loads cached embeddings from ``.emb`` sidecar files.
    * Migrates ``knowledge.json`` (legacy flat-file format) if present.
    """
    global _base_dir, _entries, _inv_index, _embeddings

    _base_dir = Path(base_dir)
    _base_dir.mkdir(parents=True, exist_ok=True)

    _entries = {}
    _inv_index = {}
    _embeddings = {}

    # Migrate legacy knowledge.json if present and store is otherwise empty
    _migrate_legacy_json()

    # Scan existing files
    for json_file in _base_dir.rglob("*.json"):
        rel = json_file.relative_to(_base_dir)
        parts = rel.with_suffix("").parts  # ("category", "slug")
        if len(parts) < 1:
            continue
        path = "/" + "/".join(parts)
        try:
            entry = json.loads(json_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        _entries[path] = entry

        # Load cached embedding
        emb_file = json_file.with_suffix(".emb")
        if emb_file.exists():
            try:
                _embeddings[path] = json.loads(emb_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass

    _inv_index = _build_inverted_index(_entries)


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------


def knowledge_write(path: str, entry: dict[str, Any]) -> str:
    """Create or overwrite a knowledge entry at *path*."""
    _ensure_init()
    category, slug = _path_to_fs(path)
    fs_path = _base_dir / category / f"{slug}.json"

    with _lock:
        fs_path.parent.mkdir(parents=True, exist_ok=True)
        fs_path.write_text(
            json.dumps(entry, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        _entries[path] = entry
        _update_inverted_index(path, entry)
        _update_embedding(path, entry)

    return f"Written: {path}"


def knowledge_read(path: str) -> str:
    """Read a specific knowledge entry by its full path."""
    _ensure_init()
    entry = _entries.get(path)
    if entry is None:
        return json.dumps({"error": f"Not found: {path}"})
    return json.dumps(entry, default=str)


def knowledge_delete(path: str) -> str:
    """Delete a knowledge entry."""
    _ensure_init()
    category, slug = _path_to_fs(path)
    fs_path = _base_dir / category / f"{slug}.json"
    emb_path = fs_path.with_suffix(".emb")

    with _lock:
        if fs_path.exists():
            fs_path.unlink()
        if emb_path.exists():
            emb_path.unlink()
        _entries.pop(path, None)
        _embeddings.pop(path, None)
        # Rebuild inverted index (simpler than surgical removal)
        _rebuild_inverted_index()

    return f"Deleted: {path}"


def knowledge_list(request: str, path: str = "/", depth: int = 1) -> str:
    """List the structure of the knowledge base at *path*.

    *depth* controls how many levels to expand:
    - ``1`` (default): immediate children only
    - ``0``: unlimited depth
    - ``N``: up to N levels deep
    """
    _ = request
    _ensure_init()
    prefix = path.rstrip("/")
    if prefix == "":
        prefix = ""

    sub_dirs: set[str] = set()
    entries_at_path: list[dict[str, Any]] = []

    for entry_path, entry in _entries.items():
        if prefix and not entry_path.startswith(prefix + "/"):
            continue
        if not prefix:
            # root listing — everything matches
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


def knowledge_search(
    query: str,
    path: str = "/",
    limit: int = 5,
    mode: str = "hybrid",
) -> str:
    """Search the knowledge base.

    *mode*: ``"keyword"`` | ``"semantic"`` | ``"hybrid"`` (default).
    """
    _ensure_init()
    if mode == "keyword":
        results = _keyword_search(query, path, limit)
    elif mode == "semantic":
        results = _semantic_search(query, path, limit)
    else:
        results = _hybrid_search(query, path, limit)

    if not results:
        return "No results found."

    entries = []
    for entry_path, score in results:
        entry = _entries.get(entry_path, {})
        entries.append({"path": entry_path, "value": entry, "score": round(score, 4)})
    return json.dumps(entries, default=str)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


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


def _ensure_init() -> None:
    if _base_dir is None:
        raise RuntimeError("Knowledge store not initialized — call init() first")


# ---------------------------------------------------------------------------
# Tokenisation & inverted index
# ---------------------------------------------------------------------------

_SPLIT_RE = re.compile(r"[\s\-_/.,;:!?()]+")


def _tokenize(text: str) -> set[str]:
    """Lowercase split on whitespace / punctuation."""
    return {t for t in _SPLIT_RE.split(text.lower()) if t}


def _build_inverted_index(entries: dict[str, dict]) -> dict[str, set[str]]:
    index: dict[str, set[str]] = {}
    for path, entry in entries.items():
        tokens = _indexable_tokens(entry)
        for tok in tokens:
            index.setdefault(tok, set()).add(path)
    return index


def _indexable_tokens(entry: dict) -> set[str]:
    """Tokens from title + description + tags."""
    parts: list[str] = []
    if "title" in entry:
        parts.append(str(entry["title"]))
    if "description" in entry:
        parts.append(str(entry["description"]))
    for tag in entry.get("tags", []):
        parts.append(str(tag))
    return _tokenize(" ".join(parts))


def _update_inverted_index(path: str, entry: dict) -> None:
    """Add/replace tokens for *path* in the global inverted index."""
    # Remove old tokens
    for tok, paths in list(_inv_index.items()):
        paths.discard(path)
        if not paths:
            del _inv_index[tok]
    # Add new tokens
    for tok in _indexable_tokens(entry):
        _inv_index.setdefault(tok, set()).add(path)


def _rebuild_inverted_index() -> None:
    global _inv_index
    _inv_index = _build_inverted_index(_entries)


# ---------------------------------------------------------------------------
# Keyword search
# ---------------------------------------------------------------------------


def _keyword_search(
    query: str, path: str, limit: int
) -> list[tuple[str, float]]:
    """AND-semantics keyword search with prefix/substring matching."""
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    prefix = path.rstrip("/")

    candidates: dict[str, int] = {}
    for qt in query_tokens:
        # Find all index tokens that contain qt as prefix or substring
        matched_paths: set[str] = set()
        for idx_token, paths in _inv_index.items():
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


# ---------------------------------------------------------------------------
# Embedding / semantic search
# ---------------------------------------------------------------------------


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


def _compute_embedding(text: str) -> list[float] | None:
    model = _get_model()
    if model is None:
        return None
    vec = model.encode(text, normalize_embeddings=True)
    return vec.tolist()


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _entry_text(entry: dict) -> str:
    return (entry.get("title", "") + " " + entry.get("description", "")).strip()


def _update_embedding(path: str, entry: dict) -> None:
    """Compute and cache embedding for an entry."""
    text = _entry_text(entry)
    if not text:
        return
    vec = _compute_embedding(text)
    if vec is None:
        return
    _embeddings[path] = vec
    # Write sidecar
    category, slug = _path_to_fs(path)
    emb_path = _base_dir / category / f"{slug}.emb"
    try:
        emb_path.write_text(json.dumps(vec), encoding="utf-8")
    except OSError:
        pass


def _semantic_search(
    query: str, path: str, limit: int
) -> list[tuple[str, float]]:
    """Cosine similarity search using cached embeddings."""
    q_vec = _compute_embedding(query)
    if q_vec is None:
        # Model unavailable — fall back to keyword
        return _keyword_search(query, path, limit)

    prefix = path.rstrip("/")
    scored: list[tuple[str, float]] = []
    for p, vec in _embeddings.items():
        if prefix and not p.startswith(prefix + "/") and p != prefix:
            continue
        sim = _cosine_similarity(q_vec, vec)
        scored.append((p, sim))

    scored.sort(key=lambda x: -x[1])
    return scored[:limit]


# ---------------------------------------------------------------------------
# Hybrid search (RRF fusion)
# ---------------------------------------------------------------------------


def _hybrid_search(
    query: str, path: str, limit: int
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion of keyword + semantic results."""
    kw_results = _keyword_search(query, path, limit * 2)
    sem_results = _semantic_search(query, path, limit * 2)

    k = 60  # RRF constant
    rrf_scores: dict[str, float] = {}

    for rank, (p, _score) in enumerate(kw_results):
        rrf_scores[p] = rrf_scores.get(p, 0.0) + 1.0 / (k + rank)

    for rank, (p, _score) in enumerate(sem_results):
        rrf_scores[p] = rrf_scores.get(p, 0.0) + 1.0 / (k + rank)

    ranked = sorted(rrf_scores.items(), key=lambda x: -x[1])
    return ranked[:limit]


# ---------------------------------------------------------------------------
# Legacy migration
# ---------------------------------------------------------------------------


def _migrate_legacy_json() -> None:
    """One-time migration from flat ``knowledge.json`` to per-entry files."""
    legacy = _base_dir / "knowledge.json"
    if not legacy.exists():
        return

    # Only migrate if directory is otherwise empty (no json files yet)
    existing_jsons = list(_base_dir.rglob("*.json"))
    if len(existing_jsons) > 1:
        # Other json files exist besides knowledge.json — skip migration
        return

    try:
        data = json.loads(legacy.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return

    if not isinstance(data, list):
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
        fs_path = _base_dir / category / f"{slug}.json"
        fs_path.parent.mkdir(parents=True, exist_ok=True)
        # Strip the 'path' field from the stored entry
        entry = {k: v for k, v in item.items() if k != "path"}
        fs_path.write_text(
            json.dumps(entry, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # Rename to mark as migrated
    legacy.rename(_base_dir / "knowledge.json.migrated")
