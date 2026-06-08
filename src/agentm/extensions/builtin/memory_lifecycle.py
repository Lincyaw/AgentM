"""Builtin ``memory_lifecycle`` atom -- session lifecycle hooks for memory.

Bridges the gap between the ``memory`` atom (which stores typed frontmatter
files under ``.agentm/memory/``) and the session lifecycle, providing:

1. **Auto-extract** (``TurnEndEvent``, POST priority) -- regex heuristics
   detect memory-worthy facts in user messages (preferences, identity,
   corrections, decisions, project facts). Writes through
   ``ResourceWriter`` with no LLM call, keeping cost zero for most turns.

2. **Pre-compact sink** (``BeforeCompactEvent``) -- before compaction
   discards messages, re-scan the buffer for uncaptured facts using the
   same heuristics. Safety net so auto-extract misses don't become
   permanent losses.

3. **Per-turn prefetch** (``ContextEvent``, NORMAL priority) -- substring
   search against memory file names + descriptions, inject matching bodies
   into the last user message so the LLM sees relevant past facts without
   calling ``memory_read``.

The atom is purely event-driven -- no tools, invisible to the user/LLM.
Reads go through ``api.get_operations().file``; writes go through
``api.get_resource_writer()`` (matching the ``memory`` atom's contract).
S11-compliant: no ``core.runtime.*`` or atom-to-atom imports.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any, Final

from agentm.core.abi import TextContent, UserMessage
from agentm.core.abi.events import (
    BeforeCompactEvent,
    BusPriority,
    ContextEvent,
    TurnEndEvent,
)
from agentm.core.abi.extension import ExtensionAPI
from agentm.core.lib.frontmatter import parse_frontmatter
from agentm.extensions import ExtensionManifest

logger = logging.getLogger(__name__)

MANIFEST = ExtensionManifest(
    name="memory_lifecycle",
    description=(
        "Session lifecycle hooks for memory: auto-extract on turn end, "
        "pre-compact sink, per-turn prefetch injection. Zero LLM cost — "
        "regex heuristics only."
    ),
    registers=(
        "event:turn_end",
        "event:before_compact",
        "event:context",
    ),
    requires=(),
)


# ---------------------------------------------------------------------------
# Heuristic patterns (inspired by Hermes holographic plugin)
# ---------------------------------------------------------------------------

_MIN_MSG_LEN: Final[int] = 30
_MAX_EXTRACT_CHARS: Final[int] = 400
_MAX_PREFETCH_RESULTS: Final[int] = 3
_MAX_PREFETCH_BODY_CHARS: Final[int] = 600
_MAX_PREFETCH_CACHE: Final[int] = 64

_PATTERNS: Final[list[tuple[str, re.Pattern[str]]]] = [
    # Durable preferences (not transient "I want X done")
    ("user", re.compile(
        r"\b(?:I|my)\s+(?:always|never|prefer(?:red)?|favorite)\b",
        re.IGNORECASE,
    )),
    # User identity
    ("user", re.compile(
        r"\b(?:my\s+name\s+is|I\s+am\s+a\s+\w+|I\s+work\s+(?:at|as|in)\s+\w+)\b",
        re.IGNORECASE,
    )),
    # Explicit memory requests (EN)
    ("feedback", re.compile(
        r"\b(?:remember\s+(?:that|this)|don'?t\s+(?:ever|again)|please\s+(?:always|never)\s+\w+)\b",
        re.IGNORECASE,
    )),
    # Explicit memory requests (ZH)
    ("feedback", re.compile(
        r"(?:记住|请记得|不要再|以后都|以后不要)",
    )),
    # Team/project decisions
    ("project", re.compile(
        r"\b(?:we\s+(?:decided|agreed|settled)\s+(?:to|on)\s+"
        r"|the\s+project\s+(?:uses|requires)\s+)",
        re.IGNORECASE,
    )),
]


def _extract_user_text(messages: tuple[Any, ...] | list[Any]) -> str | None:
    """Return the text of the last user message, or None."""
    for msg in reversed(messages):
        if isinstance(msg, UserMessage):
            parts: list[str] = []
            for block in msg.content:
                if isinstance(block, TextContent):
                    parts.append(block.text)
            text = " ".join(parts).strip()
            return text if text else None
    return None


def _slugify(text: str, max_words: int = 4) -> str:
    """Turn the first few words of ``text`` into a kebab-case slug."""
    words = re.findall(r"[A-Za-z0-9一-鿿]+", text)
    slug = "-".join(w.lower() for w in words[:max_words])
    return slug[:40] if slug else "note"


def _match_patterns(text: str) -> str | None:
    """Return the memory type if any heuristic pattern fires, else None."""
    for mem_type, pattern in _PATTERNS:
        if pattern.search(text):
            return mem_type
    return None


def _serialize_memory(mem_type: str, name: str, description: str, content: str) -> str:
    safe_desc = description.replace("\\", "\\\\").replace('"', '\\"')
    body = content if content.endswith("\n") else content + "\n"
    return (
        "---\n"
        f"name: {name}\n"
        f'description: "{safe_desc}"\n'
        f"type: {mem_type}\n"
        "---\n\n"
        f"{body}"
    )


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    base_path = _resolve_base(api.cwd, config.get("path", ".agentm/memory"))
    cwd_path = Path(api.cwd).resolve()
    file_ops = api.get_operations().file
    writer = api.get_resource_writer()

    extracted_names: set[str] = set()
    prefetch_cache: dict[str, str] = {}
    index_state: dict[str, Any] = {"version": 0, "entries": []}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _to_cwd_relative(path: Path) -> str:
        try:
            return str(path.resolve().relative_to(cwd_path))
        except ValueError:
            return str(path)

    async def _list_memory_files() -> list[Path]:
        try:
            names = await file_ops.list_dir(str(base_path))
        except Exception:
            return []
        out: list[Path] = []
        for entry in names:
            if not entry.endswith(".md") or entry == "MEMORY.md":
                continue
            out.append(base_path / entry)
        return sorted(out)

    async def _rebuild_index() -> None:
        """Regenerate MEMORY.md from current memory files."""
        entries: list[tuple[str, str, str]] = []
        for path in await _list_memory_files():
            try:
                data = await file_ops.read_file(str(path))
            except Exception:
                continue
            meta, _body = parse_frontmatter(data.decode("utf-8", errors="replace"))
            name = str(meta.get("name", path.stem))
            mem_type = str(meta.get("type", ""))
            description = str(meta.get("description", ""))
            entries.append((name, mem_type, description))
        entries.sort(key=lambda row: row[0])
        lines = [
            f"- [{mem_type}/{name}] {desc}"
            for name, mem_type, desc in entries
        ]
        body = "\n".join(lines) + ("\n" if lines else "")
        rel = _to_cwd_relative(base_path / "MEMORY.md")
        try:
            await writer.write(rel, body.encode("utf-8"), rationale="memory_lifecycle_index_rebuild")
        except Exception:
            pass

    async def _load_index_entries() -> list[tuple[str, str, str, Path]]:
        """Return [(name, type, description, path), ...] from memory files."""
        version = index_state["version"]
        if index_state["entries"] and index_state.get("cached_version") == version:
            return index_state["entries"]
        entries: list[tuple[str, str, str, Path]] = []
        for path in await _list_memory_files():
            try:
                data = await file_ops.read_file(str(path))
            except Exception:
                continue
            meta, _body = parse_frontmatter(data.decode("utf-8", errors="replace"))
            name = str(meta.get("name", path.stem))
            mem_type = str(meta.get("type", ""))
            description = str(meta.get("description", ""))
            entries.append((name, mem_type, description, path))
        index_state["entries"] = entries
        index_state["cached_version"] = version
        return entries

    async def _write_memory(
        mem_type: str, name: str, description: str, content: str
    ) -> bool:
        """Write a memory file and rebuild the index. Returns True on success."""
        filename = f"{mem_type}_{name}.md"
        filepath = base_path / filename
        body = _serialize_memory(mem_type, name, description, content)
        rel = _to_cwd_relative(filepath)
        try:
            result = await writer.write(rel, body.encode("utf-8"), rationale="memory_lifecycle_auto_extract")
            if getattr(result, "error", None) is not None:
                logger.debug("memory_lifecycle: write failed: %s", result.error)
                return False
        except Exception:
            logger.debug("memory_lifecycle: failed to write %s", filepath)
            return False
        index_state["version"] += 1
        prefetch_cache.clear()
        await _rebuild_index()
        return True

    async def _extract_from_text(user_text: str) -> None:
        """Run heuristic extraction on a single user message text."""
        if len(user_text) < _MIN_MSG_LEN:
            return
        mem_type = _match_patterns(user_text)
        if mem_type is None:
            return

        slug = _slugify(user_text)
        ts = str(time.time_ns())
        name = f"auto-{slug}-{ts}"
        name = re.sub(r"[^A-Za-z0-9_一-鿿-]", "-", name)

        if name in extracted_names:
            return
        extracted_names.add(name)

        truncated = user_text[:_MAX_EXTRACT_CHARS]
        description = truncated[:80].replace("\n", " ").strip()
        await _write_memory(mem_type, name, description, truncated)

    # ------------------------------------------------------------------
    # 1. Auto-extract on TurnEndEvent (POST priority)
    # ------------------------------------------------------------------

    async def _on_turn_end(event: TurnEndEvent) -> None:
        user_text = _extract_user_text(event.messages)
        if user_text is None:
            return
        await _extract_from_text(user_text)

    api.on(TurnEndEvent.CHANNEL, _on_turn_end, priority=BusPriority.POST)

    # ------------------------------------------------------------------
    # 2. Pre-compact memory sink (BeforeCompactEvent)
    # ------------------------------------------------------------------

    async def _on_before_compact(event: BeforeCompactEvent) -> None:
        for msg in event.messages:
            if not isinstance(msg, UserMessage):
                continue
            parts: list[str] = []
            for block in msg.content:
                if isinstance(block, TextContent):
                    parts.append(block.text)
            text = " ".join(parts).strip()
            if not text:
                continue
            await _extract_from_text(text)

    api.on(BeforeCompactEvent.CHANNEL, _on_before_compact)

    # ------------------------------------------------------------------
    # 3. Per-turn memory prefetch (ContextEvent, NORMAL priority)
    # ------------------------------------------------------------------

    async def _on_context(event: ContextEvent) -> None:
        user_text = _extract_user_text(event.messages)
        if not user_text or len(user_text) < _MIN_MSG_LEN:
            return

        cache_key = str(hash(user_text))
        if len(prefetch_cache) > _MAX_PREFETCH_CACHE:
            prefetch_cache.clear()
        cached = prefetch_cache.get(cache_key)
        if cached is not None:
            if cached:
                _inject_into_last_user(event.messages, cached)
            return

        # Split into keywords, search memory entries
        keywords = _extract_keywords(user_text)
        if not keywords:
            prefetch_cache[cache_key] = ""
            return

        entries = await _load_index_entries()
        matches: list[tuple[str, str, Path]] = []
        for name, mem_type, description, path in entries:
            haystack = f"{name} {description}".lower()
            if any(kw in haystack for kw in keywords):
                matches.append((name, mem_type, path))
                if len(matches) >= _MAX_PREFETCH_RESULTS:
                    break

        if not matches:
            prefetch_cache[cache_key] = ""
            return

        # Read matching memory bodies
        sections: list[str] = []
        for name, mem_type, path in matches:
            try:
                data = await file_ops.read_file(str(path))
            except Exception:
                continue
            _meta, body = parse_frontmatter(data.decode("utf-8", errors="replace"))
            body = body.strip()
            if not body:
                continue
            if len(body) > _MAX_PREFETCH_BODY_CHARS:
                body = body[:_MAX_PREFETCH_BODY_CHARS] + "..."
            sections.append(f"[{mem_type}/{name}] {body}")

        if not sections:
            prefetch_cache[cache_key] = ""
            return

        block_text = (
            "<recalled_memory>\n"
            + "\n\n".join(sections)
            + "\n</recalled_memory>"
        )
        prefetch_cache[cache_key] = block_text
        _inject_into_last_user(event.messages, block_text)

    api.on(ContextEvent.CHANNEL, _on_context, priority=BusPriority.NORMAL)


# ---------------------------------------------------------------------------
# Module-level helpers (not closures -- stateless)
# ---------------------------------------------------------------------------


def _resolve_base(cwd: str, raw_path: str | None) -> Path:
    if raw_path is None:
        raw_path = ".agentm/memory"
    raw = Path(raw_path).expanduser()
    return raw if raw.is_absolute() else (Path(cwd) / raw).resolve()


def _extract_keywords(text: str) -> list[str]:
    """Extract lowercase keywords (3+ chars) from text for search."""
    # Split on whitespace and punctuation, keep words 3+ chars
    words = re.findall(r"[A-Za-z0-9一-鿿]{3,}", text.lower())
    # Deduplicate preserving order, skip very common words
    stopwords = frozenset({
        "the", "and", "for", "are", "but", "not", "you", "all",
        "can", "her", "was", "one", "our", "out", "has", "have",
        "this", "that", "with", "from", "they", "been", "said",
        "will", "each", "make", "like", "how", "what", "when",
        "where", "which", "their", "would", "there", "about",
    })
    seen: set[str] = set()
    result: list[str] = []
    for w in words:
        if w in stopwords or w in seen:
            continue
        seen.add(w)
        result.append(w)
    return result[:8]


def _inject_into_last_user(messages: list[Any], text: str) -> None:
    """Append a recalled_memory text block to the last user message.

    Per cache discipline: append to the last message's tail, never the
    system prompt, to preserve KV/prefix cache.
    """
    for msg in reversed(messages):
        if isinstance(msg, UserMessage):
            msg.content.append(TextContent(type="text", text=text))
            return
