"""Contrib ``memory_lifecycle`` atom — session lifecycle hooks for memory.

Bridges the gap between the ``memory`` atom (which stores typed frontmatter
files under ``.agentm/memory/``) and the session lifecycle, providing:

1. **Auto-extract** (``TurnEndEvent``, POST priority) — regex heuristics
   detect memory-worthy facts in user messages (preferences, identity,
   corrections, decisions, project facts). Writes directly through
   ``file_ops`` with no LLM call, keeping cost zero for most turns.

2. **Pre-compact sink** (``BeforeCompactEvent``) — before compaction
   discards messages, re-scan the buffer for uncaptured facts using the
   same heuristics. Safety net so auto-extract misses don't become
   permanent losses.

3. **Per-turn prefetch** (``ContextEvent``, NORMAL priority) — substring
   search against memory file names + descriptions, inject matching bodies
   into the last user message so the LLM sees relevant past facts without
   calling ``memory_read``.

The atom is purely event-driven — no tools, invisible to the user/LLM.
File I/O goes through ``api.get_operations().file`` (write) and
``api.get_operations().file`` (read).  S11-compliant: no ``core.runtime.*``
or atom-to-atom imports.
"""

from __future__ import annotations

import asyncio
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

_MIN_MSG_LEN: Final[int] = 15
_MAX_EXTRACT_CHARS: Final[int] = 400
_MAX_PREFETCH_RESULTS: Final[int] = 3
_MAX_PREFETCH_BODY_CHARS: Final[int] = 600

_PATTERNS: Final[list[tuple[str, re.Pattern[str]]]] = [
    # User preferences
    ("user", re.compile(
        r"(?:I|i)\s+(?:prefer|like|use|want|need|always|usually)\s+",
        re.IGNORECASE,
    )),
    # User identity
    ("user", re.compile(
        r"(?:my\s+name\s+is|I\s+am\s+a|I\s+work\s+(?:at|on|as|in))\s+",
        re.IGNORECASE,
    )),
    # Corrections / feedback (EN)
    ("feedback", re.compile(
        r"(?:don'?t\s+do\s+that|remember\s+that|please\s+(?:always|never))\b",
        re.IGNORECASE,
    )),
    # Corrections / feedback (ZH)
    ("feedback", re.compile(
        r"(?:不要|记住|请记得|请不要|以后)",
    )),
    # Decisions
    ("project", re.compile(
        r"(?:we|I)\s+(?:decided|agreed|chose|will\s+use|settled\s+on)\s+",
        re.IGNORECASE,
    )),
    # Project facts
    ("project", re.compile(
        r"(?:the\s+project|this\s+project|our\s+project|the\s+codebase)"
        r"\s+(?:uses?|needs?|requires?|depends?\s+on)\s+",
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
    body = content if content.endswith("\n") else content + "\n"
    return (
        "---\n"
        f"name: {name}\n"
        f"description: {description}\n"
        f"type: {mem_type}\n"
        "---\n\n"
        f"{body}"
    )


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    base_path = _resolve_base(api.cwd, config.get("path", ".agentm/memory"))
    file_ops = api.get_operations().file

    # Session-scoped dedup set: names already extracted this session.
    extracted_names: set[str] = set()

    # Prefetch cache: keyed by hash of last user message text.
    prefetch_cache: dict[str, str] = {}

    # Memory index cache: (mtime_proxy, entries). mtime_proxy is bumped
    # after every write so we know when to re-scan.
    index_state: dict[str, Any] = {"version": 0, "entries": []}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _ensure_dir() -> None:
        """Create the memory directory if it does not exist."""
        def _mkdir() -> None:
            base_path.mkdir(parents=True, exist_ok=True)
        try:
            await asyncio.to_thread(_mkdir)
        except Exception:
            pass

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
        def _write_index() -> None:
            (base_path / "MEMORY.md").write_bytes(body.encode("utf-8"))
        try:
            await asyncio.to_thread(_write_index)
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
        def _do_write() -> None:
            filepath.write_bytes(body.encode("utf-8"))
        try:
            await _ensure_dir()
            await asyncio.to_thread(_do_write)
        except Exception:
            logger.debug("memory_lifecycle: failed to write %s", filepath)
            return False
        index_state["version"] += 1
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
        ts = str(int(time.time()))
        name = f"auto-{slug}-{ts}"
        # Sanitise to valid memory name chars
        name = re.sub(r"[^A-Za-z0-9_-]", "-", name)

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
# Module-level helpers (not closures — stateless)
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
