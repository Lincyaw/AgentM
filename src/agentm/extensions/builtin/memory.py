"""Builtin ``memory`` atom.

Project-local persistent memory built on a layered model (L2 global /
L3 SOP / L4 archive) with a typed-frontmatter convention. Skills handle
"how to do X"; memory handles "what X is" — facts about the user, the
project, prior feedback, and external references.

On-disk layout (under ``<cwd>/.agentm/memory/`` by default):

    .agentm/memory/
    ├── MEMORY.md            # one-line index, always loaded into system prompt
    ├── feedback_*.md        # type=feedback — user guidance / preferences
    ├── project_*.md         # type=project — facts about the project
    ├── user_*.md            # type=user — user role / habits
    ├── reference_*.md       # type=reference — pointers to external systems
    └── access_stats.json    # {name: {count, last_access}} — evidence for evolution

Each memory file carries YAML frontmatter::

    ---
    name: testing_no_mocks
    description: integration tests must hit real database, not mocks
    type: feedback
    ---
    <body>

Only ``MEMORY.md`` (the index) is injected into the system prompt; bodies
are loaded on demand via the ``memory_read`` tool. This is the GA <30K
context trick — keep only relevance hints in prompt, let the agent decide
what to expand.

The atom is self-contained and §11-compliant: file reads go through
``api.get_operations().file``; writes go through ``api.get_resource_writer()``.
Access bookkeeping is intentionally write-through to disk so it survives
restarts and can be mined by future evolution/query atoms.
"""

from __future__ import annotations

import json
from loguru import logger
import re
import time
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel

from agentm.core.abi import (
    BeforeAgentStartEvent,
    ExtensionAPI,
    FunctionTool,
    TextContent,
    ToolResult,
)
from agentm.core.lib import parse_frontmatter, with_model_note
from agentm.extensions import ExtensionManifest


_VALID_TYPES: Final[tuple[str, ...]] = ("feedback", "project", "user", "reference")
_NAME_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9_-]+$")
_DEFAULT_MAX_INDEX_LINES: Final[int] = 200

class MemoryConfig(BaseModel):
    path: str = ".agentm/memory"
    index_in_system_prompt: bool = True
    max_index_lines: int = _DEFAULT_MAX_INDEX_LINES

MANIFEST = ExtensionManifest(
    name="memory",
    description=(
        "Project-local persistent memory: MEMORY.md index injected into the "
        "system prompt, typed frontmatter files (feedback/project/user/"
        "reference) loaded on demand, access counter for evolution evidence."
    ),
    registers=(
        "event:before_agent_start",
        "tool:memory_save",
        "tool:memory_read",
        "tool:memory_search",
        "tool:memory_delete",
    ),
    config_schema=MemoryConfig,
    requires=(),
)

_SAVE_PARAMS: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "type": {
            "type": "string",
            "enum": list(_VALID_TYPES),
            "description": "Memory category.",
        },
        "name": {
            "type": "string",
            "description": (
                "Short identifier; letters/digits/underscore/hyphen only. "
                "File is stored as <type>_<name>.md."
            ),
        },
        "description": {
            "type": "string",
            "description": (
                "One-line relevance hint shown in MEMORY.md index. Used for "
                "future memory_search ranking."
            ),
        },
        "content": {
            "type": "string",
            "description": "Memory body. Markdown is fine.",
        },
    },
    "required": ["type", "name", "description", "content"],
    "additionalProperties": False,
}

_READ_PARAMS: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": (
                "Memory name (without the type_ prefix or .md extension). "
                "Increments the access counter on success."
            ),
        }
    },
    "required": ["name"],
    "additionalProperties": False,
}

_SEARCH_PARAMS: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": (
                "Substring matched case-insensitively against each memory's "
                "name + description. Returns up to ``limit`` results."
            ),
        },
        "limit": {
            "type": "integer",
            "description": "Max results to return (default 10).",
        },
    },
    "required": ["query"],
    "additionalProperties": False,
}

_DELETE_PARAMS: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "Memory name to delete; also removes its MEMORY.md entry.",
        }
    },
    "required": ["name"],
    "additionalProperties": False,
}

def install(api: ExtensionAPI, config: MemoryConfig) -> None:
    base_path = _resolve_base(api.cwd, config.path)
    index_in_prompt = config.index_in_system_prompt
    max_index_lines = config.max_index_lines

    file_ops = api.get_operations().file
    writer = api.get_resource_writer()

    if index_in_prompt:

        async def _before_agent_start(
            event: BeforeAgentStartEvent,
        ) -> dict[str, str] | None:
            block = await _build_index_block(file_ops, base_path, max_index_lines)
            if not block:
                return None
            current = str(event.system or "")
            updated = f"{block}\n\n{current}" if current else block
            # Mutate AND return: the kernel's ``collect_system_replacement``
            # reads handler returns, so a mutate-only handler is dropped when
            # it fires last (no later handler folds its mutation into a
            # return). Returning makes the index order-independent; the
            # mutation keeps it visible to handlers firing after this one.
            event.system = updated
            return {"system": updated}

        api.on(BeforeAgentStartEvent.CHANNEL, _before_agent_start)

    async def _save(args: dict[str, Any]) -> ToolResult:
        mem_type = str(args["type"])
        name = str(args["name"])
        description = str(args["description"]).strip()
        content = str(args["content"])

        if mem_type not in _VALID_TYPES:
            return _error(f"unknown type {mem_type!r}; expected one of {list(_VALID_TYPES)}")
        if not _NAME_RE.match(name):
            return _error(
                f"invalid name {name!r}; use letters/digits/underscore/hyphen only"
            )
        if "\n" in description:
            return _error("description must be single-line (no newlines)")

        rel_md = _memory_relpath(base_path, mem_type, name, api.cwd)
        body = _serialize_memory(mem_type, name, description, content)

        try:
            write_result = await writer.write(rel_md, body.encode("utf-8"), rationale="memory_save")
            if getattr(write_result, "error", None) is not None:
                return _error(f"write failed: {write_result.error}")
        except Exception as exc:
            return _error(f"write failed: {exc}")

        index_error = await _rewrite_index(file_ops, writer, base_path, api.cwd)
        if index_error is not None:
            return _error(index_error)
        return _ok(f"saved memory {mem_type}/{name}")

    async def _read(args: dict[str, Any]) -> ToolResult:
        name = str(args["name"])
        path = await _resolve_memory_path(file_ops, base_path, name)
        if path is None:
            return _error(f"memory {name!r} not found in {base_path}")
        try:
            data = await file_ops.read_file(str(path))
        except Exception as exc:
            return _error(f"read failed: {exc}")
        text = data.decode("utf-8", errors="replace")
        await _record_access(file_ops, writer, base_path, name, api.cwd)
        return _ok(text)

    async def _search(args: dict[str, Any]) -> ToolResult:
        query = str(args["query"]).lower().strip()
        limit = int(args.get("limit", 10))
        if not query:
            return _error("query is empty")

        entries: list[tuple[str, str, str]] = []
        skipped: list[str] = []
        for path in await _list_memory_files(file_ops, base_path):
            try:
                data = await file_ops.read_file(str(path))
            except Exception as exc:  # noqa: BLE001
                # Skip an unreadable memory file rather than failing the search,
                # but record it so the model learns recall may be incomplete.
                logger.debug("memory: skipping unreadable file {}: {}", path, exc)
                skipped.append(path.name)
                continue
            meta, _body = parse_frontmatter(data.decode("utf-8", errors="replace"))
            name = str(meta.get("name", path.stem))
            description = str(meta.get("description", ""))
            mem_type = str(meta.get("type", ""))
            haystack = f"{name} {description}".lower()
            if query in haystack:
                entries.append((name, mem_type, description))

        if not entries:
            result = _ok(f"no memories matched {query!r}")
        else:
            entries.sort(key=lambda row: row[0])
            lines = [f"- {name} [{mem_type}] — {desc}" for name, mem_type, desc in entries[:limit]]
            result = _ok("\n".join(lines))
        if skipped:
            # Surface the silently-skipped files to the model, not just the log.
            with_model_note(
                result,
                f"{len(skipped)} memory file(s) could not be read and were "
                f"skipped ({', '.join(skipped)}); this recall may be incomplete.",
            )
        return result

    async def _delete(args: dict[str, Any]) -> ToolResult:
        name = str(args["name"])
        path = await _resolve_memory_path(file_ops, base_path, name)
        if path is None:
            return _error(f"memory {name!r} not found in {base_path}")
        rel = _to_cwd_relative(path, api.cwd)
        try:
            await writer.delete(rel, rationale="memory_delete")
        except Exception as exc:
            return _error(f"delete failed: {exc}")
        index_error = await _rewrite_index(file_ops, writer, base_path, api.cwd)
        if index_error is not None:
            return _error(index_error)
        return _ok(f"deleted memory {name}")

    api.register_tool(
        FunctionTool(
            name="memory_save",
            description=(
                "Persist a typed memory (feedback/project/user/reference) "
                "into .agentm/memory/. Updates MEMORY.md index."
            ),
            parameters=_SAVE_PARAMS,
            fn=_save,
            metadata={"memory_op": "save"},
        )
    )
    api.register_tool(
        FunctionTool(
            name="memory_read",
            description=(
                "Read a memory's body by name. Increments access counter so "
                "evolution can later identify hot/cold memories."
            ),
            parameters=_READ_PARAMS,
            fn=_read,
            metadata={"memory_op": "read"},
        )
    )
    api.register_tool(
        FunctionTool(
            name="memory_search",
            description=(
                "Substring search across memory name+description; returns "
                "up to limit candidates with their type and one-line hint."
            ),
            parameters=_SEARCH_PARAMS,
            fn=_search,
            metadata={"memory_op": "search"},
        )
    )
    api.register_tool(
        FunctionTool(
            name="memory_delete",
            description="Delete a memory file and remove its MEMORY.md entry.",
            parameters=_DELETE_PARAMS,
            fn=_delete,
            metadata={"memory_op": "delete"},
        )
    )

def _resolve_base(cwd: str, raw_path: str) -> Path:
    raw = Path(raw_path).expanduser()
    return raw if raw.is_absolute() else (Path(cwd) / raw).resolve()

def _memory_relpath(base: Path, mem_type: str, name: str, cwd: str) -> str:
    abs_path = base / f"{mem_type}_{name}.md"
    return _to_cwd_relative(abs_path, cwd)

def _to_cwd_relative(path: Path, cwd: str) -> str:
    cwd_path = Path(cwd).resolve()
    try:
        return str(path.resolve().relative_to(cwd_path))
    except ValueError:
        return str(path)

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

async def _list_memory_files(file_ops: Any, base: Path) -> list[Path]:
    try:
        names = await file_ops.list_dir(str(base))
    except Exception as exc:
        logger.warning(f"memory: failed to list {base}: {exc}")
        return []
    out: list[Path] = []
    for entry in names:
        if not entry.endswith(".md") or entry == "MEMORY.md":
            continue
        out.append(base / entry)
    return sorted(out)

async def _resolve_memory_path(file_ops: Any, base: Path, name: str) -> Path | None:
    """Find ``<type>_<name>.md`` without forcing the caller to know the type."""

    for mem_type in _VALID_TYPES:
        candidate = base / f"{mem_type}_{name}.md"
        try:
            if await file_ops.access(str(candidate)):
                return candidate
        except Exception as exc:  # noqa: BLE001
            # Access check failed for this type variant — try the next one.
            logger.debug("memory: access check failed for {}: {}", candidate, exc)
            continue
    return None

async def _build_index_block(file_ops: Any, base: Path, max_lines: int) -> str:
    index_path = base / "MEMORY.md"
    try:
        if not await file_ops.access(str(index_path)):
            return ""
        raw = await file_ops.read_file(str(index_path))
    except Exception as exc:
        logger.warning(f"memory: failed to read index {index_path}: {exc}")
        return ""
    text = raw.decode("utf-8", errors="replace").strip()
    if not text:
        return ""
    lines = text.splitlines()
    truncated = ""
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        truncated = f"\n... ({len(text.splitlines()) - max_lines} more lines, use memory_search)"
    body = "\n".join(lines) + truncated
    return (
        "# Memory\n\n"
        "You have a persistent memory system. The index below contains one-line "
        "summaries of stored memories — it is loaded every turn so you can see "
        "what is available.\n"
        "- To read a memory's full content, call `memory_read` with its name.\n"
        "- To save new information the user shares (preferences, project context, "
        "corrections), call `memory_save`.\n"
        "- To find memories by keyword, call `memory_search`.\n"
        "- To remove outdated information, call `memory_delete`.\n\n"
        "When a task relates to a memory entry listed below, read it first — "
        "the one-line summary is a relevance hint, not the full content. "
        "When the user explicitly asks you to remember or recall something, "
        "always use the memory tools.\n\n"
        f"<memory_index>\n{body}\n</memory_index>"
    )

async def _rewrite_index(
    file_ops: Any,
    writer: Any,
    base: Path,
    cwd: str,
) -> str | None:
    """Regenerate MEMORY.md from current files. Returns error string or None."""

    entries: list[tuple[str, str, str]] = []
    for path in await _list_memory_files(file_ops, base):
        try:
            data = await file_ops.read_file(str(path))
        except Exception as exc:  # noqa: BLE001
            # Skip an unreadable memory file when rebuilding the index.
            logger.debug("memory: skipping unreadable file {} during reindex: {}", path, exc)
            continue
        meta, _body = parse_frontmatter(data.decode("utf-8", errors="replace"))
        name = str(meta.get("name", path.stem))
        mem_type = str(meta.get("type", ""))
        description = str(meta.get("description", ""))
        entries.append((name, mem_type, description))
    entries.sort(key=lambda row: row[0])

    lines = [f"- [{mem_type}/{name}] {description}" for name, mem_type, description in entries]
    body = "\n".join(lines) + ("\n" if lines else "")
    rel = _to_cwd_relative(base / "MEMORY.md", cwd)
    try:
        result = await writer.write(rel, body.encode("utf-8"), rationale="memory_index_rebuild")
        if getattr(result, "error", None) is not None:
            return f"index rebuild failed: {result.error}"
    except Exception as exc:
        return f"index rebuild failed: {exc}"
    return None

async def _record_access(
    file_ops: Any,
    writer: Any,
    base: Path,
    name: str,
    cwd: str,
) -> None:
    """Increment ``access_stats.json[name]``. Best-effort: failures are
    silent so they never break the read path."""

    stats_path = base / "access_stats.json"
    stats: dict[str, Any] = {}
    try:
        if await file_ops.access(str(stats_path)):
            raw = await file_ops.read_file(str(stats_path))
            stats = json.loads(raw.decode("utf-8", errors="replace"))
            if not isinstance(stats, dict):
                stats = {}
    except Exception:
        stats = {}

    prev = stats.get(name)
    record: dict[str, Any] = prev if isinstance(prev, dict) else {}
    record["count"] = int(record.get("count", 0)) + 1
    record["last_access"] = time.strftime("%Y-%m-%d %H:%M:%S")
    stats[name] = record

    rel = _to_cwd_relative(stats_path, cwd)
    payload = json.dumps(stats, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    try:
        await writer.write(rel, payload.encode("utf-8"), rationale="memory_access_stats")
    except Exception:
        return

def _ok(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)])

def _error(text: str) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)], is_error=True)
