"""Shared append-only artifact store for root session trees."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
import time
from typing import Any, Final

from pydantic import BaseModel

from agentm.core.abi import FunctionTool, TextContent, ToolResult
from agentm.core.lib import to_jsonable
from agentm.core.lib.artifact_files import (
    ArtifactCreator,
    ArtifactMetadata,
    artifacts_dir_for,
    find_metadata_files,
    list_artifacts_for_task,
    scan_artifact_metadata,
)
from agentm.extensions import ExtensionManifest
from agentm.core.abi.events import SessionReadyEvent
from agentm.core.abi.extension import ExtensionAPI

_DEFAULT_INLINE_BYTES = 8 * 1024
_DEFAULT_LIST_LIMIT = 50
_DEFAULT_GREP_MAX_HITS = 20
_DEFAULT_SNIPPET_LINES = 2
_NEXT_ID_WIDTH = 3
_SLUG_RE = re.compile(r"[^a-z0-9]+")
_KIND_RE = re.compile(r"[^A-Za-z0-9_-]+")


class ArtifactStoreConfig(BaseModel):
    inline_max_bytes: int = _DEFAULT_INLINE_BYTES
    max_inline_bytes: int = _DEFAULT_INLINE_BYTES
    list_max_results: int = _DEFAULT_LIST_LIMIT
    grep_max_matches: int = _DEFAULT_GREP_MAX_HITS
    root_session_id: str | None = None
    task_id: str | None = None
    persona: str | None = None


MANIFEST = ExtensionManifest(
    name="artifact_store",
    description="Shared append-only filesystem artifact store for session trees.",
    registers=(
        "tool:artifact_write",
        "tool:artifact_read",
        "tool:artifact_list",
        "tool:artifact_grep",
        "event:session_ready",
    ),
    config_schema=ArtifactStoreConfig,
    requires=(),  # Leaf atom: registers its own tools and service.
)


@dataclass(slots=True)
class _StoreContext:
    cwd: Path
    layout: Any
    session_id: str
    root_session_id: str
    task_id: str | None
    persona: str | None
    max_inline_bytes: int
    list_max_results: int
    grep_max_matches: int

    @property
    def artifacts_dir(self) -> Path:
        return artifacts_dir_for(self.layout, self.root_session_id)


class ArtifactStore:
    def __init__(self, api: ExtensionAPI, config: ArtifactStoreConfig) -> None:
        self._api = api
        self._id_lock = asyncio.Lock()
        root_session_id = str(config.root_session_id or api.session_id)
        # inline_max_bytes takes priority when explicitly set (non-default);
        # otherwise fall back to max_inline_bytes for backward compat.
        max_inline = config.inline_max_bytes if config.inline_max_bytes != _DEFAULT_INLINE_BYTES else config.max_inline_bytes
        self._ctx = _StoreContext(
            cwd=Path(api.cwd),
            layout=api.get_project_layout(),
            session_id=api.session_id,
            root_session_id=root_session_id,
            task_id=_maybe_str(config.task_id),
            persona=_maybe_str(config.persona),
            max_inline_bytes=max_inline,
            list_max_results=config.list_max_results,
            grep_max_matches=config.grep_max_matches,
        )

    async def on_session_ready(self, event: SessionReadyEvent) -> None:
        self._ctx.session_id = event.session_id
        self._ctx.root_session_id = event.root_session_id
        self._ctx.task_id = event.task_id
        self._ctx.persona = event.persona

    async def write(self, args: dict[str, Any]) -> ToolResult:
        try:
            result = await self.write_artifact(
                kind=str(args.get("kind", "")),
                title=str(args.get("title", "")),
                body=str(args.get("body", "")),
                tags=args.get("tags"),
                parent_artifact_ids=args.get("parent_artifact_ids"),
            )
        except ValueError as exc:
            return _error(str(exc))
        except OSError as exc:
            return _error(f"failed to write artifact: {exc}")
        return _ok(result)

    async def write_artifact(
        self,
        *,
        kind: str,
        title: str,
        body: str,
        tags: Any = None,
        parent_artifact_ids: Any = None,
    ) -> dict[str, str]:
        clean_kind = _sanitize_kind(str(kind).strip())
        clean_title = str(title).strip()
        if not clean_kind or not clean_title:
            raise ValueError("kind and title are required")
        clean_tags = _coerce_tags(tags)
        clean_parent_ids = _coerce_parent_ids(parent_artifact_ids)
        artifact_id = await self._allocate_id()
        slug = _slugify(clean_title)
        body_path = self._ctx.artifacts_dir / f"{artifact_id}__{clean_kind}__{slug}.md"
        meta_path = body_path.with_suffix(".meta.json")
        timestamp = time.time()
        metadata = {
            "artifact_id": artifact_id,
            "kind": clean_kind,
            "title": clean_title,
            "slug": slug,
            "path": str(body_path),
            "parent_id": clean_parent_ids[0] if clean_parent_ids else None,
            "parent_artifact_ids": clean_parent_ids,
            "tags": clean_tags,
            "created_by": {
                "session_id": self._ctx.session_id,
                "task_id": self._ctx.task_id,
                "persona": self._ctx.persona,
                "timestamp": timestamp,
            },
        }
        await asyncio.to_thread(_atomic_write_text, body_path, str(body))
        await asyncio.to_thread(
            _atomic_write_text,
            meta_path,
            json.dumps(to_jsonable(metadata), ensure_ascii=False, indent=2, sort_keys=True),
        )
        return {"artifact_id": artifact_id, "path": str(body_path)}

    async def read(self, args: dict[str, Any]) -> ToolResult:
        artifact_id = str(args.get("artifact_id", "")).strip()
        if not artifact_id:
            return _error("artifact_id is required")
        meta = await self._load_metadata(artifact_id)
        if meta is None:
            return _error(f"unknown artifact_id: {artifact_id}")
        body_path = Path(str(meta["path"]))
        try:
            raw = await asyncio.to_thread(body_path.read_bytes)
        except OSError as exc:
            return _error(f"failed to read artifact body for {artifact_id}: {exc}")
        range_arg = args.get("range")
        text: str
        if range_arg is None:
            text = _read_default(raw, self._ctx.max_inline_bytes)
        else:
            try:
                text = _read_range(raw, range_arg)
            except ValueError as exc:
                return _error(str(exc))
        payload = {
            "artifact_id": artifact_id,
            "kind": meta["kind"],
            "title": meta["title"],
            "path": meta["path"],
            "created_by": meta["created_by"],
            "body": text,
        }
        return _ok(payload)

    async def list_artifacts(self, args: dict[str, Any]) -> ToolResult:
        try:
            kind_filter = _maybe_str(args.get("kind"))
            tags_filter = _coerce_tags(args.get("tags"))
            created_by_task = _maybe_str(args.get("created_by_task"))
            since = _parse_since(args.get("since"))
            limit = max(1, int(args.get("limit", self._ctx.list_max_results)))
        except ValueError as exc:
            return _error(str(exc))
        metas = await self._filtered_metadata(
            kind=kind_filter,
            tags=tags_filter,
            created_by_task=created_by_task,
            since=since,
        )
        artifacts = [
            {
                "id": str(meta["artifact_id"]),
                "kind": str(meta["kind"]),
                "title": str(meta["title"]),
                "path": str(meta["path"]),
                "created_by": meta["created_by"],
                "tags": list(meta.get("tags", [])),
                "size_bytes": _file_size(meta),
            }
            for meta in metas[:limit]
        ]
        return _ok({"artifacts": artifacts})

    async def grep(self, args: dict[str, Any]) -> ToolResult:
        pattern = str(args.get("pattern", ""))
        if not pattern:
            return _error("pattern is required")
        try:
            compiled = re.compile(pattern)
        except re.error as exc:
            return _error(f"invalid regex: {exc}")
        try:
            kind_filter = _maybe_str(args.get("kind"))
            tags_filter = _coerce_tags(args.get("tags"))
            max_hits = max(1, int(args.get("max_hits", self._ctx.grep_max_matches)))
            snippet_lines = max(0, int(args.get("snippet_lines", _DEFAULT_SNIPPET_LINES)))
        except ValueError as exc:
            return _error(str(exc))
        metas = await self._filtered_metadata(kind=kind_filter, tags=tags_filter)
        hits: list[dict[str, Any]] = []
        for meta in metas:
            if len(hits) >= max_hits:
                break
            body_path = Path(str(meta["path"]))
            try:
                lines = (await asyncio.to_thread(body_path.read_text, encoding="utf-8")).splitlines()
            except OSError:
                continue
            for index, line in enumerate(lines, start=1):
                if compiled.search(line) is None:
                    continue
                start = max(1, index - snippet_lines)
                end = min(len(lines), index + snippet_lines)
                snippet = "\n".join(lines[start - 1 : end])
                hits.append(
                    {
                        "id": str(meta["artifact_id"]),
                        "kind": str(meta["kind"]),
                        "title": str(meta["title"]),
                        "line_no": index,
                        "snippet": snippet,
                    }
                )
                if len(hits) >= max_hits:
                    break
        return _ok({"hits": hits})

    async def _allocate_id(self) -> str:
        artifacts_dir = self._ctx.artifacts_dir
        async with self._id_lock:
            return await asyncio.to_thread(_allocate_id_sync, artifacts_dir)

    async def _filtered_metadata(
        self,
        *,
        kind: str | None = None,
        tags: list[str] | None = None,
        created_by_task: str | None = None,
        since: float | None = None,
    ) -> list[ArtifactMetadata]:
        metas = await asyncio.to_thread(self._scan_metadata)
        filtered: list[ArtifactMetadata] = []
        for meta in metas:
            if kind is not None and str(meta.get("kind")) != kind:
                continue
            meta_tags = [str(tag) for tag in meta.get("tags", []) if isinstance(tag, str)]
            if tags and not set(tags).issubset(set(meta_tags)):
                continue
            raw_created_by = meta.get("created_by")
            created_by: ArtifactCreator
            if isinstance(raw_created_by, dict):
                created_by = raw_created_by
            else:
                created_by = {}
            if created_by_task is not None and created_by.get("task_id") != created_by_task:
                continue
            timestamp = created_by.get("timestamp")
            if since is not None and isinstance(timestamp, (int, float)) and float(timestamp) < since:
                continue
            if since is not None and not isinstance(timestamp, (int, float)):
                continue
            filtered.append(meta)
        filtered.sort(
            key=lambda meta: float(meta.get("created_by", {}).get("timestamp", 0.0)),
            reverse=True,
        )
        return filtered

    async def _load_metadata(self, artifact_id: str) -> dict[str, Any] | None:
        matches = await asyncio.to_thread(
            find_metadata_files, self._ctx.artifacts_dir, artifact_id
        )
        if not matches:
            return None
        try:
            return json.loads(matches[0].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _scan_metadata(self) -> list[ArtifactMetadata]:
        return scan_artifact_metadata(self._ctx.artifacts_dir)


def install(api: ExtensionAPI, config: ArtifactStoreConfig) -> None:
    store = ArtifactStore(api, config)
    api.set_service("artifact_store", store)

    def _store() -> ArtifactStore:
        registered = api.get_service("artifact_store")
        if not isinstance(registered, ArtifactStore):
            raise RuntimeError("artifact_store service is not registered")
        return registered

    async def _write(args: dict[str, Any]) -> ToolResult:
        return await _store().write(args)

    async def _read(args: dict[str, Any]) -> ToolResult:
        return await _store().read(args)

    async def _list_artifacts(args: dict[str, Any]) -> ToolResult:
        return await _store().list_artifacts(args)

    async def _grep(args: dict[str, Any]) -> ToolResult:
        return await _store().grep(args)

    api.on(SessionReadyEvent.CHANNEL, store.on_session_ready)
    api.register_tool(
        FunctionTool(
            name="artifact_write",
            description="Append a new shared artifact to the session-tree store.",
            parameters={
                "type": "object",
                "properties": {
                    "kind": {"type": "string"},
                    "title": {"type": "string"},
                    "body": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "parent_artifact_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["kind", "title", "body"],
                "additionalProperties": False,
            },
            fn=_write,
        )
    )
    api.register_tool(
        FunctionTool(
            name="artifact_read",
            description="Read one artifact by id, optionally with a byte/line range.",
            parameters={
                "type": "object",
                "properties": {
                    "artifact_id": {"type": "string"},
                    "range": {
                        "type": "object",
                        "properties": {
                            "mode": {"type": "string", "enum": ["bytes", "lines"]},
                            "start": {"type": "integer", "minimum": 0},
                            "end": {"type": "integer", "minimum": 0},
                        },
                        "required": ["mode", "start", "end"],
                        "additionalProperties": False,
                    },
                },
                "required": ["artifact_id"],
                "additionalProperties": False,
            },
            fn=_read,
        )
    )
    api.register_tool(
        FunctionTool(
            name="artifact_list",
            description="List artifact metadata with simple filters.",
            parameters={
                "type": "object",
                "properties": {
                    "kind": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "created_by_task": {"type": "string"},
                    "since": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1},
                },
                "additionalProperties": False,
            },
            fn=_list_artifacts,
        )
    )
    api.register_tool(
        FunctionTool(
            name="artifact_grep",
            description="Regex-search text artifacts and return snippets.",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "kind": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "max_hits": {"type": "integer", "minimum": 1},
                    "snippet_lines": {"type": "integer", "minimum": 0},
                },
                "required": ["pattern"],
                "additionalProperties": False,
            },
            fn=_grep,
        )
    )


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(text, encoding="utf-8")
    os.replace(tmp_path, path)


def _allocate_id_sync(artifacts_dir: Path) -> str:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    lock_path = artifacts_dir / ".next_id.lock"
    fd: int | None = None
    while fd is None:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            time.sleep(0.005)
    try:
        next_id_path = artifacts_dir / ".next_id"
        next_value = 1
        if next_id_path.exists():
            try:
                next_value = int(next_id_path.read_text(encoding="utf-8").strip())
            except ValueError:
                next_value = 1
        artifact_id = f"art_{next_value:0{_NEXT_ID_WIDTH}d}"
        _atomic_write_text(next_id_path, str(next_value + 1))
        return artifact_id
    finally:
        os.close(fd)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _coerce_tags(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("tags must be a list of strings")
    tags: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            tags.append(text)
    return tags


def _coerce_parent_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("parent_artifact_ids must be a list of strings")
    ids: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            ids.append(text)
    return ids


def _read_default(raw: bytes, cap: int) -> str:
    if len(raw) <= cap:
        return raw.decode("utf-8", errors="replace")
    head = raw[:cap].decode("utf-8", errors="replace")
    marker = (
        f"\n\n[Truncated: bytes 0-{cap} of {len(raw)}. "
        "Call artifact_read again with range={bytes:[start,end]} or a line range.]"
    )
    return head + marker


def _read_range(raw: bytes, range_arg: Any) -> str:
    if not isinstance(range_arg, dict):
        raise ValueError("range must be an object")
    keys = [key for key in ("lines", "bytes", "head", "tail") if key in range_arg]
    if len(keys) != 1:
        raise ValueError("range must contain exactly one of lines, bytes, head, or tail")
    key = keys[0]
    if key == "bytes":
        raw_range = range_arg["bytes"]
        if not isinstance(raw_range, list) or len(raw_range) != 2:
            raise ValueError("range.bytes must be [start, end]")
        start = max(0, int(raw_range[0]))
        end = max(start, int(raw_range[1]))
        return raw[start:end].decode("utf-8", errors="replace")
    text = raw.decode("utf-8", errors="replace")
    lines = text.splitlines()
    if key == "lines":
        raw_range = range_arg["lines"]
        if not isinstance(raw_range, list) or len(raw_range) != 2:
            raise ValueError("range.lines must be [start, end]")
        start = max(1, int(raw_range[0]))
        end = max(start, int(raw_range[1]))
        return "\n".join(lines[start - 1 : end])
    if key == "head":
        count = max(1, int(range_arg["head"]))
        return "\n".join(lines[:count])
    count = max(1, int(range_arg["tail"]))
    return "\n".join(lines[-count:])


def _parse_since(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            iso = text.replace("Z", "+00:00")
            return datetime.fromisoformat(iso).timestamp()
    raise ValueError("since must be a unix timestamp or ISO-8601 string")


def _slugify(title: str) -> str:
    slug = _SLUG_RE.sub("-", title.lower()).strip("-")
    return slug or "artifact"


def _sanitize_kind(kind: str) -> str:
    clean = _KIND_RE.sub("_", kind).strip("_")
    return clean or "artifact"


def _maybe_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _file_size(meta: ArtifactMetadata) -> int:
    try:
        return Path(str(meta["path"])).stat().st_size
    except OSError:
        return 0



def _ok(payload: dict[str, Any]) -> ToolResult:
    return ToolResult(
        content=[TextContent(type="text", text=json.dumps(to_jsonable(payload), ensure_ascii=False))],
        extras=payload,
    )


def _error(message: str) -> ToolResult:
    payload = {"error": message}
    return ToolResult(
        content=[TextContent(type="text", text=json.dumps(to_jsonable(payload), ensure_ascii=False))],
        is_error=True,
        extras=payload,
    )

__all__: Final = [
    "MANIFEST",
    "ArtifactStore",
    "artifacts_dir_for",
    "find_metadata_files",
    "install",
    "list_artifacts_for_task",
    "scan_artifact_metadata",
]
