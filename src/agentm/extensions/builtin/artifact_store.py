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
from typing import Any

from agentm.core.abi import FunctionTool, TextContent, ToolResult
from agentm.extensions import ExtensionManifest
from agentm.harness.events import SessionReadyEvent
from agentm.harness.extension import ExtensionAPI

_DEFAULT_INLINE_BYTES = 8 * 1024
_DEFAULT_LIST_LIMIT = 50
_DEFAULT_GREP_MAX_HITS = 20
_DEFAULT_SNIPPET_LINES = 2
_NEXT_ID_WIDTH = 3
_SLUG_RE = re.compile(r"[^a-z0-9]+")
_KIND_RE = re.compile(r"[^A-Za-z0-9_-]+")

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
    config_schema={
        "type": "object",
        "properties": {
            "max_inline_bytes": {"type": "integer", "minimum": 1, "default": _DEFAULT_INLINE_BYTES},
            "root_session_id": {"type": "string"},
            "task_id": {"type": ["string", "null"]},
            "persona": {"type": ["string", "null"]},
        },
        "additionalProperties": False,
    },
)


@dataclass(slots=True)
class _StoreContext:
    cwd: Path
    session_id: str
    root_session_id: str
    task_id: str | None
    persona: str | None
    max_inline_bytes: int

    @property
    def artifacts_dir(self) -> Path:
        return self.cwd / ".agentm" / "artifacts" / self.root_session_id


class _ArtifactStore:
    def __init__(self, api: ExtensionAPI, config: dict[str, Any]) -> None:
        self._api = api
        root_session_id = str(config.get("root_session_id") or api.session_id)
        self._ctx = _StoreContext(
            cwd=Path(api.cwd),
            session_id=api.session_id,
            root_session_id=root_session_id,
            task_id=_maybe_str(config.get("task_id")),
            persona=_maybe_str(config.get("persona")),
            max_inline_bytes=int(config.get("max_inline_bytes", _DEFAULT_INLINE_BYTES)),
        )
        self._id_lock = asyncio.Lock()

    async def on_session_ready(self, event: SessionReadyEvent) -> None:
        self._ctx.session_id = event.session_id
        self._ctx.root_session_id = event.root_session_id
        self._ctx.task_id = event.task_id
        self._ctx.persona = event.persona

    async def write(self, args: dict[str, Any]) -> ToolResult:
        kind = _sanitize_kind(str(args.get("kind", "")).strip())
        title = str(args.get("title", "")).strip()
        body = str(args.get("body", ""))
        if not kind or not title:
            return _error("kind and title are required")
        tags = _coerce_tags(args.get("tags"))
        parent_artifact_ids = _coerce_parent_ids(args.get("parent_artifact_ids"))
        artifact_id = await self._allocate_id()
        slug = _slugify(title)
        body_path = self._ctx.artifacts_dir / f"{artifact_id}__{kind}__{slug}.md"
        meta_path = body_path.with_suffix(".meta.json")
        timestamp = time.time()
        metadata = {
            "artifact_id": artifact_id,
            "kind": kind,
            "title": title,
            "slug": slug,
            "path": str(body_path),
            "parent_id": parent_artifact_ids[0] if parent_artifact_ids else None,
            "parent_artifact_ids": parent_artifact_ids,
            "tags": tags,
            "created_by": {
                "session_id": self._ctx.session_id,
                "task_id": self._ctx.task_id,
                "persona": self._ctx.persona,
                "timestamp": timestamp,
            },
        }
        try:
            await asyncio.to_thread(_atomic_write_text, body_path, body)
            await asyncio.to_thread(
                _atomic_write_text,
                meta_path,
                json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True),
            )
        except Exception as exc:  # noqa: BLE001
            return _error(f"failed to write artifact {artifact_id}: {exc}")
        return _ok({"artifact_id": artifact_id, "path": str(body_path)})

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

    async def list(self, args: dict[str, Any]) -> ToolResult:
        kind_filter = _maybe_str(args.get("kind"))
        tags_filter = _coerce_tags(args.get("tags"))
        created_by_task = _maybe_str(args.get("created_by_task"))
        since = _parse_since(args.get("since"))
        limit = max(1, int(args.get("limit", _DEFAULT_LIST_LIMIT)))
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
        kind_filter = _maybe_str(args.get("kind"))
        tags_filter = _coerce_tags(args.get("tags"))
        max_hits = max(1, int(args.get("max_hits", _DEFAULT_GREP_MAX_HITS)))
        snippet_lines = max(0, int(args.get("snippet_lines", _DEFAULT_SNIPPET_LINES)))
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
        async with self._id_lock:
            artifacts_dir = self._ctx.artifacts_dir
            await asyncio.to_thread(artifacts_dir.mkdir, parents=True, exist_ok=True)
            next_id_path = artifacts_dir / ".next_id"
            next_value = 1
            if next_id_path.exists():
                try:
                    next_value = int(next_id_path.read_text(encoding="utf-8").strip())
                except ValueError:
                    next_value = 1
            artifact_id = f"art_{next_value:0{_NEXT_ID_WIDTH}d}"
            await asyncio.to_thread(
                _atomic_write_text,
                next_id_path,
                str(next_value + 1),
            )
            return artifact_id

    async def _filtered_metadata(
        self,
        *,
        kind: str | None = None,
        tags: list[str] | None = None,
        created_by_task: str | None = None,
        since: float | None = None,
    ) -> list[dict[str, Any]]:
        metas = await asyncio.to_thread(self._scan_metadata)
        filtered: list[dict[str, Any]] = []
        for meta in metas:
            if kind is not None and str(meta.get("kind")) != kind:
                continue
            meta_tags = [str(tag) for tag in meta.get("tags", []) if isinstance(tag, str)]
            if tags and not set(tags).issubset(set(meta_tags)):
                continue
            created_by = meta.get("created_by") if isinstance(meta.get("created_by"), dict) else {}
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
        matches = await asyncio.to_thread(self._find_metadata_files, artifact_id)
        if not matches:
            return None
        try:
            return json.loads(matches[0].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _find_metadata_files(self, artifact_id: str) -> list[Path]:
        root = self._ctx.cwd / ".agentm" / "artifacts"
        if not root.exists():
            return []
        pattern = f"{artifact_id}__*.meta.json"
        return sorted(root.glob(f"*/{pattern}"))

    def _scan_metadata(self) -> list[dict[str, Any]]:
        root = self._ctx.artifacts_dir
        if not root.exists():
            return []
        metas: list[dict[str, Any]] = []
        for meta_path in sorted(root.glob("*.meta.json")):
            try:
                metas.append(json.loads(meta_path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
        return metas


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    store = _ArtifactStore(api, config)
    api.on("session_ready", store.on_session_ready)
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
            fn=store.write,
        )
    )
    api.register_tool(
        FunctionTool(
            name="artifact_read",
            description="Read a shared artifact body with bounded ranges.",
            parameters={
                "type": "object",
                "properties": {
                    "artifact_id": {"type": "string"},
                    "range": {
                        "type": "object",
                        "properties": {
                            "lines": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "minItems": 2,
                                "maxItems": 2,
                            },
                            "bytes": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "minItems": 2,
                                "maxItems": 2,
                            },
                            "head": {"type": "integer", "minimum": 1},
                            "tail": {"type": "integer", "minimum": 1},
                        },
                        "additionalProperties": False,
                    },
                },
                "required": ["artifact_id"],
                "additionalProperties": False,
            },
            fn=store.read,
        )
    )
    api.register_tool(
        FunctionTool(
            name="artifact_list",
            description="List shared artifacts without loading their bodies.",
            parameters={
                "type": "object",
                "properties": {
                    "kind": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "created_by_task": {"type": "string"},
                    "since": {"type": ["string", "number"]},
                    "limit": {"type": "integer", "minimum": 1, "default": _DEFAULT_LIST_LIMIT},
                },
                "additionalProperties": False,
            },
            fn=store.list,
        )
    )
    api.register_tool(
        FunctionTool(
            name="artifact_grep",
            description="Regex-scan shared artifacts with contextual snippets.",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "kind": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "max_hits": {"type": "integer", "minimum": 1, "default": _DEFAULT_GREP_MAX_HITS},
                    "snippet_lines": {"type": "integer", "minimum": 0, "default": _DEFAULT_SNIPPET_LINES},
                },
                "required": ["pattern"],
                "additionalProperties": False,
            },
            fn=store.grep,
        )
    )


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(text, encoding="utf-8")
    os.replace(tmp_path, path)


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


def _file_size(meta: dict[str, Any]) -> int:
    try:
        return Path(str(meta["path"])).stat().st_size
    except OSError:
        return 0


def _ok(payload: dict[str, Any]) -> ToolResult:
    return ToolResult(
        content=[TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))],
        details=payload,
    )


def _error(message: str) -> ToolResult:
    payload = {"error": message}
    return ToolResult(
        content=[TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))],
        is_error=True,
        details=payload,
    )


__all__ = ["MANIFEST", "install"]
