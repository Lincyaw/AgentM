# code-health: ignore-file[AM025] -- IFG utilities normalize untyped JSON values
"""Small JSON, id, path, and confidence helpers for IFG modules."""

from __future__ import annotations

import hashlib
import json
import posixpath
from collections.abc import Iterable, Mapping, Sequence

from sqlalchemy.engine import RowMapping


def _row_is_error(result: object, processed: object) -> bool:
    for value in (processed, result):
        if isinstance(value, Mapping):
            raw = value.get("is_error")
            if isinstance(raw, bool):
                return raw
            if isinstance(raw, str):
                return raw.lower() in {"1", "true", "yes"}
            if value.get("error"):
                return True
    return False


def _tool_result_text(result: object) -> str | None:
    if not isinstance(result, Mapping):
        return None
    text = result.get("text")
    if isinstance(text, str):
        return text
    content = result.get("content")
    if not isinstance(content, Sequence) or isinstance(content, (str, bytes)):
        return None
    parts: list[str] = []
    for block in content:
        if isinstance(block, Mapping) and isinstance(block.get("text"), str):
            parts.append(str(block["text"]))
    if not parts:
        return None
    return "\n".join(parts)


def _path_info(path: str, *, cwd: str | None) -> Mapping[str, object]:
    if path.startswith("/"):
        return {
            "original_path": path,
            "normalized_path": posixpath.normpath(path),
            "path_kind": "absolute",
            "cwd": cwd,
        }
    if cwd and cwd.startswith("/"):
        normalized = posixpath.normpath(posixpath.join(cwd, path))
        return {
            "original_path": path,
            "normalized_path": normalized,
            "path_kind": "cwd_relative",
            "cwd": cwd,
        }
    return {
        "original_path": path,
        "normalized_path": posixpath.normpath(path),
        "path_kind": "repo_relative",
        "cwd": cwd,
    }


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _aggregate_confidence(values: Iterable[str]) -> str:
    rank = {"high": 3, "medium": 2, "low": 1}
    best = "low"
    for value in values:
        if rank.get(value, 0) > rank.get(best, 0):
            best = value
    return best


def _loads(raw: object) -> object:
    if raw is None:
        return {}
    if isinstance(raw, (dict, list)):
        return raw
    if not isinstance(raw, str):
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def _first_str(*values: object) -> str | None:
    for value in values:
        if isinstance(value, str) and value:
            return value
    return None


def _mapping_str(value: object, key: str) -> str | None:
    if not isinstance(value, Mapping):
        return None
    raw = value.get(key)
    return raw if isinstance(raw, str) and raw else None


def _row_value(row: RowMapping, key: str) -> object:
    try:
        return row[key]
    except KeyError:
        return None


def _nested_mapping_str(value: object, parent: str, key: str) -> str | None:
    if not isinstance(value, Mapping):
        return None
    child = value.get(parent)
    return _mapping_str(child, key)


def _stable_id(*parts: str) -> str:
    raw = "\x1f".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _to_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
