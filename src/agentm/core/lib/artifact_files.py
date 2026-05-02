"""Filesystem helpers shared by artifact-aware extensions and scenarios."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def artifacts_dir_for(cwd: Path, root_session_id: str) -> Path:
    return cwd / ".agentm" / "artifacts" / root_session_id


def find_metadata_files(artifacts_dir: Path, artifact_id: str) -> list[Path]:
    if not artifacts_dir.exists():
        return []
    pattern = f"{artifact_id}__*.meta.json"
    return sorted(artifacts_dir.glob(pattern))


def scan_artifact_metadata(artifacts_dir: Path) -> list[dict[str, Any]]:
    if not artifacts_dir.exists():
        return []
    metas: list[dict[str, Any]] = []
    for meta_path in sorted(artifacts_dir.glob("*.meta.json")):
        try:
            metas.append(json.loads(meta_path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return metas


def list_artifacts_for_task(
    *,
    cwd: Path,
    root_session_id: str,
    task_id: str,
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for meta in scan_artifact_metadata(artifacts_dir_for(cwd, root_session_id)):
        created_by = meta.get("created_by")
        if not isinstance(created_by, dict):
            continue
        if created_by.get("task_id") != task_id:
            continue
        matches.append(meta)
    matches.sort(
        key=lambda meta: float(meta.get("created_by", {}).get("timestamp", 0.0))
    )
    return matches


__all__ = [
    "artifacts_dir_for",
    "find_metadata_files",
    "list_artifacts_for_task",
    "scan_artifact_metadata",
]
