"""Filesystem helpers shared by artifact-aware extensions and scenarios."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, cast

from loguru import logger

if TYPE_CHECKING:
    from agentm.core.abi.project_layout import ProjectLayout


class ArtifactCreator(TypedDict, total=False):
    session_id: str
    task_id: str
    persona: str | None
    timestamp: float


class ArtifactMetadata(TypedDict, total=False):
    artifact_id: str
    kind: str
    title: str
    slug: str
    path: str
    parent_id: str | None
    parent_artifact_ids: list[str]
    tags: list[str]
    created_by: ArtifactCreator


def artifacts_dir_for(layout: "ProjectLayout", root_session_id: str) -> Path:
    """Return the artifacts directory for ``root_session_id`` under ``layout``."""

    return layout.artifacts_root(root_session_id)


def find_metadata_files(artifacts_dir: Path, artifact_id: str) -> list[Path]:
    if not artifacts_dir.exists():
        return []
    pattern = f"{artifact_id}__*.meta.json"
    return sorted(artifacts_dir.glob(pattern))


def scan_artifact_metadata(artifacts_dir: Path) -> list[ArtifactMetadata]:
    if not artifacts_dir.exists():
        return []
    metas: list[ArtifactMetadata] = []
    for meta_path in sorted(artifacts_dir.glob("*.meta.json")):
        try:
            raw = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("artifact_files: skipping corrupt metadata {}: {}", meta_path, exc)
            continue
        if isinstance(raw, dict):
            metas.append(cast(ArtifactMetadata, raw))
    return metas


def list_artifacts_for_task(
    *,
    layout: "ProjectLayout",
    root_session_id: str,
    task_id: str,
) -> list[ArtifactMetadata]:
    matches: list[ArtifactMetadata] = []
    for meta in scan_artifact_metadata(artifacts_dir_for(layout, root_session_id)):
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
    "ArtifactCreator",
    "ArtifactMetadata",
    "artifacts_dir_for",
    "find_metadata_files",
    "list_artifacts_for_task",
    "scan_artifact_metadata",
]
