"""§11 atom + helper: sample-id binding for distill data collection.

The atom is mounted on the **main agent** session (the one running the
target scenario, e.g. rca). At install time it reads the sample id from
config-or-env and writes a small meta sidecar next to the replay log so
the offline labeler can join records → GT.

Mount::

    LLMHARNESS_DISTILL_SAMPLE_ID=ts0-mysql-corrupt-kwx8n5 \\
    LLMHARNESS_DISTILL_DATASET=/path/to/data.jsonl \\
    agentm --extension llmharness.adapters.agentm \\
           --extension llmharness.distill.binding ...

§11 single-file: no atom-to-atom imports, no core.runtime imports,
trivial install side effect.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentm.core.abi.extension import ExtensionAPI
from agentm.extensions import ExtensionManifest

_logger = logging.getLogger(__name__)

MANIFEST = ExtensionManifest(
    name="distill_binding",
    description=(
        "Write a sample-id sidecar next to the replay log so the offline "
        "distill labeler can join trajectory records to ground truth."
    ),
    registers=(),
    config_schema={
        "type": "object",
        "properties": {
            "sample_id": {"type": "string"},
            "dataset_name": {"type": "string"},
            "dataset_path": {"type": "string"},
        },
        "additionalProperties": False,
    },
    api_version=1,
    tier=1,
)


_SAMPLE_ID_ENV = "LLMHARNESS_DISTILL_SAMPLE_ID"
_DATASET_NAME_ENV = "LLMHARNESS_DISTILL_DATASET_NAME"
_DATASET_PATH_ENV = "LLMHARNESS_DISTILL_DATASET"


def meta_path_for(cwd: str | os.PathLike[str], root_session_id: str) -> Path:
    """Canonical sidecar path. Mirrors ``replay.record.replay_log_path``."""
    return Path(cwd) / ".agentm" / "audit_replay" / f"{root_session_id}.meta.json"


@dataclass(frozen=True)
class SampleMeta:
    sample_id: str
    dataset_name: str
    dataset_path: str
    root_session_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "dataset_name": self.dataset_name,
            "dataset_path": self.dataset_path,
            "root_session_id": self.root_session_id,
        }


def read_sample_meta(path: Path) -> SampleMeta | None:
    """Read a meta sidecar. Returns None on missing/malformed file."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    sample_id = raw.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        return None
    return SampleMeta(
        sample_id=sample_id,
        dataset_name=str(raw.get("dataset_name") or ""),
        dataset_path=str(raw.get("dataset_path") or ""),
        root_session_id=str(raw.get("root_session_id") or ""),
    )


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    sample_id = config.get("sample_id") or os.environ.get(_SAMPLE_ID_ENV) or ""
    if not sample_id:
        _logger.debug(
            "distill_binding mounted without sample_id; meta sidecar will not be written"
        )
        return
    dataset_name = (
        config.get("dataset_name") or os.environ.get(_DATASET_NAME_ENV) or ""
    )
    dataset_path = (
        config.get("dataset_path") or os.environ.get(_DATASET_PATH_ENV) or ""
    )
    meta = SampleMeta(
        sample_id=str(sample_id),
        dataset_name=str(dataset_name),
        dataset_path=str(dataset_path),
        root_session_id=api.root_session_id,
    )
    path = meta_path_for(api.cwd, api.root_session_id)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(meta.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError:
        _logger.warning("distill_binding sidecar write failed: %s", path, exc_info=True)


__all__ = [
    "MANIFEST",
    "SampleMeta",
    "install",
    "meta_path_for",
    "read_sample_meta",
]
