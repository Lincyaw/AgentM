"""Trajectory corpus: baseline RCA sessions paired with their GT case dirs.

A trajectory is a recorded baseline session (the fixed actor pi_A's full RCA
investigation). The corpus pairs each one with its ground-truth case directory
(holding ``causal_graph_verified.json`` / ``injection.json`` / parquet) and a
``repository_id`` used as the statistical split key (doc §9.4).
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from agentm.core.abi import AgentMessage
from agentm.core.abi.session_store import SessionState, SessionStore


@dataclass(frozen=True)
class TrajectoryRef:
    """One baseline trajectory + its GT case directory."""

    trajectory_id: str  # source session id
    case_id: str
    data_dir: str  # GT case directory for the judge
    repository_id: str
    actor_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "case_id": self.case_id,
            "data_dir": self.data_dir,
            "repository_id": self.repository_id,
            "actor_id": self.actor_id,
            "metadata": self.metadata,
        }


def load_corpus(path: Path, *, data_root: Path | None = None) -> list[TrajectoryRef]:
    """Load a corpus manifest (YAML / JSON / CSV / TSV).

    Each row needs ``trajectory_id`` and ``case_id``. ``data_dir`` may be given
    explicitly, else it is resolved as ``data_root / case_id``. ``repository_id``
    defaults to ``case_id`` when absent (one cluster per case).
    """

    rows = _read_rows(path)
    refs: list[TrajectoryRef] = []
    for raw in rows:
        trajectory_id = str(raw.get("trajectory_id") or raw.get("session_id") or "").strip()
        case_id = str(raw.get("case_id") or "").strip()
        if not trajectory_id or not case_id:
            raise ValueError(f"corpus row missing trajectory_id/case_id: {raw!r}")
        data_dir = str(raw.get("data_dir") or "").strip()
        if not data_dir:
            if data_root is None:
                raise ValueError(
                    f"row {case_id!r} has no data_dir and no --data-root was given"
                )
            data_dir = str(data_root / case_id)
        refs.append(
            TrajectoryRef(
                trajectory_id=trajectory_id,
                case_id=case_id,
                data_dir=data_dir,
                repository_id=str(raw.get("repository_id") or case_id).strip(),
                actor_id=str(raw.get("actor_id") or "").strip(),
                metadata=_row_metadata(raw),
            )
        )
    return refs


def load_trajectory_messages(
    ref: TrajectoryRef, *, store: SessionStore
) -> list[AgentMessage]:
    """Load the recorded message list for one baseline trajectory."""

    source = store.open(ref.trajectory_id)
    return _source_messages(source)


def _source_messages(source: SessionState) -> list[AgentMessage]:
    raw_getter = getattr(source, "get_raw_messages", None)
    if callable(raw_getter):
        return list(cast(list[AgentMessage], raw_getter()))
    return list(source.build_session_context().messages)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix == ".json":
        data = json.loads(text)
        rows = data.get("trajectories", data) if isinstance(data, dict) else data
        if not isinstance(rows, list):
            raise ValueError(f"{path} JSON must be a list or have a 'trajectories' list")
        return [dict(row) for row in rows]
    if suffix in {".yaml", ".yml"}:
        import yaml

        data = yaml.safe_load(text)
        rows = data.get("trajectories", data) if isinstance(data, dict) else data
        if not isinstance(rows, list):
            raise ValueError(f"{path} YAML must be a list or have a 'trajectories' list")
        return [dict(row) for row in rows]
    # CSV / TSV
    delimiter = "\t" if suffix in {".tsv", ".txt"} else ","
    reader = csv.DictReader(text.splitlines(), delimiter=delimiter)
    return [dict(row) for row in reader]


def _row_metadata(raw: dict[str, Any]) -> dict[str, Any]:
    known = {
        "trajectory_id",
        "session_id",
        "case_id",
        "data_dir",
        "repository_id",
        "actor_id",
    }
    return {key: value for key, value in raw.items() if key not in known}
