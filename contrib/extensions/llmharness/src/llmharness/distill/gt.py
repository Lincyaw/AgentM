"""Ground-truth loader keyed by sample_id (rca dataset shape).

The rca dataset (``/home/ddq/AoyangSpace/dataset/rca/data.jsonl``) has
rows like::

    {"id": 5, "source": "ts0-mysql-corrupt-kwx8n5",
     "ground_truth": ["mysql", "ts-station-service"],
     "fault_type": "NetworkCorrupt", "fault_category": "NetworkChaos",
     "datapack_name": "ts0-mysql-corrupt-kwx8n5", ...}

``sample_id`` ≡ ``source`` ≡ ``datapack_name``. The labeler joins each
replay record's meta sidecar (see :mod:`llmharness.distill.binding`) to
a :class:`GroundTruth` instance here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GroundTruth:
    sample_id: str
    root_causes: tuple[str, ...]
    fault_type: str
    fault_category: str
    extra: dict[str, Any] = field(default_factory=dict)

    def to_prompt_block(self) -> str:
        """Compact string the oracle prompt embeds verbatim."""
        lines = [
            f"sample_id: {self.sample_id}",
            f"root_causes: {list(self.root_causes)}",
            f"fault_type: {self.fault_type}",
            f"fault_category: {self.fault_category}",
        ]
        return "\n".join(lines)


def load_dataset(jsonl_path: Path) -> dict[str, GroundTruth]:
    """Index a rca-shaped JSONL by ``source``. Skips malformed rows."""
    out: dict[str, GroundTruth] = {}
    with jsonl_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            sample_id = row.get("source") or row.get("datapack_name")
            if not isinstance(sample_id, str):
                continue
            raw_rc = row.get("ground_truth") or []
            if not isinstance(raw_rc, list):
                raw_rc = []
            out[sample_id] = GroundTruth(
                sample_id=sample_id,
                root_causes=tuple(str(x) for x in raw_rc),
                fault_type=str(row.get("fault_type") or ""),
                fault_category=str(row.get("fault_category") or ""),
                extra={
                    "answer": row.get("answer"),
                    "difficulty_spl": row.get("difficulty_spl"),
                    "difficulty_n_svc": row.get("difficulty_n_svc"),
                    "tags": row.get("tags"),
                },
            )
    return out


__all__ = ["GroundTruth", "load_dataset"]
