"""Append-only ``EvalUnit`` store — the rollout/analysis contract (DESIGN §1)."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from .units import EvalUnit


class EvalUnitStore:
    """One JSONL file of ``EvalUnit`` rows. Writes append; reads return all rows.

    The store is deliberately dumb: rollouts append rows, analysis reads them.
    Adaptive-K (doc §9.3) is "append more rows, re-aggregate", so the store never
    rewrites or dedups on write — dedup is the caller's job via ``existing_cells``.
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, units: Iterable[EvalUnit]) -> int:
        rows = list(units)
        if not rows:
            return 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            for unit in rows:
                fh.write(json.dumps(unit.to_dict(), ensure_ascii=False) + "\n")
        return len(rows)

    def read_all(self) -> list[EvalUnit]:
        if not self.path.exists():
            return []
        units: list[EvalUnit] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            units.append(EvalUnit.from_dict(json.loads(line)))
        return units

    def existing_cells(self) -> set[tuple[str, str, int]]:
        """``(prefix_id, treatment_id, branch_seed)`` already in the store."""
        return {unit.cell_key for unit in self.read_all()}

    def count(self) -> int:
        if not self.path.exists():
            return 0
        return sum(
            1
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
