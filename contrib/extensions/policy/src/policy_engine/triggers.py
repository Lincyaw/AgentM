# code-health: ignore-file[AM025] -- checklist YAML is untyped at the boundary
"""Trigger layer: checklist items bound to trajectory signals.

Two tiers, decided by the 2026-07-24 emission audit:

- ``structural`` items fire deterministically from ``TrajectoryState``
  signals and inject a templated message quoting the agent's own commands.
  Only signals with zero pass-disturbance across the calibration corpus
  may live in this tier.
- ``critic`` items are questions no token-shape rule can answer; when a
  session reaches the stop decision they are handed to the delivery layer
  (self-check injection or subagent review) together with the noisy
  structural evidence (unresolved reds, repeated failures).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml
from loguru import logger

from .signals import TrajectoryState


@dataclass(slots=True, frozen=True)
class ChecklistItem:
    item_id: str
    dimension: str
    check: str
    advice: str
    tier: str  # "structural" | "critic"
    checkpoint: str  # "continuous" | "stop"


@dataclass(slots=True, frozen=True)
class Firing:
    item: ChecklistItem
    facts: tuple[str, ...]
    suggestion: str


def load_items(path: Path) -> dict[str, ChecklistItem]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        logger.error("policy triggers: cannot load {}: {}", path, exc)
        return {}
    items: dict[str, ChecklistItem] = {}
    for entry in (raw or {}).get("items", []):
        if not isinstance(entry, Mapping):
            continue
        item = ChecklistItem(
            item_id=str(entry.get("id", "")),
            dimension=str(entry.get("dimension", "")),
            check=" ".join(str(entry.get("check", "")).split()),
            advice=" ".join(str(entry.get("advice", "")).split()),
            tier=str(entry.get("tier", "critic")),
            checkpoint=str(entry.get("checkpoint", "stop")),
        )
        if item.item_id:
            items[item.item_id] = item
    return items


def _clip(text: str, limit: int = 110) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def render_message(firing: Firing) -> str:
    lines = [f"Process check from the validation monitor ({firing.item.dimension}):"]
    lines.append(firing.item.check)
    if firing.facts:
        lines.append("")
        lines.append("Observed in this session:")
        lines.extend(f"- {fact}" for fact in firing.facts)
    if firing.suggestion:
        lines.append("")
        lines.append(firing.suggestion)
    if firing.item.advice:
        lines.append(firing.item.advice)
    return "\n".join(lines)


@dataclass(slots=True)
class StructuralTriggers:
    """Evaluates the structural tier; each item latches after one firing."""

    items: dict[str, ChecklistItem]
    min_narrow_runs: int = 2
    _fired: set[str] | None = None

    def _latched(self, item_id: str) -> bool:
        if self._fired is None:
            self._fired = set()
        return item_id in self._fired

    def _latch(self, item_id: str) -> None:
        if self._fired is None:
            self._fired = set()
        self._fired.add(item_id)

    def evaluate(self, state: TrajectoryState, *, stopping: bool) -> Firing | None:
        firing = self._narrow_only(state)
        if firing is None and stopping:
            firing = self._self_authored_only(state) or self._under_validation(state)
        if firing is not None:
            self._latch(firing.item.item_id)
        return firing

    def _narrow_only(self, state: TrajectoryState) -> Firing | None:
        item = self.items.get("narrow_only_validation")
        if item is None or item.tier != "structural" or self._latched(item.item_id):
            return None
        groups = state.narrow_only_groups(min_runs=self.min_narrow_runs)
        if not groups:
            return None
        facts: list[str] = []
        for group in groups[:3]:
            for run in group.runs[-2:]:
                facts.append(
                    f"`{_clip(run.segment.raw)}` "
                    f"(narrowing tokens: {', '.join(run.selectors[:4])})"
                )
        shortest = min(
            (run for group in groups for run in group.runs),
            key=lambda run: len(run.segment.ordered),
        )
        suggestion = (
            f"For example, re-run `{_clip(shortest.segment.raw)}` without "
            f"{', '.join(shortest.selectors[:3])}."
        )
        return Firing(item=item, facts=tuple(facts), suggestion=suggestion)

    def _self_authored_only(self, state: TrajectoryState) -> Firing | None:
        item = self.items.get("self_authored_oracle_only")
        if item is None or item.tier != "structural" or self._latched(item.item_id):
            return None
        evidence = state.self_authored_only()
        if evidence is None:
            return None
        facts = [
            f"test files you edited: {', '.join(evidence.edited_test_stems[:4])}",
            *(
                f"green run: `{_clip(segment.raw)}`"
                for segment in evidence.green_runs[-2:]
            ),
        ]
        return Firing(item=item, facts=tuple(facts), suggestion="")

    def _under_validation(self, state: TrajectoryState) -> Firing | None:
        item = self.items.get("under_validation_at_stop")
        if item is None or item.tier != "structural" or self._latched(item.item_id):
            return None
        evidence = state.under_validation()
        if evidence is None:
            return None
        facts = [
            f"{evidence.mutation_count} file modifications vs "
            f"{evidence.validation_count} validation runs",
            *(
                f"validation run: `{_clip(segment.raw)}`"
                for segment in evidence.validations[-2:]
            ),
        ]
        return Firing(item=item, facts=tuple(facts), suggestion="")

    def critic_evidence(self, state: TrajectoryState) -> tuple[str, ...]:
        """Noisy-signal digest handed to the critic at stop time."""

        facts: list[str] = []
        for record in state.unresolved_reds()[-3:]:
            facts.append(
                f"failed run never superseded: `{_clip(record.raw)}` "
                f"(exit {record.exit_code})"
            )
        for key, count in state.repeated_failures()[:3]:
            facts.append(f"repeated failure x{count}: {key}")
        return tuple(facts)

    def critic_items(self) -> tuple[ChecklistItem, ...]:
        return tuple(item for item in self.items.values() if item.tier == "critic")
