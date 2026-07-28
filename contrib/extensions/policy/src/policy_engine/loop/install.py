"""What a pass decided, written where the runtime reads it.

The stage that was missing. ``select`` produced ``selection.json`` and nothing
read it: an accepted candidate stayed a line in a report, and reaching a running
agent meant somebody copying YAML by hand. Four stages -- abstract, compile,
replay, select -- ran every pass and delivered nothing that could act.

Merged, not overwritten, and the two directions are not symmetric:

* **accepted** -- installed, replacing any earlier version of the same item.
* **rejected with evidence** -- removed. It was measured and it did not help.
* **undecided** -- left exactly as it was. A candidate whose replays were all
  lost has not been shown to be useless; removing it would let an infrastructure
  failure look like a finding.

The target file is deliberately a parameter with no default pointing at the
live checklist. What this writes is a proposal from a column that has yet to
move a graded score across twenty-one measured replays, so it goes to the
opt-in file and a human decides whether to load it. Closing the loop
mechanically and switching it on are two decisions, and only the first is made
here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from loguru import logger

from .contracts import CompiledCandidate, Verdict


@dataclass(slots=True, frozen=True)
class InstallReport:
    path: Path
    installed: tuple[str, ...] = ()
    removed: tuple[str, ...] = ()
    undecided: tuple[str, ...] = ()
    #: Accepted but with nothing to render -- a verdict whose candidate is not
    #: in the compiled artifact. Reported rather than skipped silently, because
    #: it means the two files came from different passes.
    unresolved: tuple[str, ...] = ()

    def summary(self) -> str:
        return (
            f"{len(self.installed)} installed, {len(self.removed)} removed, "
            f"{len(self.undecided)} undecided -> {self.path}"
        )


def to_item(compiled: CompiledCandidate) -> dict[str, object]:
    """One compiled candidate as the runtime's checklist reads it."""
    candidate = compiled.candidate
    when: dict[str, object] = {
        "checkpoint": candidate.checkpoint,
        "trigger": compiled.trigger,
    }
    if compiled.precondition:
        when["precondition"] = compiled.precondition
    return {
        "id": compiled.item_id or candidate.candidate_id,
        "dimension": candidate.dimension,
        "check": candidate.check,
        "advice": "",
        # Every candidate this column produces is a sentence for the agent to
        # read. A gate worth a blocking review is a different artifact and is
        # not something `abstract` currently knows how to write.
        "deliver": "inject",
        "when": when,
        "when_note": candidate.when_note,
    }


@dataclass(slots=True)
class _Existing:
    items: list[dict[str, object]] = field(default_factory=list)

    def index(self) -> dict[str, int]:
        return {str(item.get("id", "")): i for i, item in enumerate(self.items)}


def _read(path: Path) -> _Existing:
    if not path.is_file():
        return _Existing()
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        logger.error("install: cannot read {}: {}; treating as empty", path, exc)
        return _Existing()
    items = raw.get("items")
    if not isinstance(items, list):  # code-health: ignore[AM025]
        return _Existing()
    return _Existing(
        items=[i for i in items if isinstance(i, dict)]  # code-health: ignore[AM025]
    )


def install(
    verdicts: Sequence[Verdict],
    compiled: Sequence[CompiledCandidate],
    path: Path,
) -> InstallReport:
    """Apply a pass's verdicts to a checklist file."""
    by_candidate: Mapping[str, CompiledCandidate] = {
        c.candidate.candidate_id: c for c in compiled
    }
    existing = _read(path)
    index = existing.index()

    installed: list[str] = []
    removed: list[str] = []
    undecided: list[str] = []
    unresolved: list[str] = []

    for verdict in verdicts:
        found = by_candidate.get(verdict.candidate_id)
        item_id = (found.item_id or verdict.candidate_id) if found else ""

        if verdict.accepted:
            if found is None:
                unresolved.append(verdict.candidate_id)
                continue
            item = to_item(found)
            item_id = str(item["id"])
            at = index.get(item_id)
            if at is None:
                existing.items.append(item)
                index[item_id] = len(existing.items) - 1
            else:
                existing.items[at] = item
            installed.append(item_id)
        elif verdict.measured:
            at = index.get(item_id)
            if at is not None:
                existing.items.pop(at)
                index = existing.index()
            removed.append(item_id or verdict.candidate_id)
        else:
            undecided.append(item_id or verdict.candidate_id)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {"items": existing.items}, sort_keys=True, allow_unicode=True, width=120
        ),
        encoding="utf-8",
    )
    report = InstallReport(
        path=path,
        installed=tuple(installed),
        removed=tuple(removed),
        undecided=tuple(undecided),
        unresolved=tuple(unresolved),
    )
    for candidate_id in report.unresolved:
        logger.warning(
            "install: {} was accepted but is not in the compiled artifact; "
            "the two files are from different passes",
            candidate_id,
        )
    logger.info("install: {}", report.summary())
    return report


__all__ = ["InstallReport", "install", "to_item"]
