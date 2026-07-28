"""The driver, and nothing more.

Every stage is a function over one artifact. This module runs them in order and
writes each result to the run directory. It deliberately holds no logic the
stages do not: whatever ``run`` does, the six subcommands do identically, and
the artifacts either way are the same. That is what makes a bad stage
debuggable -- re-run it alone against the file its predecessor wrote, or against
one edited by hand.

It knows a ``CaseSource`` and a ``ReplayBackend`` and nothing about which
benchmark they are. Ordering is not a preference: each stage reads what the one
before it wrote. Nothing here needs the batch's environments to still exist --
replay rebuilds a workspace from the recorded trajectory -- but diagnose is
better off inside the window, because there the attempt's own machine can still
be forked instead of reconstructed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

from .abstract import abstract
from .collect import collect
from .compile_gate import compile_candidates
from .contracts import (
    Candidate,
    CompiledCandidate,
    Diagnosis,
    FailureCase,
    ReplayMeasurement,
    RepositoryNote,
    Verdict,
    read_artifact,
    write_artifact,
)
from .diagnose import diagnose
from .protocols import CaseSource, ReplayBackend
from .replay import replay
from .select import select

#: This package's own directory. Split from parent access so the symlink
#: resolution is one explicit step.
_PACKAGE_ROOT = Path(__file__).parent.parent / "runtime"

#: One file per stage. Names what ``run`` writes where; the subcommands take an
#: explicit ``--out``, so running them by hand reproduces a chained run only if
#: you point them at these names.
ARTIFACTS: Mapping[str, str] = {
    "collect": "cases.json",
    "diagnose": "diagnoses.json",
    "abstract": "candidates.json",
    "compile": "compiled.json",
    "replay": "measurements.json",
    "select": "selection.json",
}

#: The predicate vocabulary a compile stage gates against, when none is named.
DEFAULT_VOCABULARY = _PACKAGE_ROOT / "vocabulary.yaml"


@dataclass(slots=True)
class LoopPaths:
    """Where one pass through the loop keeps its work."""

    root: Path

    @staticmethod
    def for_run(base: Path, run_id: str = "") -> LoopPaths:
        stamp = run_id or datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
        return LoopPaths(root=base / stamp)

    def artifact(self, stage: str) -> Path:
        return self.root / ARTIFACTS[stage]


@dataclass(slots=True)
class LoopConfig:
    """What one pass needs. The benchmark arrives as two objects; nothing here
    names it."""

    batch: str
    paths: LoopPaths
    source: CaseSource
    backend: ReplayBackend | None = None
    vocabulary_path: Path | None = None
    provider: str = ""
    user_config: str = ""
    concurrency: int = 4
    only: str = ""


async def run(config: LoopConfig) -> list[Verdict]:
    """A batch on disk to accepted candidates, writing every intermediate."""
    config.paths.root.mkdir(parents=True, exist_ok=True)
    logger.info(
        "loop: {} [{}] -> {}", config.batch, config.source.name, config.paths.root
    )

    cases = collect(config.source, config.batch, only=config.only)
    write_artifact(config.paths.artifact("collect"), cases)
    if not cases:
        return []

    diagnoses = await diagnose(
        cases,
        provider=config.provider,
        user_config=config.user_config,
        concurrency=config.concurrency,
        source=config.source,
    )
    write_artifact(config.paths.artifact("diagnose"), diagnoses)

    candidates = await abstract(
        diagnoses,
        provider=config.provider,
        user_config=config.user_config,
        concurrency=config.concurrency,
    )
    write_artifact(config.paths.artifact("abstract"), candidates)
    if not candidates:
        logger.info("loop: no candidate survived abstraction; nothing to measure")
        return []

    compiled = await compile_candidates(
        candidates,
        vocabulary_path=config.vocabulary_path or DEFAULT_VOCABULARY,
        provider=config.provider,
        user_config=config.user_config,
        concurrency=config.concurrency,
    )
    write_artifact(config.paths.artifact("compile"), compiled)

    if config.backend is None:
        logger.warning("loop: no replay backend; candidates cannot be measured")
        return []

    # ``replay`` owns this artifact: it writes before the first arm and after
    # every one, so a stop mid-batch still leaves the measurements taken.
    measurements = replay(
        config.backend,
        compiled,
        cases,
        out_path=config.paths.artifact("replay"),
    )

    verdicts = select(measurements)
    write_artifact(config.paths.artifact("select"), verdicts)
    return verdicts


# -- reading artifacts back, for standalone stages ----------------------------


def load_cases(path: Path) -> list[FailureCase]:
    return [FailureCase.from_json(raw) for raw in read_artifact(path)]


def load_diagnoses(path: Path) -> list[Diagnosis]:
    return [Diagnosis.from_json(raw) for raw in read_artifact(path)]


def load_candidates(path: Path) -> list[Candidate]:
    return [Candidate.from_json(raw) for raw in read_artifact(path)]


def load_compiled(path: Path) -> list[CompiledCandidate]:
    return [CompiledCandidate.from_json(raw) for raw in read_artifact(path)]


def load_measurements(path: Path) -> list[ReplayMeasurement]:
    return [ReplayMeasurement.from_json(raw) for raw in read_artifact(path)]


def load_notes(path: Path) -> list[RepositoryNote]:
    return [RepositoryNote.from_json(raw) for raw in read_artifact(path)]


def summarise(verdicts: Sequence[Verdict]) -> str:
    if not verdicts:
        return "no candidates measured"
    lines = []
    for verdict in verdicts:
        mark = "keep" if verdict.accepted else "drop"
        detail = ""
        if verdict.improved_cases:
            detail += f"  improved={list(verdict.improved_cases)}"
        if verdict.lost_cases:
            detail += f"  lost={list(verdict.lost_cases)}"
        lines.append(f"  [{mark}] {verdict.candidate_id}  {verdict.reason}{detail}")
    return "\n".join(lines)


__all__ = [
    "ARTIFACTS",
    "DEFAULT_VOCABULARY",
    "LoopConfig",
    "LoopPaths",
    "load_candidates",
    "load_cases",
    "load_compiled",
    "load_diagnoses",
    "load_measurements",
    "run",
    "summarise",
]
