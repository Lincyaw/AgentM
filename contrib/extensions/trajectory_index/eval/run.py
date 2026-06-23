"""Eval runner for the trajectory-index entity extractor.

Usage:
    uv run python contrib/extensions/trajectory_index/eval/run.py [--model qwen] [--cases cases/]
"""
from __future__ import annotations

import asyncio
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

import typer
import yaml
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from trajectory_index.agents.entity_extractor.schema import (
    ExtractedEntity,
    ReportEntitiesParams,
)
from trajectory_index.data import JsonValue, ProviderSpec, extract, resolve_provider

# ---------------------------------------------------------------------------
# Case loading
# ---------------------------------------------------------------------------


@dataclass
class EvalCase:
    id: str
    description: str
    messages: list[dict[str, JsonValue]]
    expected_entities: list[dict[str, str]]
    expected_relations: list[dict[str, str]]

    @classmethod
    def from_yaml(cls, path: Path) -> EvalCase:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls(
            id=data["id"],
            description=data.get("description", ""),
            messages=data["messages"],
            expected_entities=data.get("expected", {}).get("entities", []),
            expected_relations=data.get("expected", {}).get("relations", []),
        )


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------


@dataclass
class GradeResult:
    case_id: str
    expected_entity_count: int = 0
    matched_entity_count: int = 0
    extra_entity_count: int = 0
    entity_kind_matches: int = 0
    entity_kind_total: int = 0
    expected_relation_count: int = 0
    matched_relation_count: int = 0
    total_extracted_entities: int = 0
    total_extracted_mentions: int = 0
    total_extracted_relations: int = 0
    latency_s: float = 0.0
    error: str | None = None
    missed_entities: list[str] = field(default_factory=list)
    extra_entities: list[str] = field(default_factory=list)
    missed_relations: list[str] = field(default_factory=list)

    @property
    def entity_recall(self) -> float:
        return self.matched_entity_count / self.expected_entity_count if self.expected_entity_count else 1.0

    @property
    def entity_precision(self) -> float:
        total = self.matched_entity_count + self.extra_entity_count
        return self.matched_entity_count / total if total else 1.0

    @property
    def relation_recall(self) -> float:
        return self.matched_relation_count / self.expected_relation_count if self.expected_relation_count else 1.0

    @property
    def kind_accuracy(self) -> float:
        return self.entity_kind_matches / self.entity_kind_total if self.entity_kind_total else 1.0


def _normalize(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def grade(case: EvalCase, result: ReportEntitiesParams) -> GradeResult:
    g = GradeResult(case_id=case.id)
    g.total_extracted_entities = len(result.entities)
    g.total_extracted_mentions = len(result.mentions)
    g.total_extracted_relations = len(result.relations)

    extracted_by_norm: dict[str, ExtractedEntity] = {}
    for ent in result.entities:
        extracted_by_norm[_normalize(ent.name)] = ent
        for alias in ent.aliases:
            extracted_by_norm[_normalize(alias)] = ent

    expected_names = set()
    g.expected_entity_count = len(case.expected_entities)
    for exp in case.expected_entities:
        norm = _normalize(exp["name"])
        expected_names.add(norm)
        if norm in extracted_by_norm:
            g.matched_entity_count += 1
            g.entity_kind_total += 1
            if extracted_by_norm[norm].kind == exp.get("kind", ""):
                g.entity_kind_matches += 1
        else:
            g.missed_entities.append(exp["name"])

    for ent in result.entities:
        if _normalize(ent.name) not in expected_names:
            g.extra_entity_count += 1
            g.extra_entities.append(f"{ent.name} ({ent.kind})")

    extracted_rels: set[tuple[str, str, str]] = set()
    for rel in result.relations:
        extracted_rels.add((_normalize(rel.from_entity), _normalize(rel.to_entity), _normalize(rel.relation_type)))

    g.expected_relation_count = len(case.expected_relations)
    for exp in case.expected_relations:
        key = (_normalize(exp["from"]), _normalize(exp["to"]), _normalize(exp["type"]))
        if key in extracted_rels:
            g.matched_relation_count += 1
        else:
            g.missed_relations.append(f"{exp['from']} --{exp['type']}--> {exp['to']}")

    return g


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


async def run_case(
    case: EvalCase,
    provider: ProviderSpec,
) -> GradeResult:
    t0 = time.monotonic()
    try:
        result = await extract(case.messages, provider)
        if result is None:
            g = GradeResult(case_id=case.id, error="no parseable JSON in response")
            g.latency_s = time.monotonic() - t0
            return g
        g = grade(case, result)
        g.latency_s = time.monotonic() - t0
        return g
    except Exception as exc:
        g = GradeResult(case_id=case.id, error=str(exc))
        g.latency_s = time.monotonic() - t0
        return g


def log_report(grades: list[GradeResult]) -> None:
    logger.info("=" * 60)
    for g in grades:
        if g.error:
            logger.error(f"[{g.case_id}] ERROR: {g.error}  ({g.latency_s:.1f}s)")
            continue
        logger.info(
            f"[{g.case_id}] "
            f"ent={g.matched_entity_count}/{g.expected_entity_count} "
            f"recall={g.entity_recall:.0%} prec={g.entity_precision:.0%} "
            f"kind_acc={g.kind_accuracy:.0%}  "
            f"rel={g.matched_relation_count}/{g.expected_relation_count} "
            f"recall={g.relation_recall:.0%}  "
            f"{g.latency_s:.1f}s"
        )
        if g.missed_entities:
            logger.warning(f"  [{g.case_id}] MISSED ent: {', '.join(g.missed_entities)}")
        if g.extra_entities:
            logger.warning(f"  [{g.case_id}] EXTRA ent:  {', '.join(g.extra_entities)}")
        if g.missed_relations:
            logger.warning(f"  [{g.case_id}] MISSED rel: {', '.join(g.missed_relations)}")

    valid = [g for g in grades if g.error is None]
    if not valid:
        logger.error("No successful runs.")
        return
    logger.info(
        f"AGGREGATE ({len(valid)}/{len(grades)} passed) "
        f"ent_recall={sum(g.entity_recall for g in valid) / len(valid):.0%} "
        f"ent_prec={sum(g.entity_precision for g in valid) / len(valid):.0%} "
        f"kind_acc={sum(g.kind_accuracy for g in valid) / len(valid):.0%} "
        f"rel_recall={sum(g.relation_recall for g in valid) / len(valid):.0%} "
        f"avg_lat={sum(g.latency_s for g in valid) / len(valid):.1f}s "
        f"errors={sum(1 for g in grades if g.error)}"
    )
    logger.info("=" * 60)


async def async_main(model: str, cases_dir: Path, concurrency: int) -> list[GradeResult]:
    provider = resolve_provider(model)
    case_files = sorted(cases_dir.glob("*.yaml"))
    if not case_files:
        logger.error(f"No YAML cases found in {cases_dir}")
        return []

    cases = [EvalCase.from_yaml(p) for p in case_files]
    logger.info(f"model={model} cases={len(cases)} concurrency={concurrency}")

    sem = asyncio.Semaphore(concurrency)

    async def _run_with_sem(case: EvalCase) -> GradeResult:
        async with sem:
            logger.info(f"[{case.id}] starting...")
            g = await run_case(case, provider)
            if g.error:
                logger.error(f"[{case.id}] failed: {g.error}")
            else:
                logger.info(
                    f"[{case.id}] done: {g.matched_entity_count}/{g.expected_entity_count} ent, "
                    f"{g.matched_relation_count}/{g.expected_relation_count} rel, "
                    f"{g.latency_s:.1f}s"
                )
            return g

    grades = await asyncio.gather(*[_run_with_sem(c) for c in cases])
    log_report(list(grades))
    return list(grades)


app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def main(
    model: Annotated[str, typer.Option(help="config.toml model profile name")] = "qwen",
    cases: Annotated[Path, typer.Option(help="YAML cases directory")] = Path(__file__).parent / "cases",
    concurrency: Annotated[int, typer.Option(help="Max concurrent cases")] = 2,
) -> None:
    """Run entity extraction eval against the trajectory-index extractor agent."""
    asyncio.run(async_main(model, cases, concurrency))


if __name__ == "__main__":
    app()
