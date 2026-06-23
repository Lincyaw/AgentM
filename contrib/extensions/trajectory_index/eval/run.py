"""Eval runner for the trajectory-index entity extractor.

Usage:
    uv run python contrib/extensions/trajectory_index/eval/run.py [--model qwen] [--cases cases/] [--concurrency 2]

Runs each YAML case through the extraction agent, grades against expected
entities/relations, and logs a per-case + aggregate report.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml
from loguru import logger

# Ensure the workspace packages are importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from agentm.core.abi import AssistantMessage, LoopConfig, TextContent  # noqa: E402
from agentm.core.abi.session_config import AgentSessionConfig  # noqa: E402
from agentm.core.runtime.session import AgentSession  # noqa: E402

from trajectory_index.agents import extractor_scenario  # noqa: E402
from trajectory_index.agents.entity_extractor.schema import ReportEntitiesParams  # noqa: E402

# Truncate step content to stay within context limits.
# Qwen 3.5 2B has a 32K window; system prompt + tool schema + retry history
# eat ~6-10K tokens; 16K chars ≈ 8K tokens prompt leaves room for retries.
_MAX_STEP_CONTENT_CHARS = 500
_MAX_TOTAL_PROMPT_CHARS = 16_000


# ---------------------------------------------------------------------------
# Provider resolution (same pattern as the atom)
# ---------------------------------------------------------------------------


def _resolve_provider(model_name: str) -> tuple[str, dict[str, Any]]:
    from agentm.ai import DEFAULT_PROVIDER_DESCRIPTORS
    from agentm.core.lib import resolve_model_profile

    profile = resolve_model_profile(model_name)
    if profile is None:
        raise RuntimeError(f"model profile {model_name!r} not found in config.toml")
    for desc in DEFAULT_PROVIDER_DESCRIPTORS:
        if desc.id == profile.provider and desc.extension_module:
            return (desc.extension_module, dict(profile.to_build_config()))
    raise RuntimeError(f"no extension module for provider {profile.provider!r}")


# ---------------------------------------------------------------------------
# Case loading
# ---------------------------------------------------------------------------


@dataclass
class EvalCase:
    id: str
    description: str
    steps: list[dict[str, Any]]
    expected_entities: list[dict[str, str]]
    expected_relations: list[dict[str, str]]

    @classmethod
    def from_yaml(cls, path: Path) -> "EvalCase":
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls(
            id=data["id"],
            description=data.get("description", ""),
            steps=data["steps"],
            expected_entities=data.get("expected", {}).get("entities", []),
            expected_relations=data.get("expected", {}).get("relations", []),
        )


def _truncate_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Truncate step content to fit within context limits."""
    out = []
    for s in steps:
        s = dict(s)
        content = s.get("content", "")
        if len(content) > _MAX_STEP_CONTENT_CHARS:
            s["content"] = content[:_MAX_STEP_CONTENT_CHARS] + "..."
        out.append(s)

    prompt = json.dumps(out, ensure_ascii=False)
    if len(prompt) > _MAX_TOTAL_PROMPT_CHARS:
        # Drop steps from the middle to stay within budget
        while len(out) > 3 and len(json.dumps(out, ensure_ascii=False)) > _MAX_TOTAL_PROMPT_CHARS:
            mid = len(out) // 2
            out.pop(mid)
        logger.warning(f"truncated to {len(out)} steps to fit context window")

    return out


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------


@dataclass
class GradeResult:
    case_id: str
    # entity matching
    expected_entity_count: int = 0
    matched_entity_count: int = 0
    extra_entity_count: int = 0
    entity_kind_matches: int = 0
    entity_kind_total: int = 0
    # relation matching
    expected_relation_count: int = 0
    matched_relation_count: int = 0
    extra_relation_count: int = 0
    # meta
    total_extracted_entities: int = 0
    total_extracted_mentions: int = 0
    total_extracted_relations: int = 0
    tool_calls: int = 0
    retries: int = 0
    latency_s: float = 0.0
    error: str | None = None
    # detail
    matched_entities: list[str] = field(default_factory=list)
    missed_entities: list[str] = field(default_factory=list)
    extra_entities: list[str] = field(default_factory=list)
    matched_relations: list[str] = field(default_factory=list)
    missed_relations: list[str] = field(default_factory=list)

    @property
    def entity_recall(self) -> float:
        return (
            self.matched_entity_count / self.expected_entity_count
            if self.expected_entity_count
            else 1.0
        )

    @property
    def entity_precision(self) -> float:
        total = self.matched_entity_count + self.extra_entity_count
        return self.matched_entity_count / total if total else 1.0

    @property
    def relation_recall(self) -> float:
        return (
            self.matched_relation_count / self.expected_relation_count
            if self.expected_relation_count
            else 1.0
        )

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

    # --- Entity matching (by normalized name) ---
    extracted_by_norm: dict[str, Any] = {}
    for ent in result.entities:
        key = _normalize(ent.name)
        extracted_by_norm[key] = ent
        for alias in ent.aliases:
            extracted_by_norm[_normalize(alias)] = ent

    expected_names = set()
    g.expected_entity_count = len(case.expected_entities)
    for exp in case.expected_entities:
        norm = _normalize(exp["name"])
        expected_names.add(norm)
        if norm in extracted_by_norm:
            g.matched_entity_count += 1
            g.matched_entities.append(exp["name"])
            g.entity_kind_total += 1
            if extracted_by_norm[norm].kind == exp.get("kind", ""):
                g.entity_kind_matches += 1
        else:
            g.missed_entities.append(exp["name"])

    for ent in result.entities:
        if _normalize(ent.name) not in expected_names:
            g.extra_entity_count += 1
            g.extra_entities.append(f"{ent.name} ({ent.kind})")

    # --- Relation matching (by normalized from/to/type) ---
    extracted_rels: set[tuple[str, str, str]] = set()
    for rel in result.relations:
        extracted_rels.add(
            (_normalize(rel.from_entity), _normalize(rel.to_entity), _normalize(rel.relation_type))
        )

    g.expected_relation_count = len(case.expected_relations)
    for exp in case.expected_relations:
        key = (_normalize(exp["from"]), _normalize(exp["to"]), _normalize(exp["type"]))
        if key in extracted_rels:
            g.matched_relation_count += 1
            g.matched_relations.append(f"{exp['from']} --{exp['type']}--> {exp['to']}")
        else:
            g.missed_relations.append(f"{exp['from']} --{exp['type']}--> {exp['to']}")

    g.extra_relation_count = max(0, len(result.relations) - g.matched_relation_count)
    return g


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*\n(.*?)\n\s*```", re.DOTALL)


def _extract_json_from_messages(messages: list[Any]) -> dict[str, Any] | None:
    for msg in reversed(messages):
        if not isinstance(msg, AssistantMessage):
            continue
        for block in reversed(msg.content):
            if not isinstance(block, TextContent):
                continue
            text = block.text.strip()
            for candidate in _json_candidates(text):
                try:
                    obj = json.loads(candidate)
                    if isinstance(obj, dict):
                        return obj
                except json.JSONDecodeError:
                    continue
    return None


def _json_candidates(text: str) -> list[str]:
    """Yield candidate JSON strings from text, most-likely-first."""
    candidates = [text]
    m = _JSON_BLOCK_RE.search(text)
    if m:
        candidates.append(m.group(1))
    start = text.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : i + 1])
                    break
    return candidates


async def run_case(
    case: EvalCase,
    provider: tuple[str, dict[str, Any]],
) -> GradeResult:
    scenario = extractor_scenario()
    config = AgentSessionConfig(
        cwd="/tmp",
        provider=provider,
        scenario=scenario,
        purpose=f"eval_{case.id}",
        loop_config=LoopConfig(max_turns=1),
        log_trace_command=False,
    )
    truncated_steps = _truncate_steps(case.steps)
    prompt = json.dumps(truncated_steps, ensure_ascii=False, indent=2)

    t0 = time.monotonic()
    try:
        session = await AgentSession.create(config)
        try:
            messages = await session.prompt(prompt)
        finally:
            with contextlib.suppress(Exception):
                await session.shutdown()

        obj = _extract_json_from_messages(messages)
        if obj is None:
            g = GradeResult(case_id=case.id, error="no parseable JSON in response")
            g.latency_s = time.monotonic() - t0
            return g

        result = ReportEntitiesParams.model_validate(obj)
        g = grade(case, result)
        g.latency_s = time.monotonic() - t0
        return g

    except Exception as exc:
        g = GradeResult(case_id=case.id, error=str(exc))
        g.latency_s = time.monotonic() - t0
        return g


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def log_report(grades: list[GradeResult]) -> None:
    logger.info("=" * 60)
    logger.info("TRAJECTORY INDEX EXTRACTION EVAL")
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
            f"calls={g.tool_calls} retries={g.retries} {g.latency_s:.1f}s"
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

    avg_recall = sum(g.entity_recall for g in valid) / len(valid)
    avg_precision = sum(g.entity_precision for g in valid) / len(valid)
    avg_kind_acc = sum(g.kind_accuracy for g in valid) / len(valid)
    avg_rel_recall = sum(g.relation_recall for g in valid) / len(valid)
    avg_latency = sum(g.latency_s for g in valid) / len(valid)
    total_retries = sum(g.retries for g in valid)
    errors = sum(1 for g in grades if g.error)

    logger.info("=" * 60)
    logger.info(
        f"AGGREGATE ({len(valid)}/{len(grades)} passed) "
        f"ent_recall={avg_recall:.0%} ent_prec={avg_precision:.0%} "
        f"kind_acc={avg_kind_acc:.0%} rel_recall={avg_rel_recall:.0%} "
        f"avg_lat={avg_latency:.1f}s retries={total_retries} errors={errors}"
    )
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def async_main(model: str, cases_dir: Path, concurrency: int) -> list[GradeResult]:
    provider = _resolve_provider(model)
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
