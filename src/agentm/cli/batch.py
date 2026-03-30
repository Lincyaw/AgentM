"""Batch trajectory analysis — group cases and run analysis at scale.

Supports two data sources:
1. Directory of exported eval JSON files (with _eval_meta)
2. Direct database query (requires LLM_EVAL_DB_URL and psycopg2)

Each batch groups N trajectories into a single analysis run so the agent
can identify cross-case patterns, not just per-case errors.

Configuration is driven by a YAML file (see ``config/batch/``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table

from agentm.cli.run import run_trajectory_analysis
from agentm.exceptions import AgentMError

logger = logging.getLogger(__name__)
console = Console()


# ---------------------------------------------------------------------------
# Batch configuration schema
# ---------------------------------------------------------------------------


class SourceConfig(BaseModel):
    """Where to get trajectories from."""

    type: Literal["directory", "database"] = "directory"
    directory: str = "./eval-trajectories"
    filter: Literal["incorrect", "correct", "all"] = "incorrect"
    exp_id: str | None = None
    agent_type: str | None = None
    limit: int | None = None
    data_base_dir: str | None = None
    source_path_pattern: str | None = None
    """Pattern for resolving source → data directory.

    Placeholders: ``{data_base_dir}``, ``{source}``.
    Default (when only *data_base_dir* is set): ``{data_base_dir}/{source}``
    """


class BatchExecutionConfig(BaseModel):
    """How to group and run batches."""

    size: int = Field(default=10, description="Trajectories per analysis run")
    concurrency: int = Field(default=1, description="Parallel batch runs")
    max_steps: int = Field(default=60, description="Max orchestrator steps per batch")


class TaskConfig(BaseModel):
    """Analysis task prompt configuration.

    If ``goals`` is provided, they replace the default analysis goals in
    the auto-generated prompt. If ``custom`` is provided, it replaces the
    entire auto-generated prompt (goals are ignored).
    """

    goals: list[str] | None = None
    custom: str | None = None


class OutputBatchConfig(BaseModel):
    """Output and export settings."""

    export_dir: str = Field(
        default="./eval-trajectories",
        description="Directory for DB-exported JSON files",
    )
    verbose: bool = False
    dashboard: bool = False
    dashboard_port: int = 8765
    dashboard_host: str = "0.0.0.0"


class BatchConfig(BaseModel):
    """Top-level batch analysis configuration (batch YAML file)."""

    source: SourceConfig = SourceConfig()
    batch: BatchExecutionConfig = BatchExecutionConfig()
    task: TaskConfig = TaskConfig()
    output: OutputBatchConfig = OutputBatchConfig()
    scenario: str = "config/scenarios/trajectory_analysis"
    system_config: str = "config/system.yaml"


def load_batch_config(path: str | Path) -> BatchConfig:
    """Load a batch config YAML file with env var substitution."""
    from agentm.config.loader import substitute_env_vars

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    resolved = substitute_env_vars(raw)
    return BatchConfig(**resolved)


# ---------------------------------------------------------------------------
# Case metadata
# ---------------------------------------------------------------------------


@dataclass
class CaseInfo:
    """Metadata for one evaluation case."""

    file_path: Path
    case_id: str
    exp_id: str = ""
    correct: bool | None = None
    correct_answer: str = ""
    extracted_final_answer: str = ""
    reasoning: str = ""
    agent_type: str = ""
    model_name: str = ""
    dataset_index: int | None = None
    source: str = ""
    data_dir: str = ""


# ---------------------------------------------------------------------------
# Collect from directory
# ---------------------------------------------------------------------------


def collect_from_directory(
    directory: str,
    filter_correctness: str = "incorrect",
    exp_id: str | None = None,
    agent_type: str | None = None,
    limit: int | None = None,
    data_base_dir: str | None = None,
    source_path_pattern: str | None = None,
    source_path_fn: SourcePathFn | None = None,
) -> list[CaseInfo]:
    """Scan a directory for exported eval JSON files and load metadata.

    Args:
        directory: Path to directory containing ``_eval_meta`` JSON files.
        filter_correctness: ``"incorrect"``, ``"correct"``, or ``"all"``.
        exp_id: Optional experiment ID filter.
        agent_type: Optional agent type filter.
        limit: Maximum number of cases to return.
        data_base_dir: Root directory for dataset sources.
        source_path_pattern: Pattern for resolving source → data dir.
        source_path_fn: Explicit resolver (takes priority over pattern/base_dir).
    """
    base = Path(directory)
    if not base.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    resolve_fn = source_path_fn or build_source_path_fn(data_base_dir, source_path_pattern)

    meta_filter = _EvalMetaFilter(
        filter_correctness=filter_correctness,
        exp_id=exp_id,
        agent_type=agent_type,
    )

    cases: list[CaseInfo] = []
    for path in sorted(base.glob("*.json")):
        if path.name.endswith(".export.json"):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.warning("Skipping unreadable file: %s", path)
            continue

        meta = data.get("_eval_meta")
        if not isinstance(meta, dict):
            logger.debug("Skipping non-eval file: %s", path)
            continue

        correct = meta.get("correct")
        if not meta_filter.matches(correct, meta.get("exp_id", ""), meta.get("agent_type", "") or ""):
            continue

        source = meta.get("source", "") or ""
        resolved_data_dir = _resolve_data_dir(source, resolve_fn)

        # Enrich reasoning with fault context from difficulty if present
        base_reasoning = meta.get("reasoning", "") or ""
        difficulty = meta.get("difficulty")
        if isinstance(difficulty, dict):
            fault_ctx = _extract_fault_context_from_meta({"difficulty": difficulty})
            if fault_ctx and "Fault type:" not in base_reasoning:
                reasoning = f"{base_reasoning} | {fault_ctx}" if base_reasoning else fault_ctx
            else:
                reasoning = base_reasoning
        else:
            reasoning = base_reasoning

        cases.append(
            CaseInfo(
                file_path=path,
                case_id=str(meta.get("id", path.stem)),
                exp_id=meta.get("exp_id", ""),
                correct=correct,
                correct_answer=meta.get("correct_answer", "") or "",
                extracted_final_answer=meta.get("extracted_final_answer", "") or "",
                reasoning=reasoning,
                agent_type=meta.get("agent_type", "") or "",
                model_name=meta.get("model_name", "") or "",
                dataset_index=meta.get("dataset_index"),
                source=source,
                data_dir=resolved_data_dir,
            )
        )

    if limit:
        cases = cases[:limit]
    return cases


# ---------------------------------------------------------------------------
# Collect from database
# ---------------------------------------------------------------------------


def collect_from_db(
    exp_id: str | None = None,
    filter_correctness: str = "incorrect",
    agent_type: str | None = None,
    limit: int | None = None,
    output_dir: str | None = None,
    data_base_dir: str | None = None,
    source_path_pattern: str | None = None,
    source_path_fn: SourcePathFn | None = None,
) -> list[CaseInfo]:
    """Query the eval DB and export cases to JSON files.

    Requires ``LLM_EVAL_DB_URL`` environment variable and ``psycopg2``.

    Returns CaseInfo list with ``file_path`` pointing to exported JSON files
    in the same ``_eval_meta`` format as ``export_eval_trajectories.py``.
    """
    try:
        import psycopg2
        import psycopg2.extras
    except ImportError:
        raise ImportError(
            "psycopg2 is required for database source. "
            "Install with: uv add psycopg2-binary"
        ) from None

    db_url = os.environ.get("LLM_EVAL_DB_URL")
    if not db_url:
        raise ValueError(
            "LLM_EVAL_DB_URL environment variable is required for database source"
        )

    query, params = _build_db_query(exp_id, filter_correctness, agent_type, limit)

    console.print("Connecting to eval database...")
    conn = psycopg2.connect(db_url)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        console.print("[yellow]No matching rows found in database.[/]")
        return []

    # Export to files
    out_dir = Path(output_dir) if output_dir else Path("eval-trajectories")
    out_dir.mkdir(parents=True, exist_ok=True)

    resolve_fn = source_path_fn or build_source_path_fn(data_base_dir, source_path_pattern)

    cases: list[CaseInfo] = []
    for row in rows:
        row = dict(row)
        row_id = row["id"]
        row_exp_id = row["exp_id"] or "unknown"
        label = "correct" if row["correct"] else "incorrect"
        filename = f"{row_exp_id}_{row_id}_{label}.json"

        source = row.get("source", "") or ""
        resolved_data_dir = _resolve_data_dir(source, resolve_fn)

        # Extract fault context from DB meta
        row_meta = row.get("meta")
        if isinstance(row_meta, str):
            try:
                row_meta = json.loads(row_meta)
            except json.JSONDecodeError:
                row_meta = None

        base_reasoning = row["reasoning"] or ""
        fault_ctx = _extract_fault_context_from_meta(row_meta)
        if fault_ctx and "Fault type:" not in base_reasoning:
            reasoning = f"{base_reasoning} | {fault_ctx}" if base_reasoning else fault_ctx
        else:
            reasoning = base_reasoning

        # Build eval_meta with enriched reasoning and difficulty
        eval_meta: dict[str, Any] = {
            "id": row_id,
            "exp_id": row_exp_id,
            "dataset_index": row["dataset_index"],
            "correct": row["correct"],
            "correct_answer": row["correct_answer"],
            "extracted_final_answer": row["extracted_final_answer"],
            "agent_type": row["agent_type"],
            "model_name": row["model_name"],
            "reasoning": reasoning,
            "source": source,
        }
        if isinstance(row_meta, dict) and row_meta.get("difficulty"):
            eval_meta["difficulty"] = row_meta["difficulty"]

        trajectories = _parse_trajectory_data(row["trajectories"])

        output: dict[str, Any] = {
            "_eval_meta": eval_meta,
            "trajectories": trajectories,
        }

        filepath = out_dir / filename
        filepath.write_text(
            json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        cases.append(
            CaseInfo(
                file_path=filepath,
                case_id=str(row_id),
                exp_id=row_exp_id,
                correct=row["correct"],
                correct_answer=row["correct_answer"] or "",
                extracted_final_answer=row["extracted_final_answer"] or "",
                reasoning=reasoning,
                agent_type=row["agent_type"] or "",
                model_name=row["model_name"] or "",
                dataset_index=row["dataset_index"],
                source=source,
                data_dir=resolved_data_dir,
            )
        )

    console.print(f"Exported [green]{len(cases)}[/] cases to [cyan]{out_dir}[/]")
    return cases


# ---------------------------------------------------------------------------
# Collect for judging — returns trajectory data and ground truth
# ---------------------------------------------------------------------------


class _EvalMetaFilter:
    """Encapsulates filtering logic for eval metadata.

    Reusable across directory and database sources to ensure consistent
    filtering behavior.
    """

    def __init__(
        self,
        filter_correctness: str = "all",
        exp_id: str | None = None,
        agent_type: str | None = None,
    ):
        self.filter_correctness = filter_correctness
        self.exp_id = exp_id
        self.agent_type = agent_type

    def matches(self, correct: bool | None, row_exp_id: str, row_agent_type: str) -> bool:
        """Check if a record matches all filter criteria."""
        if self.filter_correctness == "incorrect" and correct is not False:
            return False
        if self.filter_correctness == "correct" and correct is not True:
            return False
        if self.exp_id is not None and row_exp_id != self.exp_id:
            return False
        if self.agent_type is not None and row_agent_type != self.agent_type:
            return False
        return True


def parse_ground_truth(correct_answer: str) -> list[str]:
    """Parse comma-separated ground truth services into a list."""
    return [s.strip() for s in correct_answer.split(",") if s.strip()]


def _extract_fault_context_from_meta(meta: dict[str, Any] | None) -> str:
    """Extract fault injection context from evaluation_data.meta.

    meta.difficulty contains fault_type and fault_category.
    meta.datapack_name encodes the injection target.
    Falls back gracefully when fields are missing.
    """
    if not meta:
        return ""
    parts: list[str] = []

    difficulty = meta.get("difficulty")
    if isinstance(difficulty, dict):
        ft = difficulty.get("fault_type")
        fc = difficulty.get("fault_category")
        if ft:
            parts.append(f"Fault type: {ft}")
        if fc:
            parts.append(f"Fault category: {fc}")

    datapack = meta.get("datapack_name")
    if datapack:
        parts.append(f"Datapack: {datapack}")

    return "; ".join(parts)


SourcePathFn = Callable[[str], str]
"""Callable that maps a source name to a data directory path."""


def build_source_path_fn(
    data_base_dir: str | None,
    pattern: str | None = None,
) -> SourcePathFn | None:
    """Build a source path resolver from base dir and optional pattern.

    *pattern* uses ``{data_base_dir}`` and ``{source}`` placeholders,
    e.g. ``"{data_base_dir}/{source}/converted"``.

    When *pattern* is ``None``, falls back to ``{data_base_dir}/{source}``.

    Returns ``None`` when *data_base_dir* is not set.
    """
    if not data_base_dir:
        return None
    _root = data_base_dir
    _pat = pattern or "{data_base_dir}/{source}"

    def _resolve(source: str) -> str:
        return _pat.format(data_base_dir=_root, source=source)

    return _resolve


def _resolve_data_dir(source: str, source_path_fn: SourcePathFn | None) -> str:
    """Resolve data directory using the source path function."""
    if not source_path_fn or not source:
        return ""
    candidate = Path(source_path_fn(source))
    return str(candidate) if candidate.is_dir() else ""


def _parse_trajectory_data(raw_trajectories: str | dict | None) -> list[dict]:
    """Parse trajectory data from DB into a list of trajectory dicts.

    Always returns a list so the exported file has a consistent
    ``{"_eval_meta": ..., "trajectories": [...]}`` structure.

    Handles:
    - ``{"trajectories": [t1, t2, ...]}`` — extracts the list
    - ``[t1, t2, ...]`` — returns as-is
    - ``{"messages": [...], ...}`` — wraps as single-element list
    """
    if raw_trajectories is None:
        return []

    if isinstance(raw_trajectories, str):
        data = json.loads(raw_trajectories)
    else:
        data = raw_trajectories

    # Already a list of trajectories
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        # Nested wrapper: {"trajectories": [...]}
        if "trajectories" in data and isinstance(data["trajectories"], list):
            return data["trajectories"]
        # Single trajectory dict (has "messages" or other trajectory fields)
        return [data]

    return []


def _build_db_query(
    exp_id: str | None,
    filter_correctness: str,
    agent_type: str | None,
    limit: int | None,
) -> tuple[str, list[Any]]:
    """Build the database query and parameters.

    Returns (query_string, params_list) tuple.
    """
    conditions = ["stage = 'judged'", "trajectories IS NOT NULL"]
    params: list[Any] = []

    if exp_id:
        conditions.append("exp_id = %s")
        params.append(exp_id)
    if filter_correctness == "incorrect":
        conditions.append("correct = %s")
        params.append(False)
    elif filter_correctness == "correct":
        conditions.append("correct = %s")
        params.append(True)
    if agent_type:
        conditions.append("agent_type = %s")
        params.append(agent_type)

    query = (
        "SELECT id, exp_id, dataset_index, correct, correct_answer,"
        "       extracted_final_answer, agent_type, model_name, reasoning,"
        "       source, trajectories, meta "
        "FROM evaluation_data "
        f"WHERE {' AND '.join(conditions)} "
        "ORDER BY id"
    )
    if limit:
        query += " LIMIT %s"
        params.append(limit)

    return query, params


# ---------------------------------------------------------------------------
# Collect dispatcher — reads source config and delegates
# ---------------------------------------------------------------------------


def collect_cases(cfg: BatchConfig) -> list[CaseInfo]:
    """Collect cases according to the source config."""
    src = cfg.source
    if src.type == "database":
        return collect_from_db(
            exp_id=src.exp_id,
            filter_correctness=src.filter,
            agent_type=src.agent_type,
            limit=src.limit,
            output_dir=cfg.output.export_dir,
            data_base_dir=src.data_base_dir,
            source_path_pattern=src.source_path_pattern,
        )
    return collect_from_directory(
        directory=src.directory,
        filter_correctness=src.filter,
        exp_id=src.exp_id,
        agent_type=src.agent_type,
        limit=src.limit,
        data_base_dir=src.data_base_dir,
        source_path_pattern=src.source_path_pattern,
    )


# ---------------------------------------------------------------------------
# Task prompt construction
# ---------------------------------------------------------------------------

_DEFAULT_GOALS = [
    "**Classify** each case using the Outcome Classification taxonomy "
    "(CORRECT_SOUND / CORRECT_FLAWED / WRONG_NEAR_MISS / WRONG_DIVERGED) "
    "with Level 2 mechanisms",
    "**Cross-case patterns**: identify error modes that appear in "
    "multiple cases (e.g., anchoring bias, coverage gaps, premature "
    "termination)",
    "**Agent behavioral features**: tool usage patterns, hypothesis "
    "management style, exploration breadth vs depth, decision timing",
    "**Systematic biases**: failures that suggest a structural weakness "
    "rather than case-specific bad luck",
    "**Actionable recommendations**: what prompt/tool/architecture "
    "changes would fix the most common failure modes",
]


def build_batch_task(
    cases: list[CaseInfo],
    batch_index: int,
    task_config: TaskConfig | None = None,
) -> str:
    """Construct the analysis task prompt for a batch of cases.

    The prompt includes per-case metadata (agent answer, ground truth,
    judge reasoning) so the analysis agent has full context before it
    starts querying trajectories.

    If ``task_config.custom`` is set, it is returned as-is.
    If ``task_config.goals`` is set, they replace the default goals.
    """
    if task_config and task_config.custom:
        return task_config.custom

    n = len(cases)
    exp_ids = {c.exp_id for c in cases if c.exp_id}
    exp_label = ", ".join(sorted(exp_ids)) if exp_ids else "unknown"

    n_incorrect = sum(1 for c in cases if c.correct is False)
    n_correct = sum(1 for c in cases if c.correct is True)

    lines = [
        f"Batch {batch_index + 1}: analysis of {n} RCA trajectories "
        f"(experiment: {exp_label}, "
        f"{n_incorrect} incorrect, {n_correct} correct).",
        "",
        "## Cases",
        "",
    ]

    for i, c in enumerate(cases, 1):
        label = (
            "INCORRECT"
            if c.correct is False
            else "CORRECT"
            if c.correct is True
            else "UNKNOWN"
        )
        source_tag = f" source={c.source}" if c.source else ""
        agent_ans = (c.extracted_final_answer or "N/A")[:150]
        ground_truth = (c.correct_answer or "N/A")[:150]
        reasoning = (c.reasoning or "N/A")[:200]

        lines.append(
            f"{i}. **[{label}]** id={c.case_id}{source_tag}\n"
            f"   Agent answer: {agent_ans}\n"
            f"   Ground truth: {ground_truth}\n"
            f"   Judge: {reasoning}"
        )

    # Goals
    goals = _DEFAULT_GOALS
    if task_config and task_config.goals:
        goals = task_config.goals

    lines.extend(["", "## Analysis Goals", ""])
    for idx, goal in enumerate(goals, 1):
        lines.append(f"{idx}. {goal}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------


def group_into_batches(cases: list[CaseInfo], batch_size: int) -> list[list[CaseInfo]]:
    """Split cases into groups of batch_size."""
    return [cases[i : i + batch_size] for i in range(0, len(cases), batch_size)]


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------


async def run_batch_analysis(cfg: BatchConfig, cases: list[CaseInfo]) -> None:
    """Run batch trajectory analysis across grouped cases.

    Cases are split into batches of ``cfg.batch.size``. Each batch becomes
    one ``run_trajectory_analysis`` invocation, giving the agent a global
    view across multiple trajectories.
    """
    if not cases:
        console.print("[yellow]No cases to analyze.[/]")
        return

    batch_size = cfg.batch.size
    concurrency = cfg.batch.concurrency
    batches = group_into_batches(cases, batch_size)

    # Print plan
    table = Table(
        title=(
            f"Batch Analysis: {len(cases)} cases -> {len(batches)} batches "
            f"(size={batch_size}, concurrency={concurrency})"
        )
    )
    table.add_column("Batch", style="cyan", justify="right")
    table.add_column("Cases", style="green", justify="right")
    table.add_column("Incorrect", style="red", justify="right")
    table.add_column("IDs", style="dim")
    for i, batch in enumerate(batches):
        ids = ", ".join(c.case_id for c in batch[:6])
        if len(batch) > 6:
            ids += f" +{len(batch) - 6}"
        n_bad = sum(1 for c in batch if c.correct is False)
        table.add_row(f"#{i + 1}", str(len(batch)), str(n_bad), ids)
    console.print(table)
    console.print()

    # Track results
    completed = 0
    failed_batches: list[tuple[int, str]] = []

    semaphore = asyncio.Semaphore(concurrency)

    async def _run_one(batch_idx: int, batch: list[CaseInfo]) -> None:
        nonlocal completed
        async with semaphore:
            task = build_batch_task(batch, batch_idx, cfg.task)
            trajectories = [str(c.file_path) for c in batch]
            # Build case_id -> data_dir mapping for observability data access
            case_data_mapping = {c.case_id: c.data_dir for c in batch if c.data_dir}
            try:
                await run_trajectory_analysis(
                    trajectories=trajectories,
                    task=task,
                    scenario_dir=cfg.scenario,
                    config_path=cfg.system_config,
                    verbose=cfg.output.verbose,
                    max_steps=cfg.batch.max_steps,
                    dashboard=cfg.output.dashboard and batch_idx == 0,
                    dashboard_port=cfg.output.dashboard_port,
                    dashboard_host=cfg.output.dashboard_host,
                    case_data_mapping=case_data_mapping or None,
                )
                completed += 1
            except (AgentMError, asyncio.TimeoutError) as e:
                failed_batches.append((batch_idx + 1, str(e)))
                console.print(f"[red]Batch #{batch_idx + 1} failed: {e}[/]")

    if concurrency == 1:
        for i, batch in enumerate(batches):
            await _run_one(i, batch)
    else:
        tasks = [_run_one(i, batch) for i, batch in enumerate(batches)]
        await asyncio.gather(*tasks)

    # Summary
    console.print()
    console.rule("Batch Analysis Complete")
    console.print(f"Total cases: [cyan]{len(cases)}[/]")
    console.print(f"Batches completed: [green]{completed}[/] / {len(batches)}")
    if failed_batches:
        console.print(f"Batches failed: [red]{len(failed_batches)}[/]")
        for idx, err in failed_batches:
            console.print(f"  Batch #{idx}: {err[:200]}")
