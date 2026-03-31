"""Trajectory judging runner — single-pass classification.

Orchestrates batch execution of the trajectory_judger scenario.
Includes case collection, config loading, and skeleton extraction.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pydantic
import yaml
from pydantic import BaseModel, ConfigDict
from rich.console import Console
from rich.table import Table

from agentm.builder import AgentSystemContext, build_system_context, create_agent_run
from agentm.config.schema import ScenarioConfig, SkeletonConfig, SystemConfig
from agentm.exceptions import AgentMError
from agentm.scenarios.trajectory_judger.data import TrajectoryLabel
from agentm.server.app import (
    DashboardOpts,
    start_dashboard_server,
    wire_trajectory_to_ws,
)
from agentm.tools.trajectory_reader import get_reader

logger = logging.getLogger(__name__)
console = Console()


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
    fault_type: str = ""
    fault_category: str = ""
    difficulty: dict[str, Any] | None = None


def parse_ground_truth(correct_answer: str) -> list[str]:
    """Parse comma-separated ground truth services into a list."""
    return [s.strip() for s in correct_answer.split(",") if s.strip()]


# ---------------------------------------------------------------------------
# Config
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


class JudgeConfig(BaseModel):
    """Config for the judge command (loaded from batch YAML).

    Extra fields from legacy configs are silently ignored so the same
    YAML files remain compatible.
    """

    model_config = ConfigDict(extra="ignore")

    source: SourceConfig = SourceConfig()
    scenario: str = "config/scenarios/trajectory_analysis"
    system_config: str = "config/system.yaml"
    concurrency: int = 1


def load_judge_config(path: str | Path) -> JudgeConfig:
    """Load a judge/batch config YAML file with env var substitution."""
    from agentm.config.loader import substitute_env_vars

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    resolved = substitute_env_vars(raw)
    return JudgeConfig(**resolved)


# ---------------------------------------------------------------------------
# Shared helpers for case collection
# ---------------------------------------------------------------------------

SourcePathFn = Callable[[str], str]


def _build_source_path_fn(src: SourceConfig) -> SourcePathFn | None:
    """Build a source path resolver from config."""
    if not src.data_base_dir:
        return None
    root = src.data_base_dir
    pat = src.source_path_pattern or "{data_base_dir}/{source}"

    def _resolve(source: str) -> str:
        return pat.format(data_base_dir=root, source=source)

    return _resolve


def _resolve_data_dir(source: str, fn: SourcePathFn | None) -> str:
    if not fn or not source:
        return ""
    candidate = Path(fn(source))
    return str(candidate) if candidate.is_dir() else ""


def _matches_filter(
    correct: bool | None,
    row_exp_id: str,
    row_agent_type: str,
    src: SourceConfig,
) -> bool:
    """Check if a record matches source filter criteria."""
    if src.filter == "incorrect" and correct is not False:
        return False
    if src.filter == "correct" and correct is not True:
        return False
    if src.exp_id is not None and row_exp_id != src.exp_id:
        return False
    if src.agent_type is not None and row_agent_type != src.agent_type:
        return False
    return True


def _get_difficulty_field(meta: dict[str, Any] | None, field: str) -> str:
    """Extract a field from meta.difficulty (or meta directly if it has the field)."""
    if not isinstance(meta, dict):
        return ""
    difficulty = meta.get("difficulty")
    if isinstance(difficulty, dict):
        return str(difficulty.get(field, "")) if difficulty.get(field) else ""
    # meta itself might be the difficulty dict (from directory path)
    return str(meta.get(field, "")) if meta.get(field) else ""


def _extract_fault_context(meta: dict[str, Any] | None) -> str:
    """Extract fault injection context from evaluation metadata."""
    if not meta:
        return ""
    parts: list[str] = []
    difficulty = meta.get("difficulty")
    if isinstance(difficulty, dict):
        if ft := difficulty.get("fault_type"):
            parts.append(f"Fault type: {ft}")
        if fc := difficulty.get("fault_category"):
            parts.append(f"Fault category: {fc}")
    if datapack := meta.get("datapack_name"):
        parts.append(f"Datapack: {datapack}")
    return "; ".join(parts)


def _enrich_reasoning(base_reasoning: str, fault_ctx: str) -> str:
    """Append fault context to reasoning if not already present."""
    if fault_ctx and "Fault type:" not in base_reasoning:
        return f"{base_reasoning} | {fault_ctx}" if base_reasoning else fault_ctx
    return base_reasoning


# ---------------------------------------------------------------------------
# Collect from directory
# ---------------------------------------------------------------------------


def _collect_from_directory(src: SourceConfig) -> list[CaseInfo]:
    """Scan a directory for exported eval JSON files and load metadata."""
    base = Path(src.directory)
    if not base.is_dir():
        raise NotADirectoryError(f"Not a directory: {src.directory}")

    resolve_fn = _build_source_path_fn(src)
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
            continue

        correct = meta.get("correct")
        if not _matches_filter(correct, meta.get("exp_id", ""), meta.get("agent_type", "") or "", src):
            continue

        source = meta.get("source", "") or ""
        fault_ctx = _extract_fault_context({"difficulty": meta.get("difficulty")} if meta.get("difficulty") else None)
        reasoning = _enrich_reasoning(meta.get("reasoning", "") or "", fault_ctx)

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
                data_dir=_resolve_data_dir(source, resolve_fn),
                fault_type=_get_difficulty_field(meta.get("difficulty"), "fault_type"),
                fault_category=_get_difficulty_field(meta.get("difficulty"), "fault_category"),
                difficulty=meta.get("difficulty") if isinstance(meta.get("difficulty"), dict) else None,
            )
        )

    if src.limit:
        cases = cases[:src.limit]
    return cases


# ---------------------------------------------------------------------------
# Collect from database
# ---------------------------------------------------------------------------


def _parse_trajectory_data(raw_trajectories: str | dict | None) -> list[dict]:
    """Parse trajectory data from DB into a list of trajectory dicts."""
    if raw_trajectories is None:
        return []
    data = json.loads(raw_trajectories) if isinstance(raw_trajectories, str) else raw_trajectories
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if "trajectories" in data and isinstance(data["trajectories"], list):
            return data["trajectories"]
        return [data]
    return []


def _build_db_query(src: SourceConfig) -> tuple[str, list[Any]]:
    conditions = ["e.stage = 'judged'", "e.trajectories IS NOT NULL"]
    params: list[Any] = []

    if src.exp_id:
        conditions.append("e.exp_id = %s")
        params.append(src.exp_id)
    if src.filter == "incorrect":
        conditions.append("e.correct = %s")
        params.append(False)
    elif src.filter == "correct":
        conditions.append("e.correct = %s")
        params.append(True)
    if src.agent_type:
        conditions.append("e.agent_type = %s")
        params.append(src.agent_type)

    query = (
        "SELECT e.id, e.exp_id, e.dataset_index, e.correct, e.correct_answer,"
        "       e.extracted_final_answer, e.agent_type, e.model_name, e.reasoning,"
        "       e.source, e.trajectories,"
        "       d.meta AS data_meta "
        "FROM evaluation_data e "
        "LEFT JOIN data d ON e.dataset = d.dataset AND e.source = d.source "
        f"WHERE {' AND '.join(conditions)} "
        "ORDER BY e.id"
    )
    if src.limit:
        query += " LIMIT %s"
        params.append(src.limit)
    return query, params


def _collect_from_db(src: SourceConfig, output_dir: str | None = None) -> list[CaseInfo]:
    """Query the eval DB and export cases to JSON files."""
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
        raise ValueError("LLM_EVAL_DB_URL environment variable is required for database source")

    query, params = _build_db_query(src)

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

    out_dir = Path(output_dir) if output_dir else Path("eval-trajectories")
    out_dir.mkdir(parents=True, exist_ok=True)

    resolve_fn = _build_source_path_fn(src)

    cases: list[CaseInfo] = []
    for row in rows:
        row = dict(row)
        row_id = row["id"]
        row_exp_id = row["exp_id"] or "unknown"
        label = "correct" if row["correct"] else "incorrect"

        source = row.get("source", "") or ""

        data_meta = row.get("data_meta")
        if isinstance(data_meta, str):
            try:
                data_meta = json.loads(data_meta)
            except json.JSONDecodeError:
                data_meta = None

        base_reasoning = row["reasoning"] or ""
        fault_ctx = _extract_fault_context(data_meta)
        reasoning = _enrich_reasoning(base_reasoning, fault_ctx)

        difficulty = None
        if isinstance(data_meta, dict) and data_meta.get("difficulty"):
            difficulty = data_meta["difficulty"]

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
        if difficulty:
            eval_meta["difficulty"] = difficulty

        trajectories = _parse_trajectory_data(row["trajectories"])
        output: dict[str, Any] = {
            "_eval_meta": eval_meta,
            "trajectories": trajectories,
        }

        filepath = out_dir / f"{row_exp_id}_{row_id}_{label}.json"
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
                data_dir=_resolve_data_dir(source, resolve_fn),
                fault_type=_get_difficulty_field(data_meta, "fault_type"),
                fault_category=_get_difficulty_field(data_meta, "fault_category"),
                difficulty=difficulty,
            )
        )

    console.print(f"Exported [green]{len(cases)}[/] cases to [cyan]{out_dir}[/]")
    return cases


def collect_cases(cfg: JudgeConfig) -> list[CaseInfo]:
    """Collect cases according to the source config."""
    if cfg.source.type == "database":
        return _collect_from_db(cfg.source)
    return _collect_from_directory(cfg.source)


# ---------------------------------------------------------------------------
# Skeleton extraction
# ---------------------------------------------------------------------------


def _truncate(text: str, max_len: int, *, oneline: bool = False) -> str:
    if oneline:
        text = text.replace("\n", " ").replace("\r", "")
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def _should_include_tool(tool_name: str, config: SkeletonConfig) -> bool:
    if config.include_tools:
        return tool_name in config.include_tools
    if config.exclude_tools:
        return tool_name not in config.exclude_tools
    return True


def _extract_skeleton_from_json(
    data: dict, config: SkeletonConfig,
) -> list[dict]:
    """Extract skeleton from DB-exported JSON (OpenAI message format)."""
    trajectories = data.get("trajectories")
    if isinstance(trajectories, list):
        all_messages = [msg for traj in trajectories for msg in traj.get("messages", [])]
    elif isinstance(data.get("messages"), list):
        all_messages = data["messages"]
    else:
        return []

    response_map: dict[str, str] = {}
    for msg in all_messages:
        if isinstance(msg, dict) and msg.get("role") == "tool":
            tc_id = msg.get("tool_call_id", "")
            if tc_id:
                response_map[tc_id] = str(msg.get("content", ""))

    steps: list[dict] = []
    step_num = 0
    for msg in all_messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls", []):
            func = tc.get("function", {})
            tool_name = func.get("name", "")
            if not _should_include_tool(tool_name, config):
                continue

            step_num += 1
            raw_args = func.get("arguments", "")
            if isinstance(raw_args, str):
                try:
                    args_obj = json.loads(raw_args)
                    args_str = json.dumps(args_obj, ensure_ascii=False)
                except json.JSONDecodeError:
                    args_str = raw_args
            else:
                args_str = json.dumps(raw_args, ensure_ascii=False)

            tc_id = tc.get("id", "")
            response_content = response_map.get(tc_id, "")

            steps.append({
                "step": step_num,
                "tool": tool_name,
                "args": _truncate(args_str, config.max_args_length),
                "response_preview": _truncate(response_content, config.response_preview_length, oneline=True),
                "response_chars": len(response_content),
            })
    return steps


def _extract_skeleton_from_jsonl(
    path: Path, config: SkeletonConfig,
) -> list[dict]:
    """Extract skeleton from JSONL trajectory (TrajectoryCollector events)."""
    tool_calls: list[dict] = []
    tool_results: dict[int, dict] = {}

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            et = event.get("event_type", "")
            if et == "tool_call":
                tool_calls.append(event)
            elif et == "tool_result":
                tool_results[event.get("seq", -1)] = event.get("data", {})

    steps: list[dict] = []
    for i, tc_event in enumerate(tool_calls):
        data = tc_event.get("data", {})
        tool_name = data.get("tool_name", "")
        if not _should_include_tool(tool_name, config):
            continue

        args = data.get("args", {})
        args_str = json.dumps(args, ensure_ascii=False) if isinstance(args, dict) else str(args)

        tc_seq = tc_event.get("seq", -1)
        result_data = tool_results.get(tc_seq + 1, {})
        response_content = str(result_data.get("content", ""))

        steps.append({
            "step": i + 1,
            "tool": tool_name,
            "args": _truncate(args_str, config.max_args_length),
            "response_preview": _truncate(response_content, config.response_preview_length, oneline=True),
            "response_chars": len(response_content),
        })
    return steps


def extract_skeleton(file_path: Path, config: SkeletonConfig) -> list[dict]:
    """Extract tool-call skeleton from a trajectory file."""
    if not file_path.exists():
        return []
    if file_path.suffix == ".json":
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
        return _extract_skeleton_from_json(data, config)
    return _extract_skeleton_from_jsonl(file_path, config)


def _load_injection_context(data_dir: str) -> str:
    """Load injection.json from data_dir and format key fault details."""
    if not data_dir:
        return ""
    injection_path = Path(data_dir) / "injection.json"
    if not injection_path.exists():
        return ""
    try:
        injection = json.loads(injection_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return ""

    parts: list[str] = []

    display_config_raw = injection.get("display_config")
    if display_config_raw:
        try:
            dc = json.loads(display_config_raw) if isinstance(display_config_raw, str) else display_config_raw
            ip = dc.get("injection_point", {})
            if ip:
                target_parts = []
                if ip.get("app_name"):
                    target_parts.append(ip["app_name"])
                if ip.get("class_name"):
                    target_parts.append(ip["class_name"])
                if ip.get("method_name"):
                    target_parts.append(ip["method_name"])
                if target_parts:
                    parts.append(f"- **Injection Target**: {' / '.join(target_parts)}")
            if duration := dc.get("duration"):
                parts.append(f"- **Duration**: {duration} min")
            mem_type = dc.get("mem_type")
            if mem_type is not None:
                label = "Heap" if mem_type == 1 else "Stack" if mem_type == 2 else str(mem_type)
                parts.append(f"- **Memory Type**: {label}")
        except (json.JSONDecodeError, TypeError):
            pass

    gt = injection.get("ground_truth", {})
    if gt:
        if gt.get("service"):
            parts.append(f"- **GT Service**: {', '.join(gt['service'])}")
        if gt.get("function"):
            parts.append(f"- **GT Function**: {', '.join(gt['function'])}")
        if gt.get("metric"):
            parts.append(f"- **GT Metric**: {', '.join(gt['metric'])}")

    start = injection.get("start_time")
    end = injection.get("end_time")
    if start and end:
        parts.append(f"- **Time Window**: {start} ~ {end}")

    return "\n".join(parts)


def format_skeleton(steps: list[dict]) -> str:
    """Format skeleton steps as compact text for LLM consumption."""
    if not steps:
        return "(no tool calls found)"
    lines = []
    for s in steps:
        preview = f" → {s['response_preview']}" if s["response_preview"] else ""
        chars = f" ({s['response_chars']} chars)" if s["response_chars"] else ""
        lines.append(f"Step {s['step']}: {s['tool']}({s['args']}){preview}{chars}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Single-case judging
# ---------------------------------------------------------------------------

_CATEGORY_COLORS: dict[str, str] = {
    "success": "green",
    "lucky_hit": "yellow",
    "exploration_fail": "red",
    "confirmation_fail": "red",
    "judgment_fail": "red",
}
_CATEGORIES = ["success", "lucky_hit", "exploration_fail", "confirmation_fail", "judgment_fail"]


def _parse_label(output: object, case_id: str, ground_truth: list[str]) -> TrajectoryLabel | None:
    if output is None:
        return None

    if isinstance(output, dict):
        if "raw_text" in output and len(output) == 1:
            logger.warning("Case %s: structured output failed, got raw_text fallback", case_id)
            return None
        output.setdefault("trajectory_id", case_id)
        output.setdefault("case_id", case_id)
        output.setdefault("ground_truth", ground_truth)
        try:
            return TrajectoryLabel(**output)
        except pydantic.ValidationError as exc:
            logger.warning("Case %s: TrajectoryLabel validation failed: %s", case_id, exc)
            return None

    if isinstance(output, TrajectoryLabel):
        return output

    logger.warning("Case %s: unexpected output type %s", case_id, type(output).__name__)
    return None


async def _judge_single_case(
    case: CaseInfo,
    ctx: AgentSystemContext,
    broadcaster: object | None = None,
    eval_tracker: object | None = None,
) -> TrajectoryLabel | None:
    """Run trajectory_judger on a single case via AgentSystem.execute()."""
    from agentm.server.app import Broadcaster as _Broadcaster

    case_id = case.case_id
    file_path = case.file_path
    ground_truth = parse_ground_truth(case.correct_answer)

    if file_path.exists():
        get_reader().register(str(file_path))

    system = create_agent_run(ctx)

    if eval_tracker is not None and system.trajectory is not None:
        eval_tracker.update_trajectory_path(case_id, str(system.trajectory.file_path))  # type: ignore[attr-defined]

    if broadcaster is not None and system.trajectory is not None and isinstance(broadcaster, _Broadcaster):
        wire_trajectory_to_ws(system.trajectory, broadcaster, case_id)

    skeleton_config = ctx.scenario_config.orchestrator.skeleton or SkeletonConfig()
    skeleton_steps = extract_skeleton(file_path, skeleton_config)
    skeleton_text = format_skeleton(skeleton_steps)

    gt_str = ", ".join(ground_truth)
    correct_label = {True: "CORRECT", False: "INCORRECT", None: "UNKNOWN"}[case.correct]
    parts = [
        f"## Case {case_id}",
        "",
        f"- **Trajectory ID**: {case_id}",
        f"- **Ground Truth**: {gt_str}",
        f"- **Agent's Final Answer**: {case.extracted_final_answer or '(none)'}",
        f"- **Eval Result**: {correct_label}",
        f"- **jq_query thread_id**: `{case_id}`",
    ]

    logger.info(
        "Case %s: data_dir=%r, reasoning=%r",
        case_id, case.data_dir or "(empty)", (case.reasoning or "")[:80],
    )
    injection_ctx = _load_injection_context(case.data_dir)
    if injection_ctx:
        logger.info("Case %s: loaded injection context from %s", case_id, case.data_dir)
    elif case.data_dir:
        logger.warning("Case %s: no injection.json found at %s", case_id, case.data_dir)
    if injection_ctx or case.reasoning:
        parts.extend(["", "## Fault Context", ""])
        if injection_ctx:
            parts.append(injection_ctx)
        if case.reasoning:
            if injection_ctx:
                parts.append("")
            parts.append(f"**Eval Reasoning**: {case.reasoning}")

    if skeleton_steps:
        parts.extend([
            "",
            "## Trajectory Skeleton",
            "",
            skeleton_text,
        ])

    task_content = "\n".join(parts)

    try:
        async with system:
            result = await system.execute({"task_description": task_content})
        return _parse_label(result.get("output"), case_id, ground_truth)
    except (AgentMError, asyncio.TimeoutError) as e:
        logger.error("Error judging case %s: %s", case_id, e)
        return None


# ---------------------------------------------------------------------------
# Summary and output
# ---------------------------------------------------------------------------


def _print_summary(results: list[TrajectoryLabel], total: int, failed_count: int) -> None:
    console.print()
    console.rule("Judgment Complete")
    console.print(f"Total cases: [cyan]{total}[/]")
    console.print(f"Successful: [green]{len(results)}[/]")
    if failed_count:
        console.print(f"Failed: [red]{failed_count}[/]")

    if not results:
        return

    table = Table(title="Classification Results")
    table.add_column("Category", style="cyan")
    table.add_column("Count", style="green", justify="right")
    table.add_column("Percentage", style="yellow", justify="right")

    category_counts: dict[str, int] = {}
    for r in results:
        category_counts[r.category] = category_counts.get(r.category, 0) + 1

    for category in _CATEGORIES:
        count = category_counts.get(category, 0)
        if count > 0:
            pct = count / len(results) * 100
            table.add_row(category, str(count), f"{pct:.1f}%")

    console.print()
    console.print(table)


def _save_results(results: list[TrajectoryLabel], total: int, failed_count: int, output_path: str) -> None:
    output_data = {
        "total": total,
        "successful": len(results),
        "failed": failed_count,
        "results": [r.model_dump(mode="json") for r in results],
    }
    Path(output_path).write_text(
        json.dumps(output_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    console.print(f"\nResults saved to: [green]{output_path}[/]")


# ---------------------------------------------------------------------------
# JSONL cache for incremental resume
# ---------------------------------------------------------------------------


def _cache_path_for(output_path: str) -> Path:
    """Derive the JSONL cache path from the output path."""
    p = Path(output_path)
    return p.parent / (p.stem + ".cache.jsonl")


def _load_cached_results(cache_file: Path) -> dict[str, TrajectoryLabel]:
    """Load previously successful results from the JSONL cache.

    Returns a dict mapping case_id -> TrajectoryLabel.
    Malformed lines are skipped with a warning.
    """
    cached: dict[str, TrajectoryLabel] = {}
    if not cache_file.exists():
        return cached
    skipped = 0
    with open(cache_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                label = TrajectoryLabel.model_validate(data)
                cached[label.case_id] = label
            except (json.JSONDecodeError, pydantic.ValidationError):
                skipped += 1
                continue
    if skipped:
        logger.warning("Skipped %d malformed lines in cache %s", skipped, cache_file)
    return cached


def _append_to_cache(cache_file: Path, label: TrajectoryLabel) -> None:
    """Append a single result to the JSONL cache file.

    Uses synchronous I/O — acceptable because each write is tiny
    relative to the LLM API latency that dominates per-case cost.
    """
    with open(cache_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(label.model_dump(mode="json"), ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def run_judging(
    cases: list[CaseInfo],
    system_config: SystemConfig,
    scenario_config: ScenarioConfig,
    output_path: str | None = None,
    dashboard_opts: DashboardOpts | None = None,
    concurrency: int = 1,
) -> list[TrajectoryLabel]:
    """Run trajectory_judger scenario on each case.

    Args:
        concurrency: Maximum number of cases to judge in parallel.
            Defaults to 1 (sequential).
    """
    if dashboard_opts is None:
        dashboard_opts = DashboardOpts()

    if not cases:
        console.print("[yellow]No cases to judge.[/]")
        return []

    # --- Cache: load previous results and filter cases ---
    cache_file: Path | None = None
    cached_results: dict[str, TrajectoryLabel] = {}
    if output_path:
        cache_file = _cache_path_for(output_path)
        cached_results = _load_cached_results(cache_file)
        if cached_results:
            original_count = len(cases)
            cases = [c for c in cases if c.case_id not in cached_results]
            skipped = original_count - len(cases)
            console.print(
                f"Loaded [green]{len(cached_results)}[/] cached results from {cache_file.name}, "
                f"skipping [cyan]{skipped}[/] already-judged cases"
            )
            if not cases:
                console.print("[green]All cases already judged. Nothing to do.[/]")
                all_results = list(cached_results.values())
                _print_summary(all_results, total=original_count, failed_count=0)
                return all_results

    if not dashboard_opts.enabled:
        system_config.debug.trajectory.enabled = False

    console.print(f"Judging {len(cases)} cases with model: {scenario_config.orchestrator.model}")
    if concurrency > 1:
        console.print(f"Concurrency: [cyan]{concurrency}[/]")
    console.print()

    # Build immutable context once for all cases
    ctx = build_system_context(
        "trajectory_judger",
        scenario_config,
        system_config,
    )

    tracker, broadcaster, dashboard_task = None, None, None
    if dashboard_opts.enabled:
        tracker, broadcaster, dashboard_task = await start_dashboard_server(
            scenario_config=scenario_config,
            host=dashboard_opts.host,
            port=dashboard_opts.port,
            thread_id=f"judge-{cases[0].case_id if cases else 'batch'}",
        )
        for i, case in enumerate(cases):
            tracker.register_sample(
                case.case_id,
                dataset_index=i,
                data_dir=str(case.file_path.parent),
            )

    results: list[TrajectoryLabel] = []
    failed_cases: list[str] = []

    def _record_result(
        label: TrajectoryLabel | None,
        case: CaseInfo,
        *,
        prefix: str = "",
    ) -> None:
        """Record a judgment result — append to results/cache, print, update tracker."""
        tag = f"[{case.case_id}] " if prefix else ""
        if label is not None:
            results.append(label)
            if cache_file is not None:
                _append_to_cache(cache_file, label)
            color = _CATEGORY_COLORS.get(label.category, "white")
            console.print(f"    {tag}[{color}]{label.category}[/] — {label.reasoning[:80]}...")
            if tracker is not None:
                tracker.mark_completed(case.case_id)
        else:
            failed_cases.append(case.case_id)
            console.print(f"    {tag}[red]FAILED[/]")
            if tracker is not None:
                tracker.mark_failed(case.case_id, "classification failed")

    if concurrency <= 1:
        # Sequential execution
        for i, case in enumerate(cases, 1):
            console.print(f"[{i}/{len(cases)}] Judging case {case.case_id}...")

            if tracker is not None:
                tracker.mark_running(case.case_id, run_id=case.case_id)

            label = await _judge_single_case(
                case=case,
                ctx=ctx,
                broadcaster=broadcaster,
                eval_tracker=tracker,
            )
            _record_result(label, case)
    else:
        # Concurrent execution with semaphore
        sem = asyncio.Semaphore(concurrency)
        lock = asyncio.Lock()

        async def _run_case(idx: int, case: CaseInfo) -> None:
            async with sem:
                console.print(f"[{idx}/{len(cases)}] Judging case {case.case_id}...")

                if tracker is not None:
                    tracker.mark_running(case.case_id, run_id=case.case_id)

                label = await _judge_single_case(
                    case=case,
                    ctx=ctx,
                    broadcaster=broadcaster,
                    eval_tracker=tracker,
                )

                async with lock:
                    _record_result(label, case, prefix=case.case_id)

        tasks = [_run_case(i, case) for i, case in enumerate(cases, 1)]
        await asyncio.gather(*tasks)

    # Merge cached + new results for summary and output
    all_results = list(cached_results.values()) + results
    total_cases = len(all_results) + len(failed_cases)

    _print_summary(all_results, total=total_cases, failed_count=len(failed_cases))

    if output_path and all_results:
        _save_results(all_results, total=total_cases, failed_count=len(failed_cases), output_path=output_path)
        if cache_file is not None and not failed_cases:
            cache_file.unlink(missing_ok=True)
            logger.info("Cleaned up cache file %s (all cases completed)", cache_file)

    if dashboard_task is not None:
        console.print(
            f"\nDashboard running at [link=http://localhost:{dashboard_opts.port}]"
            f"http://localhost:{dashboard_opts.port}[/link] — press Ctrl+C to stop."
        )
        try:
            await dashboard_task
        except asyncio.CancelledError:
            pass

    return all_results
