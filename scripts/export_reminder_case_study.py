#!/usr/bin/env python3
"""Export reminder-injection fork case-study tables.

The script intentionally uses the public `agentm trace` commands as its data
source instead of reading OTLP JSONL or ClickHouse tables directly.

Example:

    uv run python scripts/export_reminder_case_study.py \
      --baseline-session aeb6719f76e5496c8114cf49d70d46c8 \
      --out-prefix runs/rescue-window/case-study-1188-reminder
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import typer


app = typer.Typer(
    add_completion=False,
    pretty_exceptions_show_locals=False,
)


@dataclass(frozen=True)
class Variant:
    name: str
    session_id: str


@dataclass(frozen=True)
class JudgeResult:
    correct: bool | None
    detail: dict[str, Any]
    error: str | None = None


@dataclass(frozen=True)
class CaseStudyConfig:
    baseline_session: str
    case_id: str | None = None
    variant: list[str] | None = None
    data_dir: str | None = None
    include_baseline_row: bool = False
    insert_turn_index: int | None = None
    insert_position: str = ""
    reminder: list[str] | None = None
    reminder_file: list[str] | None = None
    hypothesis: list[str] | None = None
    out_prefix: Path = Path("runs/rescue-window/case-study")
    agentm_cmd: str = "uv run agentm"


def _parse_key_value(raw: str, *, option: str) -> tuple[str, str]:
    if "=" not in raw:
        raise typer.BadParameter(f"{option} must be NAME=VALUE")
    key, value = raw.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key or not value:
        raise typer.BadParameter(f"{option} must be NAME=VALUE")
    return key, value


def _parse_variants(raw_values: list[str]) -> list[Variant]:
    variants: list[Variant] = []
    seen: set[str] = set()
    for raw in raw_values:
        name, session_id = _parse_key_value(raw, option="--variant")
        if name in seen:
            raise typer.BadParameter(f"duplicate variant name: {name}")
        seen.add(name)
        variants.append(Variant(name=name, session_id=session_id))
    return variants


def _parse_mapping(raw_values: list[str], *, option: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw in raw_values:
        key, value = _parse_key_value(raw, option=option)
        parsed[key] = value
    return parsed


def _load_reminder_files(raw_values: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw in raw_values:
        key, path = _parse_key_value(raw, option="--reminder-file")
        parsed[key] = Path(path).read_text(encoding="utf-8").strip()
    return parsed


def _run_trace(
    agentm_cmd: list[str],
    command: str,
    session_id: str,
    *,
    tool: str | None = None,
) -> list[dict[str, Any]]:
    cmd = [
        *agentm_cmd,
        "trace",
        command,
        "--session",
        session_id,
        "--format",
        "ndjson",
    ]
    if tool is not None:
        cmd.extend(["--tool", tool])
    proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
    return [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]


def _run_trace_index(
    agentm_cmd: list[str],
    *,
    children_of: str,
) -> list[dict[str, Any]]:
    cmd = [
        *agentm_cmd,
        "trace",
        "index",
        "--children-of",
        children_of,
        "--format",
        "ndjson",
    ]
    proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
    return [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]


def _session_info(agentm_cmd: list[str], session_id: str) -> dict[str, Any]:
    rows = _run_trace(agentm_cmd, "info", session_id)
    return rows[0] if rows else {}


def _config_from_info(info: dict[str, Any]) -> dict[str, Any]:
    config = info.get("header", {}).get("config")
    return config if isinstance(config, dict) else {}


def _lineage_from_info(info: dict[str, Any]) -> dict[str, Any]:
    lineage = _config_from_info(info).get("lineage")
    return lineage if isinstance(lineage, dict) else {}


def _experiment_from_info(info: dict[str, Any]) -> dict[str, Any]:
    experiment = _config_from_info(info).get("experiment")
    return experiment if isinstance(experiment, dict) else {}


def _data_dir_from_info(info: dict[str, Any]) -> str | None:
    env = _config_from_info(info).get("env")
    if isinstance(env, dict):
        value = env.get("AGENTM_RCA_DATA_DIR")
        if isinstance(value, str) and value:
            return value
    return None


def _variant_name_from_info(info: dict[str, Any], session_id: str) -> str:
    experiment = _experiment_from_info(info)
    for key in ("variant", "reminder_id", "id"):
        value = experiment.get(key)
        if isinstance(value, str) and value:
            return value
    return f"fork-{session_id[:8]}"


def _discover_variants(agentm_cmd: list[str], baseline_session: str) -> list[Variant]:
    variants: list[Variant] = []
    seen: set[str] = set()
    for row in _run_trace_index(agentm_cmd, children_of=baseline_session):
        session_id = row.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            continue
        info = _session_info(agentm_cmd, session_id)
        lineage = _lineage_from_info(info)
        if lineage.get("kind") != "fork":
            continue
        if lineage.get("source_session_id") not in {None, baseline_session}:
            continue
        name = _variant_name_from_info(info, session_id)
        if name in seen:
            name = f"{name}-{session_id[:8]}"
        seen.add(name)
        variants.append(Variant(name=name, session_id=session_id))
    return variants


def _message_text(message: dict[str, Any]) -> str:
    payload = message.get("payload") or {}
    content = payload.get("content") or []
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        text = block.get("text")
        if isinstance(text, str):
            parts.append(text)
        elif block.get("type") == "tool_call":
            parts.append(f"tool:{block.get('name', '')}")
    return " ".join(parts)


def _final_payload(tool_rows: list[dict[str, Any]], *, session_id: str) -> dict[str, Any]:
    if not tool_rows:
        raise RuntimeError(f"session {session_id} has no submit_final_report calls")
    row = next(
        (
            candidate
            for candidate in reversed(tool_rows)
            if (candidate.get("result") or {}).get("is_error") is False
        ),
        tool_rows[-1],
    )
    args = row.get("args")
    if isinstance(args, dict):
        return args
    result = row.get("result") or {}
    content = result.get("content") if isinstance(result, dict) else None
    if isinstance(content, list) and content:
        first = content[0]
        text = first.get("text") if isinstance(first, dict) else None
        if isinstance(text, str):
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
    raise RuntimeError(f"session {session_id} final report payload is not readable")


def _root_summary(payload: dict[str, Any]) -> str:
    roots = payload.get("root_causes") or []
    labels: list[str] = []
    for root in roots:
        if isinstance(root, dict):
            labels.append(f"{root.get('service')}:{root.get('fault_kind')}")
        elif isinstance(root, str):
            node = _fpg_node_by_id(payload).get(root)
            if node is not None:
                labels.append(
                    f"{node.get('subject', '')}|{node.get('predicate', '')}"
                )
            else:
                labels.append(root)
    return "; ".join(labels)


def _fpg_node_by_id(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    nodes = payload.get("nodes")
    if not isinstance(nodes, list):
        return {}
    by_id: dict[str, dict[str, Any]] = {}
    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_id = node.get("id")
        if isinstance(node_id, str):
            by_id[node_id] = node
    return by_id


def _looks_like_fpg_payload(payload: dict[str, Any]) -> bool:
    roots = payload.get("root_causes")
    nodes = payload.get("nodes")
    return isinstance(nodes, list) and isinstance(roots, list) and all(
        isinstance(root, str) for root in roots
    )


def _is_empty_fpg_payload(payload: dict[str, Any]) -> bool:
    return (
        payload.get("nodes") == []
        and payload.get("edges") == []
        and payload.get("root_causes") == []
    )


def _verified_graph_is_empty(data_dir: str) -> bool:
    graph_path = Path(data_dir) / "causal_graph_verified.json"
    try:
        raw = json.loads(graph_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    graph = raw.get("graph")
    if not isinstance(graph, dict):
        return False
    return graph.get("nodes") == [] and graph.get("edges") == []


def _metric_f1(detail: dict[str, Any], key: str) -> float | None:
    metric = detail.get(key)
    if not isinstance(metric, dict):
        return None
    value = metric.get("f1")
    return float(value) if value is not None else None


def _metric_field(detail: dict[str, Any], key: str, field: str) -> float | None:
    metric = detail.get(key)
    if not isinstance(metric, dict):
        return None
    value = metric.get(field)
    return float(value) if value is not None else None


def _fpg_graph_dimensions(detail: dict[str, Any]) -> dict[str, Any]:
    return {
        "fpg_score": detail.get("score"),
        "fpg_root_subject_f1": _metric_f1(detail, "root_subjects"),
        "fpg_subject_f1": _metric_f1(detail, "subjects"),
        "fpg_soft_subject_edge_f1": _metric_f1(detail, "soft_subject_edges"),
        "fpg_root_subject_precision": _metric_field(
            detail, "root_subjects", "precision"
        ),
        "fpg_root_subject_recall": _metric_field(detail, "root_subjects", "recall"),
        "fpg_subject_path_match_hit": detail.get("subject_path_match_hit"),
        "fpg_subject_path_reachability_hit": detail.get(
            "subject_path_reachability_hit"
        ),
        "fpg_missing_root_nodes": "; ".join(
            str(item) for item in detail.get("missing_root_nodes") or []
        ),
        "fpg_extra_root_nodes": "; ".join(
            str(item) for item in detail.get("extra_root_nodes") or []
        ),
        "fpg_extra_subjects": "; ".join(
            str(item) for item in detail.get("extra_subjects") or []
        ),
        "fpg_extra_subject_edges": "; ".join(
            str(item) for item in detail.get("extra_subject_edges") or []
        ),
    }


def _fpg_sql_dimensions(detail: dict[str, Any]) -> dict[str, Any]:
    empty = {
        "fpg_sql_evidence_count": None,
        "fpg_sql_executable_count": None,
        "fpg_sql_failed_count": None,
        "fpg_sql_executable_ratio": None,
        "fpg_sql_all_executable": None,
        "fpg_sql_failures": "",
    }
    sql = detail.get("sql_evidence")
    if not isinstance(sql, dict):
        return empty
    return {
        "fpg_sql_evidence_count": sql.get("total"),
        "fpg_sql_executable_count": sql.get("executable"),
        "fpg_sql_failed_count": sql.get("failed"),
        "fpg_sql_executable_ratio": sql.get("ratio"),
        "fpg_sql_all_executable": sql.get("all_executable"),
        "fpg_sql_failures": "; ".join(
            str(item.get("error") or item)
            for item in (sql.get("failures") or [])[:3]
            if isinstance(item, dict)
        ),
    }


def _prefixed(prefix: str, values: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def _find_original_final_insert_after(
    messages: list[dict[str, Any]],
) -> tuple[str | None, str | None]:
    for idx, message in enumerate(messages):
        payload = message.get("payload") or {}
        if payload.get("role") != "assistant":
            continue
        content = payload.get("content") or []
        if any(
            isinstance(block, dict)
            and block.get("type") == "tool_call"
            and block.get("name") == "submit_final_report"
            for block in content
        ):
            if idx == 0:
                return None, None
            prev = messages[idx - 1]
            return prev.get("id"), (prev.get("payload") or {}).get("role")
    return None, None


def _extract_reminder_message(
    messages: list[dict[str, Any]], reminder_text: str | None
) -> tuple[str | None, str]:
    user_messages = [
        message
        for message in messages
        if (message.get("payload") or {}).get("role") == "user"
    ]
    if reminder_text:
        for message in reversed(user_messages):
            if reminder_text in _message_text(message):
                return message.get("id"), reminder_text
        return None, reminder_text
    for message in reversed(user_messages):
        text = _message_text(message).strip()
        if "Reminder for this fork" in text:
            return message.get("id"), text
    if user_messages:
        message = user_messages[-1]
        return message.get("id"), _message_text(message).strip()
    return None, ""


async def _judge_payload(
    payload: dict[str, Any],
    *,
    data_dir: str | None,
    case_id: str,
) -> JudgeResult:
    if not data_dir:
        return JudgeResult(correct=None, detail={})
    if _looks_like_fpg_payload(payload):
        if _is_empty_fpg_payload(payload) and _verified_graph_is_empty(data_dir):
            perfect_empty_detail = {
                "score": 1.0,
                "root_subjects": {
                    "precision": 1.0,
                    "recall": 1.0,
                    "f1": 1.0,
                    "matched": 0.0,
                },
                "subjects": {
                    "precision": 1.0,
                    "recall": 1.0,
                    "f1": 1.0,
                    "matched": 0.0,
                },
                "soft_subject_edges": {
                    "precision": 1.0,
                    "recall": 1.0,
                    "f1": 1.0,
                    "matched": 0.0,
                },
                "subject_path_match_hit": True,
                "subject_path_reachability_hit": True,
                "missing_root_nodes": [],
                "extra_root_nodes": [],
                "extra_subjects": [],
                "extra_subject_edges": [],
                "f1": 1.0,
                "precision": 1.0,
                "recall": 1.0,
                "any_service_hit": False,
            }
            return JudgeResult(correct=True, detail=perfect_empty_detail)
        try:
            from fpg import ModelRCAOutput, Scenario, compare_model_to_ground_truth

            model_output = ModelRCAOutput.model_validate(payload)
            graph_path = Path(data_dir) / "causal_graph_verified.json"
            if not graph_path.is_file():
                graph_path = Path(data_dir) / "causal_graph.json"
            scenario = Scenario.model_validate_json(graph_path.read_text(encoding="utf-8"))
            comparison = compare_model_to_ground_truth(model_output, scenario).model_dump(mode="json")
            sql_eval = None
            if comparison is None:
                return JudgeResult(
                    correct=None,
                    detail={},
                    error="fpg graph comparison unavailable",
                )
            detail = dict(comparison)
            if sql_eval is not None:
                detail["sql_evidence"] = sql_eval
            correct = float(detail.get("score") or 0.0) >= 1.0
            detail.update(
                {
                    "f1": detail.get("score"),
                    "precision": _metric_field(
                        detail, "root_subjects", "precision"
                    ),
                    "recall": _metric_field(detail, "root_subjects", "recall"),
                    "any_service_hit": (
                        (_metric_field(detail, "root_subjects", "matched") or 0.0)
                        > 0.0
                    ),
                }
            )
            return JudgeResult(correct=correct, detail=detail)
        except Exception as exc:  # pragma: no cover - diagnostic exporter path
            return JudgeResult(correct=None, detail={}, error=f"fpg judge failed: {exc}")
    from agentm_eval.benchmarks.rca.rescue_window_judge import RcabenchJudge

    outcome = await RcabenchJudge().judge(
        agent_output_json=json.dumps(payload, ensure_ascii=False),
        data_dir=data_dir,
        case_id=case_id,
    )
    return JudgeResult(
        correct=outcome.correct,
        detail=outcome.detail,
        error=outcome.error,
    )


def _num(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _effect(baseline: JudgeResult, fork: JudgeResult) -> str:
    if baseline.correct is False and fork.correct is True:
        return "helped"
    if baseline.correct is True and fork.correct is False:
        return "harmed"

    base_f1 = _num(baseline.detail.get("f1"))
    fork_f1 = _num(fork.detail.get("f1"))
    if fork_f1 > base_f1:
        return "improved_partial"
    if fork_f1 < base_f1:
        return "harmed_partial"

    base_service = bool(baseline.detail.get("any_service_hit"))
    fork_service = bool(fork.detail.get("any_service_hit"))
    if base_service and not fork_service:
        return "harmed_partial"
    if fork_service and not base_service:
        return "improved_partial"

    base_fault = _num(baseline.detail.get("fault_kind_accuracy"), -1.0)
    fork_fault = _num(fork.detail.get("fault_kind_accuracy"), -1.0)
    if fork_fault > base_fault:
        return "improved_partial"
    if fork_fault < base_fault:
        return "harmed_partial"
    return "unchanged"


def _markdown_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


async def _build_rows(config: CaseStudyConfig) -> list[dict[str, Any]]:
    agentm_cmd = shlex.split(config.agentm_cmd)
    variants = (
        _parse_variants(config.variant)
        if config.variant
        else _discover_variants(agentm_cmd, config.baseline_session)
    )
    if not variants:
        raise RuntimeError(
            "no fork variants found; pass --variant NAME=SESSION_ID or ensure "
            "fork sessions have lineage.source_session_id metadata"
        )
    reminder_overrides = _parse_mapping(config.reminder or [], option="--reminder")
    reminder_overrides.update(_load_reminder_files(config.reminder_file or []))
    hypotheses = _parse_mapping(config.hypothesis or [], option="--hypothesis")

    baseline_info = _session_info(agentm_cmd, config.baseline_session)
    data_dir = config.data_dir or _data_dir_from_info(baseline_info)
    case_id = config.case_id or (
        Path(data_dir).name if data_dir else config.baseline_session
    )
    baseline_messages = _run_trace(agentm_cmd, "messages", config.baseline_session)
    baseline_usage = _run_trace(agentm_cmd, "usage", config.baseline_session)[0]
    baseline_final = _final_payload(
        _run_trace(
            agentm_cmd,
            "tools",
            config.baseline_session,
            tool="submit_final_report",
        ),
        session_id=config.baseline_session,
    )
    baseline_judge = await _judge_payload(
        baseline_final,
        data_dir=data_dir,
        case_id=f"{case_id}-baseline",
    )
    baseline_fpg_detail = {
        **_fpg_graph_dimensions(baseline_judge.detail),
        **_fpg_sql_dimensions(baseline_judge.detail),
    }
    insert_after_id, insert_after_role = _find_original_final_insert_after(
        baseline_messages
    )

    rows: list[dict[str, Any]] = []
    if config.include_baseline_row:
        rows.append(
            {
                "case_id": case_id,
                "variant": "baseline",
                "baseline_session_id": config.baseline_session,
                "fork_session_id": config.baseline_session,
                "source": "agentm trace",
                "insert_turn_index": "",
                "insert_position": "baseline",
                "insert_after_source_message_id": "",
                "insert_after_source_message_role": "",
                "fork_reminder_message_id": "",
                "reminder_text_hash": "",
                "reminder_text": "",
                "hypothesis": "baseline without reminder",
                "baseline_output": _root_summary(baseline_final),
                "fork_output": _root_summary(baseline_final),
                "baseline_exact_match": baseline_judge.correct,
                "fork_exact_match": baseline_judge.correct,
                "baseline_f1": baseline_judge.detail.get("f1"),
                "fork_f1": baseline_judge.detail.get("f1"),
                "baseline_precision": baseline_judge.detail.get("precision"),
                "fork_precision": baseline_judge.detail.get("precision"),
                "baseline_recall": baseline_judge.detail.get("recall"),
                "fork_recall": baseline_judge.detail.get("recall"),
                "baseline_any_service_hit": baseline_judge.detail.get(
                    "any_service_hit"
                ),
                "fork_any_service_hit": baseline_judge.detail.get("any_service_hit"),
                "baseline_fault_kind_accuracy": baseline_judge.detail.get(
                    "fault_kind_accuracy"
                ),
                "fork_fault_kind_accuracy": baseline_judge.detail.get(
                    "fault_kind_accuracy"
                ),
                **_prefixed("baseline", baseline_fpg_detail),
                **_prefixed("fork", baseline_fpg_detail),
                "baseline_judge_error": baseline_judge.error,
                "fork_judge_error": baseline_judge.error,
                "effect": "baseline",
                "change_summary": _root_summary(baseline_final),
                "baseline_turns": baseline_usage.get("turns"),
                "fork_delta_turns": 0,
                "fork_total_tokens": baseline_usage.get("total_tokens"),
                "fork_lineage_kind": "root",
                "fork_lineage_source_session_id": "",
            }
        )
    for variant in variants:
        fork_messages = _run_trace(agentm_cmd, "messages", variant.session_id)
        fork_usage = _run_trace(agentm_cmd, "usage", variant.session_id)[0]
        fork_info = _session_info(agentm_cmd, variant.session_id)
        fork_final = _final_payload(
            _run_trace(
                agentm_cmd,
                "tools",
                variant.session_id,
                tool="submit_final_report",
            ),
            session_id=variant.session_id,
        )
        fork_judge = await _judge_payload(
            fork_final,
            data_dir=data_dir,
            case_id=f"{case_id}-{variant.name}",
        )
        fork_fpg_detail = {
            **_fpg_graph_dimensions(fork_judge.detail),
            **_fpg_sql_dimensions(fork_judge.detail),
        }
        reminder_message_id, reminder_text = _extract_reminder_message(
            fork_messages,
            reminder_overrides.get(variant.name),
        )
        lineage = (
            fork_info.get("header", {})
            .get("config", {})
            .get("lineage", {})
        )
        fork_point = lineage.get("fork_point") if isinstance(lineage, dict) else {}
        fork_turn_index = (
            fork_point.get("turn_index") if isinstance(fork_point, dict) else None
        )
        insert_turn_index = (
            config.insert_turn_index
            if config.insert_turn_index is not None
            else fork_turn_index
        )
        rows.append(
            {
                "case_id": case_id,
                "variant": variant.name,
                "baseline_session_id": config.baseline_session,
                "fork_session_id": variant.session_id,
                "source": "agentm trace",
                "insert_turn_index": insert_turn_index,
                "insert_position": config.insert_position
                or (
                    f"after source turn {insert_turn_index}"
                    if insert_turn_index is not None
                    else ""
                ),
                "insert_after_source_message_id": insert_after_id,
                "insert_after_source_message_role": insert_after_role,
                "fork_reminder_message_id": reminder_message_id,
                "reminder_text_hash": (
                    hashlib.sha256(reminder_text.encode()).hexdigest()
                    if reminder_text
                    else ""
                ),
                "reminder_text": reminder_text,
                "hypothesis": hypotheses.get(variant.name, ""),
                "baseline_output": _root_summary(baseline_final),
                "fork_output": _root_summary(fork_final),
                "baseline_exact_match": baseline_judge.correct,
                "fork_exact_match": fork_judge.correct,
                "baseline_f1": baseline_judge.detail.get("f1"),
                "fork_f1": fork_judge.detail.get("f1"),
                "baseline_precision": baseline_judge.detail.get("precision"),
                "fork_precision": fork_judge.detail.get("precision"),
                "baseline_recall": baseline_judge.detail.get("recall"),
                "fork_recall": fork_judge.detail.get("recall"),
                "baseline_any_service_hit": baseline_judge.detail.get(
                    "any_service_hit"
                ),
                "fork_any_service_hit": fork_judge.detail.get("any_service_hit"),
                "baseline_fault_kind_accuracy": baseline_judge.detail.get(
                    "fault_kind_accuracy"
                ),
                "fork_fault_kind_accuracy": fork_judge.detail.get(
                    "fault_kind_accuracy"
                ),
                **_prefixed("baseline", baseline_fpg_detail),
                **_prefixed("fork", fork_fpg_detail),
                "baseline_judge_error": baseline_judge.error,
                "fork_judge_error": fork_judge.error,
                "effect": _effect(baseline_judge, fork_judge),
                "change_summary": (
                    f"{_root_summary(baseline_final)} -> {_root_summary(fork_final)}"
                ),
                "baseline_turns": baseline_usage.get("turns"),
                "fork_delta_turns": fork_usage.get("turns"),
                "fork_total_tokens": fork_usage.get("total_tokens"),
                "fork_lineage_kind": lineage.get("kind"),
                "fork_lineage_source_session_id": lineage.get("source_session_id"),
            }
        )
    return rows


def _write_outputs(rows: list[dict[str, Any]], out_prefix: Path) -> None:
    if not rows:
        raise RuntimeError("no rows to write")
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out_prefix.with_suffix(".csv")
    md_path = out_prefix.with_suffix(".md")

    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_cols = [
        "case_id",
        "variant",
        "insert_turn_index",
        "baseline_output",
        "fork_output",
        "baseline_exact_match",
        "fork_exact_match",
        "baseline_fpg_score",
        "fork_fpg_score",
        "baseline_fpg_root_subject_f1",
        "fork_fpg_root_subject_f1",
        "fork_fpg_subject_path_reachability_hit",
        "fork_fpg_sql_executable_count",
        "fork_fpg_sql_evidence_count",
        "effect",
        "fork_session_id",
    ]
    with md_path.open("w", encoding="utf-8") as fh:
        fh.write("| " + " | ".join(md_cols) + " |\n")
        fh.write("| " + " | ".join(["---"] * len(md_cols)) + " |\n")
        for row in rows:
            fh.write(
                "| "
                + " | ".join(_markdown_escape(row.get(col, "")) for col in md_cols)
                + " |\n"
            )
    print(json.dumps({"csv": str(csv_path), "md": str(md_path)}, indent=2))


@app.command()
def main(
    baseline_session: Annotated[
        str, typer.Option("--baseline-session", help="Baseline session id.")
    ],
    case_id: Annotated[
        str | None,
        typer.Option(
            "--case-id",
            help="Case identifier. Defaults to basename of inferred --data-dir.",
        ),
    ] = None,
    variant: Annotated[
        list[str] | None,
        typer.Option(
            "--variant",
            help=(
                "Fork variant as NAME=SESSION_ID. Repeatable. When omitted, "
                "fork children are discovered from baseline lineage metadata."
            ),
        ),
    ] = None,
    data_dir: Annotated[
        str | None,
        typer.Option("--data-dir", help="RCA case directory for judge."),
    ] = None,
    include_baseline_row: Annotated[
        bool,
        typer.Option(
            "--include-baseline-row",
            help="Include a standalone baseline row before fork variants.",
        ),
    ] = False,
    insert_turn_index: Annotated[
        int | None,
        typer.Option("--insert-turn-index", help="Source turn index."),
    ] = None,
    insert_position: Annotated[
        str,
        typer.Option("--insert-position", help="Human-readable insertion point."),
    ] = "",
    reminder: Annotated[
        list[str] | None,
        typer.Option(
            "--reminder",
            help="Override reminder text as NAME=TEXT. Repeatable.",
        ),
    ] = None,
    reminder_file: Annotated[
        list[str] | None,
        typer.Option(
            "--reminder-file",
            help="Read reminder text from file as NAME=PATH. Repeatable.",
        ),
    ] = None,
    hypothesis: Annotated[
        list[str] | None,
        typer.Option(
            "--hypothesis",
            help="Optional hypothesis label as NAME=TEXT. Repeatable.",
        ),
    ] = None,
    out_prefix: Annotated[
        Path,
        typer.Option(
            "--out-prefix",
            help="Output path without extension. Writes .csv and .md.",
        ),
    ] = Path("runs/rescue-window/case-study"),
    agentm_cmd: Annotated[
        str,
        typer.Option("--agentm-cmd", help="Command prefix used before `trace`."),
    ] = "uv run agentm",
) -> None:
    config = CaseStudyConfig(
        baseline_session=baseline_session,
        case_id=case_id,
        variant=variant,
        data_dir=data_dir,
        include_baseline_row=include_baseline_row,
        insert_turn_index=insert_turn_index,
        insert_position=insert_position,
        reminder=reminder,
        reminder_file=reminder_file,
        hypothesis=hypothesis,
        out_prefix=out_prefix,
        agentm_cmd=agentm_cmd,
    )
    rows = asyncio.run(_build_rows(config))
    _write_outputs(rows, config.out_prefix)


if __name__ == "__main__":
    app()
