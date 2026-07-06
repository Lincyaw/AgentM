"""Workflow-first Aegis injection campaign orchestrator.

The workflow owns round orchestration and deterministic state plumbing. Child
agents own judgment and Aegis tool use. Historical knowledge is represented as
verifier-compatible case records so injection can consume verifier outputs when
those become available.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

from agentm.extensions.builtin.workflow import WorkflowContext

FAMILY: dict[str, str] = {
    "NetworkDelay": "network",
    "NetworkLoss": "network",
    "NetworkPartition": "network",
    "NetworkCorrupt": "network",
    "NetworkBandwidth": "network",
    "NetworkDuplicate": "network",
    "PodFailure": "pod",
    "PodKill": "pod",
    "ContainerKill": "pod",
    "JVMLatency": "jvm",
    "JVMException": "jvm",
    "JVMMySQLLatency": "jvm",
    "JVMMYSQLLatency": "jvm",
    "JVMMySQLException": "jvm",
    "JVMMYSQLException": "jvm",
    "JVMReturn": "jvm",
    "JVMCPUStress": "jvm",
    "HTTPRequestDelay": "http",
    "HTTPResponseDelay": "http",
    "HTTPRequestAbort": "http",
    "HTTPResponseAbort": "http",
    "HTTPReplace": "http",
    "HTTPStatusCode": "http",
    "DNSError": "dns",
    "DNSRandom": "dns",
    "CPUStress": "stress",
    "TimeSkew": "stress",
    "MemoryStress": "stress",
}

FAMILY_CAPS: dict[str, int] = {
    "network": 25,
    "pod": 20,
    "jvm": 25,
    "http": 20,
    "dns": 10,
    "stress": 10,
}

ROUND_RESULT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": True,
    "required": ["round", "system", "status", "round_file", "trace_ids", "case_records"],
    "properties": {
        "round": {"type": "integer"},
        "system": {"type": "string"},
        "status": {"type": "string"},
        "round_file": {"type": "string"},
        "trace_ids": {"type": "array", "items": {"type": "string"}},
        "case_records": {"type": "array", "items": {"type": "object", "additionalProperties": True}},
        "notes": {"type": "string"},
    },
}


def _expand(path: str) -> Path:
    return Path(path).expanduser().resolve()


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _run_jsonl(cmd: list[str], env: dict[str, str]) -> list[dict[str, Any]]:
    proc = subprocess.run(
        cmd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env=env,
    )
    out: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _load_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        return dict(default)
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return dict(default)
    return data if isinstance(data, dict) else dict(default)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def _load_jsonl(path: Path, limit: int = 200) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows[-limit:]


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _recent_rounds(rounds_dir: Path, limit: int = 10) -> list[dict[str, Any]]:
    files = sorted(rounds_dir.glob("round-*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    rounds: list[dict[str, Any]] = []
    for path in files[:limit]:
        item = _load_json(path, {})
        if item:
            item["_path"] = str(path)
            rounds.append(item)
    return rounds


def _aegis_env(project: str) -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("AEGIS_INSECURE_SKIP_VERIFY", "true")
    env.setdefault("AEGIS_NON_INTERACTIVE", "true")
    env.setdefault("AEGIS_PROJECT", project)
    env.setdefault("AEGIS_TIMEOUT", "120")
    return env


def _family_tally(aegisctl_bin: str, project: str, system: str) -> dict[str, Any]:
    env = _aegis_env(project)
    base = [aegisctl_bin, "inject", "list", "--project", project, "--size", "100", "--page", "1", "-o", "ndjson"]
    scope = f"system={system}"
    try:
        rows = _run_jsonl([*base, "--system", system], env)
        meta = next((r.get("_meta") for r in rows if isinstance(r.get("_meta"), dict)), {})
        if int(meta.get("total") or 0) == 0:
            scope = f"project={project} (system-filter fallback)"
            rows = _run_jsonl(base, env)
    except Exception as exc:  # noqa: BLE001 - workflow should continue with degraded context
        return {"available": False, "error": str(exc), "scope": scope, "counts": {}, "percent": {}}

    counts: Counter[str] = Counter()
    for row in rows:
        if "_meta" in row:
            continue
        chaos_type = str(row.get("fault_type") or row.get("chaos_type") or "")
        counts[FAMILY.get(chaos_type, chaos_type or "unknown")] += 1
    total = sum(counts.values())
    percent = {k: (100 * v // total if total else 0) for k, v in counts.items()}
    blocked = [fam for fam, cap in FAMILY_CAPS.items() if percent.get(fam, 0) >= cap]
    return {
        "available": True,
        "scope": scope,
        "total": total,
        "counts": dict(counts),
        "percent": percent,
        "caps": FAMILY_CAPS,
        "blocked_families": blocked,
    }


def _latest_pedestal_contract(aegisctl_bin: str, project: str, system: str) -> dict[str, Any]:
    env = _aegis_env(project)
    cmd = [aegisctl_bin, "container", "versions", system, "-o", "json"]
    try:
        proc = subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        data = json.loads(proc.stdout)
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": str(exc), "pedestal_name": system}
    items = data.get("items") if isinstance(data, dict) else None
    tag = None
    if isinstance(items, list) and items:
        first = items[0]
        if isinstance(first, dict):
            tag = first.get("name") or first.get("tag") or first.get("version")
        elif isinstance(first, str):
            tag = first
    return {
        "available": bool(tag),
        "pedestal_name": system,
        "pedestal_tag": tag,
        "benchmark_name": "clickhouse",
        "benchmark_tag": "1.0.0",
        "policy": "resolve immediately before every submit; do not reuse old round tags",
    }


def _case_context(case_records: list[dict[str, Any]]) -> dict[str, Any]:
    hard: list[dict[str, Any]] = []
    easy: list[dict[str, Any]] = []
    noop: list[dict[str, Any]] = []
    verified_paths: list[dict[str, Any]] = []
    for rec in case_records[-100:]:
        evidence = rec.get("verifier_evidence") if isinstance(rec.get("verifier_evidence"), dict) else {}
        observations = rec.get("observations") if isinstance(rec.get("observations"), dict) else {}
        score = rec.get("puzzle_score") if isinstance(rec.get("puzzle_score"), dict) else {}
        if evidence.get("status") == "verified" and evidence.get("propagation_path"):
            verified_paths.append({
                "case_id": rec.get("case_id"),
                "fault": rec.get("fault"),
                "propagation_path": evidence.get("propagation_path"),
                "decoys": evidence.get("decoy_hypotheses", []),
            })
        if observations.get("injection_landed") is False:
            noop.append(rec)
        elif score.get("estimated_difficulty") == "high":
            hard.append(rec)
        elif score.get("estimated_difficulty") == "low":
            easy.append(rec)
    return {
        "hard_cases": hard[-10:],
        "easy_cases": easy[-10:],
        "noop_patterns": noop[-10:],
        "verified_paths": verified_paths[-20:],
        "note": "Propagation evidence is verifier-compatible. Treat simulated evidence as a weak prior only.",
    }


def _round_number(meta: dict[str, Any], recent: list[dict[str, Any]]) -> int:
    nums = [int(r.get("round") or 0) for r in recent if isinstance(r.get("round"), int)]
    return max([int(meta.get("total_rounds_run") or 0), *nums], default=0) + 1


def _build_campaign_state(raw_args: dict[str, Any]) -> dict[str, Any]:
    system = str(raw_args.get("system") or "ts")
    project = str(raw_args.get("project") or os.environ.get("AEGIS_PROJECT") or "pair_diagnosis")
    state_dir = _expand(str(raw_args.get("state_dir") or f"~/.aegisctl/injection-author/{system}"))
    aegisctl_bin = str(raw_args.get("aegisctl_bin") or "/tmp/aegisctl")
    rounds_dir = state_dir / "rounds"
    rounds_dir.mkdir(parents=True, exist_ok=True)
    meta_path = state_dir / "metadata.json"
    meta = _load_json(meta_path, {
        "system": system,
        "first_started_at": _now_iso(),
        "total_rounds_run": 0,
    })
    meta["system"] = system
    meta["last_started_at"] = _now_iso()
    _write_json(meta_path, meta)

    recent = _recent_rounds(rounds_dir)
    case_library = _load_jsonl(state_dir / "case_library.jsonl")
    return {
        "system": system,
        "project": project,
        "state_dir": str(state_dir),
        "rounds_dir": str(rounds_dir),
        "aegisctl_bin": aegisctl_bin,
        "source_dir": raw_args.get("source_dir"),
        "extra_instruction": raw_args.get("extra_instruction", ""),
        "validation_mode": bool(raw_args.get("validation_mode", False)),
        "max_parallel": int(raw_args.get("max_parallel") or 1),
        "metadata": meta,
        "recent_rounds": recent,
        "case_library": case_library,
        "next_round": _round_number(meta, recent),
        "family_tally": _family_tally(aegisctl_bin, project, system),
        "submit_contract": _latest_pedestal_contract(aegisctl_bin, project, system),
        "case_context": _case_context(case_library),
    }


def _archive_cases(state: dict[str, Any], result: dict[str, Any]) -> None:
    rows = result.get("case_records")
    if not isinstance(rows, list):
        return
    clean = [row for row in rows if isinstance(row, dict)]
    _append_jsonl(Path(state["state_dir"]) / "case_library.jsonl", clean)


def _update_metadata_after_round(state: dict[str, Any], result: dict[str, Any]) -> None:
    meta_path = Path(state["state_dir"]) / "metadata.json"
    meta = _load_json(meta_path, {})
    meta["last_completed_at"] = _now_iso()
    meta["total_rounds_run"] = max(int(meta.get("total_rounds_run") or 0), int(result.get("round") or 0))
    meta["last_workflow_status"] = result.get("status")
    meta["family_tally"] = state.get("family_tally")
    _write_json(meta_path, meta)


async def _run_one_round(ctx: WorkflowContext, state: dict[str, Any], scenario: str, model: str | None) -> dict[str, Any]:
    round_no = int(state["next_round"])
    context = {
        "mode": "single_round_worker",
        "round": round_no,
        "system": state["system"],
        "project": state["project"],
        "state_dir": state["state_dir"],
        "rounds_dir": state["rounds_dir"],
        "aegisctl_bin": state["aegisctl_bin"],
        "source_dir": state.get("source_dir"),
        "validation_mode": state.get("validation_mode"),
        "recent_rounds": state.get("recent_rounds", []),
        "family_tally": state.get("family_tally"),
        "submit_contract": state.get("submit_contract"),
        "case_context": state.get("case_context"),
        "extra_instruction": state.get("extra_instruction", ""),
        "verifier_case_contract": {
            "minimum_fields": ["case_id", "system", "fault", "observations", "verifier_evidence"],
            "evidence_status": ["verified", "simulated", "pending", "failed"],
            "schema_doc": "contrib/scenarios/injection/knowledge/schema.md",
        },
    }
    result = await ctx.agent(
        "Run one Aegis adversarial injection round using the workflow-provided context.",
        scenario=scenario,
        model=model,
        atom_config={"injection_context": context},
        schema=ROUND_RESULT_SCHEMA,
        timeout=3600,
        trace_label=f"injection-round-{round_no}",
    )
    if not isinstance(result, dict):
        return {
            "round": round_no,
            "system": state["system"],
            "status": "agent_result_not_dict",
            "round_file": "",
            "trace_ids": [],
            "case_records": [],
            "notes": str(result),
        }
    return result


async def run(ctx: WorkflowContext) -> dict[str, Any]:
    raw_args = dict(ctx.args)
    rounds = int(raw_args["rounds"]) if "rounds" in raw_args else 1
    sleep_seconds = int(raw_args.get("sleep") or raw_args.get("sleep_seconds") or 0)
    scenario = str(raw_args.get("scenario") or Path(__file__).parent)
    model = raw_args.get("worker_model") or raw_args.get("model")
    model_name = str(model) if model else None
    heartbeat_dir_arg = raw_args.get("heartbeat_dir")

    results: list[dict[str, Any]] = []
    for idx in range(rounds):
        ctx.phase(f"round-{idx + 1}")
        state = _build_campaign_state(raw_args)
        ctx.log(
            f"system={state['system']} next_round={state['next_round']} "
            f"family_tally={state['family_tally'].get('percent', {})}"
        )
        result = await _run_one_round(ctx, state, scenario, model_name)
        results.append(result)
        _archive_cases(state, result)
        _update_metadata_after_round(state, result)
        heartbeat_dir = _expand(str(heartbeat_dir_arg)) if heartbeat_dir_arg else Path(state["state_dir"])
        heartbeat_dir.mkdir(parents=True, exist_ok=True)
        (heartbeat_dir / "heartbeat").write_text(_now_iso() + "\n")
        if idx + 1 < rounds and sleep_seconds > 0:
            ctx.log(f"sleeping {sleep_seconds}s before next round")
            await asyncio.sleep(sleep_seconds)

    return {
        "kind": "injection_campaign",
        "system": raw_args.get("system") or "ts",
        "rounds_requested": rounds,
        "rounds_completed": len(results),
        "results": results,
        "case_library_contract": "contrib/scenarios/injection/knowledge/schema.md",
    }
