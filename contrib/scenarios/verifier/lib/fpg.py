"""fpg vocabulary mapping and node/scenario builders.

Keyed to the profile in contrib/scenarios/verifier/fpg_profile.toml.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .injection import enrich_injection_entry, fault_parameter_dict

PROFILE_PATH = Path(__file__).parents[1] / "fpg_profile.toml"

REL_MECHANISM = {
    "callee_to_caller": "sync_call_blocking",
    "caller_to_callee": "request_flow_disruption",
    "infra_dependency": "shared_infra_dependency",
    "co_deployed": "co_deployment_contention",
    "link_to_service": "network_path_effect",
}


_NAME_FAULT_TOKENS = (
    ("container-kill", "ContainerKill"),
    ("response-replace-code", "HTTPResponseStatusModified"),
    ("request-abort", "HTTPAborted"),
    ("response-abort", "HTTPAborted"),
    ("request-delay", "HTTPSlow"),
    ("response-delay", "HTTPSlow"),
    ("request-replace", "HTTPPayloadModified"),
    ("response-replace", "HTTPPayloadModified"),
    ("pod-failure", "PodFailure"),
    ("pod-kill", "PodKill"),
    ("partition", "NetworkPartition"),
    ("bandwidth", "NetworkBandwidthLimit"),
    ("duplicate", "NetworkDuplicate"),
    ("corrupt", "NetworkCorrupt"),
    ("delay", "NetworkDelay"),
    ("loss", "NetworkLoss"),
    ("cpu-exhaustion", "CPUStress"),
    ("memory-exhaustion", "MemoryStress"),
    ("stress", "Stress"),
)


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())



def _fault_from_name(injection_name: str) -> str | None:
    name = injection_name.lower()
    for token, chaos in _NAME_FAULT_TOKENS:
        if token in name:
            return chaos
    return None


def _engine_entry_from_normalized_injection(
    injection: dict[str, str],
) -> dict[str, Any]:
    """Rebuild an engine-like entry without losing link endpoints."""
    target = injection["target"]
    entry: dict[str, Any] = {
        "app": target,
        "chaos_type": injection["chaos_type"],
    }
    src = injection.get("edge_source")
    dst = injection.get("edge_target")
    if src and dst:
        if src == target:
            entry["target_service"] = dst
        else:
            entry["target_service"] = src
            entry["direction"] = "from"
    return entry


def load_injection_meta(case_dir: Path) -> dict[str, Any]:
    """Window, testbed, scenario id, and per-target engine entries."""
    injection = json.loads((case_dir / "injection.json").read_text())
    eng = injection.get("engine_config")
    if isinstance(eng, str):
        try:
            eng = json.loads(eng)
        except ValueError:
            eng = None
    entries = (
        [e for e in eng if isinstance(e, dict) and e.get("app")]
        if isinstance(eng, list) else []
    )
    if not entries:
        from .injection import get_injections

        entries = [
            _engine_entry_from_normalized_injection(i)
            for i in get_injections(case_dir)
            if i.get("target")
        ]
        named = _fault_from_name(
            injection.get("injection_name") or injection.get("name") or case_dir.name
        )
        if named:
            for entry in entries:
                if str(entry.get("chaos_type", "")).isdigit() or entry.get(
                    "chaos_type"
                ) in ("", "unknown"):
                    entry["chaos_type"] = named
    if not entries:
        raise ValueError(f"{case_dir.name}: no injection targets in injection.json")

    display = injection.get("display_config")
    if isinstance(display, str):
        try:
            display = json.loads(display)
        except ValueError:
            display = {}
    namespace = display.get("namespace") if isinstance(display, dict) else None

    return {
        "window": {
            "start": injection["start_time"],
            "end": injection["end_time"],
        },
        "scenario_id": injection.get("name")
        or injection.get("injection_name") or case_dir.name,
        "testbed": injection.get("pedestal_name")
        or injection.get("benchmark_name") or namespace
        or injection.get("benchmark") or "unknown",
        "engine": entries,
    }


def build_injection_records(meta: dict[str, Any]) -> list[dict[str, Any]]:
    """fpg Injection records, one per engine_config entry."""
    records: list[dict[str, Any]] = []
    for entry in meta["engine"]:
        normalized = enrich_injection_entry(entry)
        records.append(
            {
                "node_id": normalized["node_id"],
                "fault_type": normalized["chaos_type"],
                "target_entity": normalized["target_entity"],
                "parameters": fault_parameter_dict(entry),
                "time": meta["window"],
                "replay_count": 0,
            }
        )
    return records



def assemble_scenario(
    meta: dict[str, Any],
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    confirmed_seed_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Assemble and validate the scenario against the fpg schema."""
    from fpg import SCHEMA_VERSION, build_schema, load_profile

    schema = build_schema(load_profile(PROFILE_PATH))
    node_ids = {n["id"] for n in nodes}
    injections = [
        r for r in build_injection_records(meta)
        if r["node_id"] in node_ids
        and (confirmed_seed_ids is None or r["node_id"] in confirmed_seed_ids)
    ]
    scenario = {
        "schema_version": SCHEMA_VERSION,
        "scenario_id": meta["scenario_id"],
        "testbed": meta["testbed"],
        "vocab_version": schema.profile.vocab_version,
        "injections": injections,
        "graph": {"nodes": nodes, "edges": edges},
    }
    try:
        validated = schema.Scenario.model_validate(scenario)
        return validated.model_dump(mode="json", exclude_none=True)
    except ValidationError as exc:
        root_only = all(
            "must be a root (no incoming edges)" in str(err.get("ctx", {}).get("error", ""))
            or "must be a root (no incoming edges)" in str(err.get("msg", ""))
            for err in exc.errors()
        )
        if not root_only:
            raise

    graph = schema.Graph.model_validate(scenario["graph"]).model_dump(
        mode="json", exclude_none=True
    )
    validated_injections = [
        schema.Injection.model_validate(injection).model_dump(
            mode="json", exclude_none=True
        )
        for injection in scenario["injections"]
    ]
    return {
        "schema_version": scenario["schema_version"],
        "scenario_id": scenario["scenario_id"],
        "testbed": scenario["testbed"],
        "vocab_version": scenario["vocab_version"],
        "injections": validated_injections,
        "graph": graph,
    }
