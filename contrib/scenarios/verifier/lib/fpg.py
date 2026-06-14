"""fpg vocabulary mapping and node/scenario builders.

Keyed to the profile in contrib/scenarios/verifier/fpg_profile.toml.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

PROFILE_PATH = Path(__file__).resolve().parents[1] / "fpg_profile.toml"

REL_MECHANISM = {
    "callee_to_caller": "sync_call_blocking",
    "caller_to_callee": "request_flow_disruption",
    "infra_dependency": "shared_infra_dependency",
    "co_deployed": "co_deployment_contention",
}


_FAULT_BOILERPLATE = {
    "app", "chaos_type", "namespace", "system", "system_type",
    "time_offset", "duration",
}

_NAME_FAULT_TOKENS = (
    ("container-kill", "ContainerKill"),
    ("response-abort", "HTTPAborted"),
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
            {"app": i["target"], "chaos_type": i["chaos_type"]}
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
    return [
        {
            "node_id": entry["app"],
            "fault_type": entry.get("chaos_type", "unknown"),
            "target_entity": f"svc:{entry['app']}",
            "parameters": {
                k: v for k, v in entry.items()
                if k not in _FAULT_BOILERPLATE and v not in (None, "")
            },
            "time": meta["window"],
            "replay_count": 0,
        }
        for entry in meta["engine"]
    ]



def assemble_scenario(
    meta: dict[str, Any],
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> dict[str, Any]:
    """Assemble and validate the scenario against the fpg schema."""
    from fpg import SCHEMA_VERSION, build_schema, load_profile

    schema = build_schema(load_profile(PROFILE_PATH))
    scenario = {
        "schema_version": SCHEMA_VERSION,
        "scenario_id": meta["scenario_id"],
        "testbed": meta["testbed"],
        "vocab_version": schema.profile.vocab_version,
        "injections": build_injection_records(meta),
        "graph": {"nodes": nodes, "edges": edges},
    }
    validated = schema.Scenario.model_validate(scenario)
    return validated.model_dump(mode="json", exclude_none=True)
