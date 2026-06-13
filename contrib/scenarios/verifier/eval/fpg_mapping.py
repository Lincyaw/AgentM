"""Verifier-to-fpg vocabulary mapping and node/record builders.

The verifier natively produces fault-propagation-graph (fpg) scenarios
(https://github.com/Lincyaw/fpg-convention). This module holds the
vocabulary mapping (chaos type -> seed predicate, BFS relationship type
-> edge mechanism) and the builders for the parts known before the
workflow runs: injection records and seed nodes. Everything is keyed to
the profile in contrib/scenarios/verifier/fpg_profile.toml.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

VERIFIER_DIR = Path(__file__).resolve().parents[1]
PROFILE_PATH = VERIFIER_DIR / "fpg_profile.toml"

# chaos_type (normalized: lowercase, alphanumerics only) -> seed node
# predicate. Propagated services get service_degraded — hop agents
# confirm degradation without classifying it; the evidence carries the
# specifics.
CHAOS_PREDICATE = {
    "cpustress": "cpu_saturation",
    "jvmthreadcpustress": "cpu_saturation",
    "memstress": "memory_exhaustion",
    "memorystress": "memory_exhaustion",
    "jvmheapstress": "memory_exhaustion",
    "jvmgcpressure": "gc_pressure",
    "networkdelay": "network_degraded",
    "networkloss": "network_degraded",
    "networkcorrupt": "network_degraded",
    "networkduplicate": "network_degraded",
    "networkbandwidthlimit": "network_degraded",
    "networkbandwidth": "network_degraded",
    "networkpartition": "network_partitioned",
    "podfailure": "service_unavailable",
    "podkill": "service_unavailable",
    "podunavailable": "service_unavailable",
    "containerkill": "service_unavailable",
    "httpslow": "latency_degradation",
    "jvmmethodlatency": "latency_degradation",
    "jvmlatency": "latency_degradation",
    "jvmjdbclatency": "latency_degradation",
    "httpresponsestatusmodified": "error_rate_increase",
    "httpaborted": "error_rate_increase",
    "jvmmethodexception": "error_rate_increase",
    "jvmexception": "error_rate_increase",
    "jvmjdbcexception": "error_rate_increase",
    "httppayloadmodified": "response_corruption",
    "jvmmethodmutated": "response_corruption",
    "jvmruntimemutator": "semantic_corruption",
    "dnsresolutionfailed": "dns_resolution_failure",
    "dnsresolutionwrong": "dns_resolution_failure",
    "dnserror": "dns_resolution_failure",
    "dnsrandom": "dns_resolution_failure",
    "clockskew": "clock_skew",
}

# BFS relationship type -> edge mechanism (fpg_profile.toml vocabulary).
REL_MECHANISM = {
    "callee_to_caller": "sync_call_blocking",
    "caller_to_callee": "request_flow_disruption",
    "infra_dependency": "shared_infra_dependency",
    "co_deployed": "co_deployment_contention",
}

# Same latency comparison injection.py runs for seed evidence.
_SEED_SQL = (
    "SELECT 'normal' AS win, AVG(duration)/1e3 AS avg_ms, COUNT(*) AS n "
    "FROM normal_traces WHERE service_name = '{svc}' "
    "UNION ALL "
    "SELECT 'abnormal', AVG(duration)/1e3, COUNT(*) "
    "FROM abnormal_traces WHERE service_name = '{svc}'"
)

# engine_config keys that locate the fault rather than parameterize it.
_FAULT_BOILERPLATE = {
    "app", "chaos_type", "namespace", "system", "system_type",
    "time_offset", "duration",
}


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def seed_predicate(chaos_type: str) -> str:
    return CHAOS_PREDICATE.get(_norm(chaos_type), "service_degraded")


# Legacy injection.json carries an integer fault_type whose index table
# lives in an optional package; the injection NAME (same pipeline) is then
# the only fault-kind source, e.g. "ts5-ts-station-service-pod-kill-xyz".
# Longest token first so "container-kill" wins over "kill".
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
    # Plain "-stress-" doesn't say cpu vs memory; keep the generic label
    # (predicate falls back to service_degraded, honestly).
    ("stress", "Stress"),
)


def _fault_from_name(injection_name: str) -> str | None:
    name = injection_name.lower()
    for token, chaos in _NAME_FAULT_TOKENS:
        if token in name:
            return chaos
    return None


def load_injection_meta(case_dir: Path) -> dict[str, Any]:
    """Window, testbed, scenario id, and per-target engine entries.

    Two injection.json generations exist: the current one carries an
    ``engine_config`` LIST of fault entries; the legacy one carries a
    config-tree string there, with the target in ``ground_truth``/
    ``display_config`` and an integer ``fault_type``. For the legacy
    shape, target/chaos extraction is delegated to
    ``injection.get_injections`` (the long-standing parser for it).
    """
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
        from injection import get_injections

        entries = [
            {"app": i["target"], "chaos_type": i["chaos_type"]}
            for i in get_injections(case_dir)
            if i.get("target")
        ]
        # An all-digit chaos_type means the integer fault index could not
        # be mapped; the injection name is the remaining fault-kind source.
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


def build_seed_node(
    svc: str,
    chaos_type: str,
    window: dict[str, str],
    target_evidence: dict[str, Any],
) -> dict[str, Any]:
    """fpg EventNode for an injection seed.

    Evidence is the same normal/abnormal latency comparison the harness
    pre-computes; the numbers (when available) go into the explanation.
    """
    if (
        target_evidence.get("normal_avg_ms") is not None
        and target_evidence.get("abnormal_avg_ms") is not None
    ):
        explanation = (
            f"avg latency {target_evidence['normal_avg_ms']}ms (normal) -> "
            f"{target_evidence['abnormal_avg_ms']}ms (abnormal), "
            f"x{target_evidence.get('ratio', '?')}"
        )
    else:
        explanation = (
            "injection target (seed); latency comparison between normal "
            "and abnormal windows"
        )
    return {
        "kind": "event",
        "id": svc,
        "subject": f"svc:{svc}",
        "predicate": seed_predicate(chaos_type),
        "time": window,
        "grounding": "observed",
        "evidence": [{
            "query": {"language": "sql", "statement": _SEED_SQL.format(svc=svc)},
            "explanation": explanation,
        }],
        "annotation": "auto",
    }


def assemble_scenario(
    meta: dict[str, Any],
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> dict[str, Any]:
    """Assemble and VALIDATE the scenario against the bound schema.

    Returns the validated, JSON-ready dict. Raises pydantic
    ValidationError if the graph violates the fpg rules.
    """
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
