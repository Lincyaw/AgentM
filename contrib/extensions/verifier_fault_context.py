"""Verifier-scenario atom: expose the known fault-injection spec.

The verifier task assumes the injected fault is KNOWN — the agent's job is
to confirm the injection materialized in the observability data and to
trace the downstream propagation. This atom mounts a single read-only
tool, ``get_injection_spec``, that returns:

* the injection target(s) — component identifiers shaped like
  ``container|<name>``, ``pod|<name>``, ``service|<name>`` so the agent
  can match against the propagation-graph node vocabulary used by the
  rcabench dataset;
* the canonical ``fault_kind`` (mapped from the integer ``fault_type``
  via :mod:`rcabench_platform.v3.sdk.evaluation.v2.fault_kind`);
* the normal / abnormal observation windows from ``env.json`` (unix
  seconds; what the agent should use as ``WHERE time BETWEEN ...``).

Inputs:

* ``injection.json`` — written by the rcabench fault-injection platform.
* ``env.json`` — written by the same data export step.

Both live under ``AGENTM_RCA_DATA_DIR`` (or the ``data_dir`` config
override). ``causal_graph.json``, ``label.txt``, ``result.json`` and
``conclusion.parquet`` are GROUND TRUTH and are deliberately NOT
surfaced — the verifier exists in part to validate them, so it must not
read them.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from agentm.core.abi import FunctionTool, ToolResult
from agentm.core.abi.messages import TextContent
from agentm.core.abi.extension import ExtensionAPI
from agentm.extensions import ExtensionManifest


MANIFEST = ExtensionManifest(
    name="verifier_fault_context",
    description=(
        "Exposes get_injection_spec() — the known fault-injection target, "
        "kind, and time windows. The verifier's starting point."
    ),
    registers=("tool:get_injection_spec",),
    config_schema={
        "type": "object",
        "properties": {
            "data_dir": {"type": "string"},
        },
        "additionalProperties": False,
    },
)


def _resolve_data_dir(config: dict[str, Any]) -> Path:
    raw = config.get("data_dir") or os.environ.get("AGENTM_RCA_DATA_DIR")
    if not raw:
        raise RuntimeError(
            "verifier_fault_context: data_dir not configured and "
            "AGENTM_RCA_DATA_DIR is not set"
        )
    return Path(raw).resolve()


# ---------------------------------------------------------------------------
# Per-fault-kind mechanism table.
#
# Each entry states two facts in plain language:
#   - what_happens: what the injection physically does to the target
#   - signatures:   what trace / log / metric signals it tends to produce
#
# The table is keyed on the rcabench-platform FaultKind .value strings.
# All 26 enum values are covered; less-common kinds get conservative
# descriptions. The whole dict ships as JSON inside get_injection_spec so
# the agent always sees it, regardless of which skills it loaded.
# ---------------------------------------------------------------------------


FAULT_MECHANISMS: dict[str, dict[str, str]] = {
    "pod_failure": {
        "what_happens": (
            "Target pod is killed; it stays down until kubelet restarts it. "
            "All traffic to and from the target fails for the duration."
        ),
        "signatures": (
            "Target span volume drops sharply (often to zero). Restart "
            "counter on the target's container increments. Callers log "
            "connection errors mentioning the target."
        ),
    },
    "pod_unavailable": {
        "what_happens": (
            "Target container's process is killed; the pod object stays "
            "and the container restarts in place. Equivalent to a brief "
            "pod_failure window."
        ),
        "signatures": (
            "Brief gap in target span volume; container restart counter "
            "increments; callers log transient connection errors."
        ),
    },
    "network_delay": {
        "what_happens": (
            "tc netem adds latency on the target pod's network interface, "
            "affecting both inbound and outbound packets. Target's own CPU "
            "and memory are untouched."
        ),
        "signatures": (
            "Target inbound span duration rises by roughly the configured "
            "delay. Target outbound spans also slow. Error rate usually "
            "stable unless delay pushes callers past timeout."
        ),
    },
    "network_loss": {
        "what_happens": (
            "tc netem drops a fraction of packets on the target's interface. "
            "TCP retransmits cover small losses; large loss leads to "
            "connection timeouts."
        ),
        "signatures": (
            "Inbound and outbound spans on the target show elevated error "
            "rate and tail latency. Both inbound callers and outbound peers "
            "log connectivity errors against the target."
        ),
    },
    "network_partition": {
        "what_happens": (
            "iptables drops all packets to and from the target pod. "
            "Target's process is alive but cannot reach the network in "
            "either direction."
        ),
        "signatures": (
            "Target's outbound spans error out; inbound spans never reach "
            "the target. Target logs connection errors against its "
            "dependencies; callers log connection errors against the target."
        ),
    },
    "network_corrupt": {
        "what_happens": (
            "tc netem corrupts random bytes on packets through the target's "
            "interface. Most are caught by TCP checksums; some payloads "
            "deserialise wrong at the receiver."
        ),
        "signatures": (
            "Elevated retransmits and tail latency on target's traffic; "
            "occasional decode / deserialisation errors at callers."
        ),
    },
    "network_duplicate": {
        "what_happens": (
            "Random packet duplication on the target's interface. TCP "
            "usually dedupes transparently."
        ),
        "signatures": (
            "Often subtle. Slight throughput overhead; rarely a visible "
            "application-level signal. Be ready to mark ambiguous."
        ),
    },
    "network_bandwidth_limit": {
        "what_happens": (
            "Bandwidth cap on the target's interface. Small requests pass "
            "near-normal; large payloads queue and slow."
        ),
        "signatures": (
            "Latency rises only on spans that move large payloads; small "
            "spans look normal."
        ),
    },
    "http_aborted": {
        "what_happens": (
            "A chaos proxy at the target intercepts HTTP traffic and aborts "
            "matching requests, returning 5xx or resetting the connection. "
            "Only spans matching the configured path / method are affected."
        ),
        "signatures": (
            "Inbound spans on the target on the matched endpoint show "
            "5xx / connection-reset; sibling endpoints unaffected. The "
            "5xx is emitted by the target's proxy, not by any downstream."
        ),
    },
    "http_slow": {
        "what_happens": (
            "Chaos proxy delays HTTP responses from the target on matching "
            "paths. Target's compute itself is unchanged."
        ),
        "signatures": (
            "Inbound spans on the matched endpoint show a flat latency "
            "shift equal to the injected delay; error rate usually stable."
        ),
    },
    "http_payload_modified": {
        "what_happens": (
            "Chaos proxy mutates response body bytes on matching paths."
        ),
        "signatures": (
            "Callers of the affected endpoint show deserialisation / "
            "validation errors; target's status codes may still be 2xx."
        ),
    },
    "http_response_status_modified": {
        "what_happens": (
            "Chaos proxy rewrites HTTP response status codes on matching "
            "paths (typically 2xx → 5xx)."
        ),
        "signatures": (
            "Inbound spans on the matched endpoint show altered status; "
            "callers treat-as-failure."
        ),
    },
    "cpu_stress": {
        "what_happens": (
            "Stress-ng spawns busy threads inside the target pod, "
            "saturating CPU. Target request handlers run slowly."
        ),
        "signatures": (
            "Target CPU utilisation spikes; target inbound span latency "
            "rises; error rate may rise if timeouts trip."
        ),
    },
    "jvm_thread_cpu_stress": {
        "what_happens": (
            "JVM agent pins target threads to busy-loop inside the JVM, "
            "burning CPU."
        ),
        "signatures": "Same observable shape as cpu_stress.",
    },
    "mem_stress": {
        "what_happens": (
            "Stress-ng allocates memory inside the target pod toward OOM. "
            "May trigger GC-thrashing or OOM-kill (which degenerates to a "
            "pod_failure-like window)."
        ),
        "signatures": (
            "Target memory utilisation climbs; latency rises during GC; "
            "if OOM-killed, target shows a pod_failure-style gap."
        ),
    },
    "jvm_heap_stress": {
        "what_happens": (
            "JVM agent fills the heap, forcing GC pressure or OOM. "
            "Request handlers stall during GC pauses."
        ),
        "signatures": (
            "Target inbound latency shows multi-second spikes correlated "
            "with GC; heap metrics rise."
        ),
    },
    "jvm_gc_pressure": {
        "what_happens": "JVM agent triggers frequent full GCs on the target.",
        "signatures": "Frequent latency spikes; GC time fraction rises.",
    },
    "jvm_method_exception": {
        "what_happens": (
            "JVM agent rewrites a specific method on the target to throw. "
            "Only call sites entering that method fail; siblings unaffected."
        ),
        "signatures": (
            "Target spans whose name matches the targeted method show "
            "errors; other span names on the same service look normal."
        ),
    },
    "jvm_jdbc_exception": {
        "what_happens": (
            "JVM agent rewrites the target's JDBC code path to throw, so "
            "the target's outbound DB calls fail. The DB itself is healthy."
        ),
        "signatures": (
            "Target's DB-call sub-spans error out; the DB shows no "
            "anomalies of its own."
        ),
    },
    "jvm_method_latency": {
        "what_happens": (
            "JVM agent inserts a sleep into a specific method on the "
            "target. Only that method slows."
        ),
        "signatures": (
            "Target spans matching the targeted method name show a flat "
            "latency shift; sibling methods unaffected."
        ),
    },
    "jvm_jdbc_latency": {
        "what_happens": (
            "JVM agent inserts latency into the target's JDBC layer. The "
            "DB itself is unaffected."
        ),
        "signatures": (
            "Target's DB-call sub-spans show elevated duration; DB-side "
            "metrics are normal."
        ),
    },
    "jvm_method_mutated": {
        "what_happens": (
            "JVM agent replaces a method's body with mutated logic. "
            "Behaviour change rather than outright failure."
        ),
        "signatures": (
            "Subtle — wrong responses or downstream side effects. May not "
            "show in latency / error-rate alone; consider ambiguous."
        ),
    },
    "dns_resolution_failed": {
        "what_happens": (
            "DNS resolution inside the target pod fails. Target's outbound "
            "calls cannot resolve peer hostnames. Inbound to the target's "
            "IP is unaffected."
        ),
        "signatures": (
            "Target logs name-resolution errors; target's outbound spans "
            "error out before reaching any peer."
        ),
    },
    "dns_resolution_wrong": {
        "what_happens": (
            "DNS resolution inside the target returns wrong IPs. Outbound "
            "calls hit unintended hosts or hang."
        ),
        "signatures": (
            "Target's outbound spans show unexpected errors or timeouts."
        ),
    },
    "clock_skew": {
        "what_happens": (
            "Target's system clock is shifted. App-specific consequences: "
            "token expiry, time-window logic, trace timestamps."
        ),
        "signatures": (
            "Often silent in observability data. Be ready to mark ambiguous."
        ),
    },
    "unknown": {
        "what_happens": "Unmapped fault_kind; reason from the raw injection_point.",
        "signatures": "Probe target's span volume, error rate, and latency.",
    },
}



def _build_spec(data_dir: Path) -> dict[str, Any]:
    injection = json.loads((data_dir / "injection.json").read_text())
    env = json.loads((data_dir / "env.json").read_text())

    display = json.loads(injection.get("display_config") or "{}")
    point = display.get("injection_point") or {}
    namespace = display.get("namespace") or env.get("NAMESPACE")

    # Component identifiers in the rcabench causal-graph vocabulary:
    # "<type>|<name>". The verifier emits nodes / edges using the same
    # shape so downstream evaluation can compare directly.
    targets: list[str] = []
    gt = injection.get("ground_truth") or {}
    for kind in ("container", "pod", "service"):
        for name in gt.get(kind) or []:
            targets.append(f"{kind}|{name}")
    # Fall back to display_config if ground_truth is empty.
    if not targets:
        if point.get("container_name"):
            targets.append(f"container|{point['container_name']}")
        if point.get("pod_name"):
            targets.append(f"pod|{point['pod_name']}")
        if point.get("app_label"):
            targets.append(f"service|{point['app_label']}")

    fault_kind = _map_fault_kind(injection.get("fault_type"))
    mechanism = FAULT_MECHANISMS.get(fault_kind, FAULT_MECHANISMS["unknown"])

    spec = {
        "namespace": namespace,
        "fault_kind": fault_kind,
        "mechanism": mechanism,
        "raw_fault_type_index": injection.get("fault_type"),
        "injection_targets": targets,
        "injection_point": point,
        "windows": {
            "normal_start": _as_int(env.get("NORMAL_START")),
            "normal_end": _as_int(env.get("NORMAL_END")),
            "abnormal_start": _as_int(env.get("ABNORMAL_START")),
            "abnormal_end": _as_int(env.get("ABNORMAL_END")),
            "timezone": env.get("TIMEZONE"),
        },
        "injection_start": injection.get("start_time"),
        "injection_end": injection.get("end_time"),
        "pre_duration_minutes": injection.get("pre_duration"),
    }
    return spec


def _as_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _map_fault_kind(index: Any) -> str:
    if index is None:
        return "unknown"
    try:
        from rcabench_platform.v3.sdk.evaluation.v2.fault_kind import (
            chaos_type_from_index,
            map_chaos_type,
        )
    except ImportError:
        return "unknown"
    try:
        chaos = chaos_type_from_index(int(index))
        return str(map_chaos_type(chaos).value)
    except Exception:
        return "unknown"


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    data_dir = _resolve_data_dir(config)

    async def _get_spec(args: dict[str, Any]) -> ToolResult:
        try:
            spec = _build_spec(data_dir)
        except FileNotFoundError as exc:
            return ToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {
                                "error": "data_file_missing",
                                "detail": str(exc),
                                "data_dir": str(data_dir),
                            }
                        ),
                    )
                ],
                is_error=True,
            )
        return ToolResult(
            content=[
                TextContent(
                    type="text",
                    text=json.dumps(spec, ensure_ascii=False, indent=2),
                )
            ]
        )

    api.register_tool(
        FunctionTool(
            name="get_injection_spec",
            description=(
                "Return the known fault-injection spec for this case: "
                "target component(s), fault_kind, and the normal / abnormal "
                "observation windows. Use this as the starting point — the "
                "agent must independently confirm via parquet data that the "
                "injection actually materialized."
            ),
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
            fn=_get_spec,
        )
    )


__all__ = ["MANIFEST", "install"]
