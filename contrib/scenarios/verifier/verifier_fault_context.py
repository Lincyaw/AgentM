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

Per-fault-kind mechanism / signatures / propagation docs live as one
markdown file per kind under ``contrib/scenarios/verifier/fault_kinds/``
and are surfaced on demand by the ``verifier_fault_docs`` atom's
``get_fault_kind_doc`` tool — progressive disclosure rather than
shipping all 26 entries inline.

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


def _build_spec(data_dir: Path) -> dict[str, Any]:
    """Build the injection spec, tolerant of both dataset schemas.

    The rcabench export ships two shapes of ``injection.json``:

    * **old-style** (TrainTicket): ``fault_type`` is an integer index,
      ``ground_truth`` is a single dict, link/intensity info lives in a
      JSON-string ``display_config`` whose ``injection_point`` names the
      source / target service. One fault per case.
    * **new-style** (hs / otel-demo / ``batch-*`` hybrids):
      ``ground_truth`` is a *list* — one entry per injected fault —
      ``engine_config`` is the parallel list of per-fault chaos configs
      (``app``, ``chaos_type``, link ``direction`` / ``target_service``,
      intensity such as ``cpu_load``), and ``display_config`` is null.
      Hybrid cases carry two or more entries.

    Both collapse to a uniform ``injections`` list so the agent can judge
    each injected fault's effectiveness independently — a weak or
    ineffective fault in a hybrid case can then be excluded from the
    propagation graph rather than silently assumed effective.
    """
    injection = json.loads((data_dir / "injection.json").read_text())
    env = json.loads((data_dir / "env.json").read_text())

    injections = _build_injections(injection)

    all_targets: list[str] = []
    fault_kinds: list[str] = []
    for inj in injections:
        for t in inj["injection_targets"]:
            if t not in all_targets:
                all_targets.append(t)
        if inj["fault_kind"] not in fault_kinds:
            fault_kinds.append(inj["fault_kind"])

    return {
        "namespace": _namespace(injection, env),
        "is_hybrid": len(injections) > 1,
        "fault_count": len(injections),
        "fault_kinds": fault_kinds,
        "injections": injections,
        "all_injection_targets": all_targets,
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


def _build_injections(injection: dict[str, Any]) -> list[dict[str, Any]]:
    gt = injection.get("ground_truth")
    if isinstance(gt, list):
        eng = injection.get("engine_config")
        if not isinstance(eng, list):
            eng = injection.get("engine_config_summary")
        return _build_injections_new(gt, eng if isinstance(eng, list) else [])
    return [_build_injection_old(injection)]


def _build_injections_new(
    gt_list: list[Any], eng_list: list[Any]
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i in range(max(len(gt_list), len(eng_list))):
        gt = gt_list[i] if i < len(gt_list) and isinstance(gt_list[i], dict) else {}
        eng = eng_list[i] if i < len(eng_list) and isinstance(eng_list[i], dict) else {}
        chaos_type = eng.get("chaos_type")
        targets = _targets_from_gt(gt)
        target_service = eng.get("app") or (gt.get("service") or [None])[0]
        link = None
        if eng.get("target_service") or eng.get("direction"):
            link = {
                "source_service": eng.get("app"),
                "target_service": eng.get("target_service"),
                "direction": eng.get("direction"),
            }
        params = {
            k: v
            for k, v in eng.items()
            if k not in ("namespace", "system", "system_type", "app", "chaos_type")
        }
        out.append(
            {
                "index": i,
                "target_service": target_service,
                "fault_kind": _fault_kind_from_chaos_type(chaos_type),
                "chaos_type": chaos_type,
                "injection_targets": targets,
                "link": link,
                "parameters": params,
            }
        )
    return out


def _build_injection_old(injection: dict[str, Any]) -> dict[str, Any]:
    gt = injection.get("ground_truth")
    gt = gt if isinstance(gt, dict) else {}
    display = _parse_display(injection)
    point = display.get("injection_point") or {}

    targets = _targets_from_gt(gt)
    if not targets:
        if point.get("container_name"):
            targets.append(f"container|{point['container_name']}")
        if point.get("pod_name"):
            targets.append(f"pod|{point['pod_name']}")
        if point.get("app_label"):
            targets.append(f"service|{point['app_label']}")

    target_service = (gt.get("service") or [None])[0] or point.get("source_service")
    link = None
    if point.get("source_service") or point.get("target_service") or display.get("direction"):
        link = {
            "source_service": point.get("source_service"),
            "target_service": point.get("target_service"),
            "direction": display.get("direction"),
        }
    params = {k: v for k, v in display.items() if k not in ("injection_point", "namespace")}
    return {
        "index": 0,
        "target_service": target_service,
        "fault_kind": _map_fault_kind_index(injection.get("fault_type")),
        "chaos_type": None,
        "injection_targets": targets,
        "link": link,
        "parameters": params,
    }


def _targets_from_gt(gt: dict[str, Any]) -> list[str]:
    targets: list[str] = []
    for kind in ("service", "container", "pod"):
        for name in gt.get(kind) or []:
            t = f"{kind}|{name}"
            if t not in targets:
                targets.append(t)
    return targets


def _namespace(injection: dict[str, Any], env: dict[str, Any]) -> Any:
    ns = _parse_display(injection).get("namespace")
    if ns:
        return ns
    eng = injection.get("engine_config")
    if isinstance(eng, list):
        for e in eng:
            if isinstance(e, dict) and e.get("namespace"):
                return e["namespace"]
    return env.get("NAMESPACE")


def _parse_display(injection: dict[str, Any]) -> dict[str, Any]:
    raw = injection.get("display_config")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _as_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _fault_kind_from_chaos_type(chaos_type: Any) -> str:
    """Map a CamelCase ``chaos_type`` string (e.g. ``PodFailure``,
    ``CPUStress``, ``JVMRuntimeMutator``) to the snake_case fault_kind
    used as the ``fault_kinds/<kind>.md`` doc name."""
    if not chaos_type:
        return "unknown"
    try:
        from rcabench_platform.v3.sdk.evaluation.v2.fault_kind import map_chaos_type
    except ImportError:
        return "unknown"
    try:
        return str(map_chaos_type(str(chaos_type)).value)
    except Exception:
        return "unknown"


def _map_fault_kind_index(index: Any) -> str:
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
        except Exception as exc:  # noqa: BLE001 - never crash the session on a malformed spec
            return ToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {
                                "error": "injection_spec_parse_failed",
                                "detail": f"{type(exc).__name__}: {exc}",
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
                "Return the known fault-injection spec for this case. The "
                "spec lists one entry per injected fault under `injections` "
                "(hybrid cases inject two or more), each with its "
                "target_service, fault_kind, injection_targets, link "
                "direction (for network faults), and intensity parameters "
                "(e.g. cpu_load). `is_hybrid` and `fault_count` flag "
                "multi-fault cases. Use this as the starting point — call "
                "get_fault_kind_doc(fault_kind) per distinct fault_kind, "
                "then independently confirm via parquet data that EACH "
                "injection materialized; a weak or ineffective fault must "
                "be excluded from the propagation graph."
            ),
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
            fn=_get_spec,
        )
    )


__all__ = ["MANIFEST", "install"]
