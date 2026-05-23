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

    spec = {
        "namespace": namespace,
        "fault_kind": fault_kind,
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
