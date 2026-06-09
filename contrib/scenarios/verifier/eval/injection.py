"""Injection parsing: extract fault targets and evidence from case data."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TypedDict

from graph import _duckdb_conn

REPO = Path(__file__).resolve().parents[4]
FAULT_KINDS_DIR = REPO / "contrib" / "scenarios" / "verifier" / "fault_kinds"


class TargetEvidence(TypedDict, total=False):
    normal_avg_ms: float
    abnormal_avg_ms: float
    ratio: float

# engine_config keys that describe WHERE/WHEN, not the fault's intensity.
# Everything else (corrupt %, loss %, delay, mutator_config, method, …) is
# the strength the hop agent needs to judge how much impact to expect.
_FAULT_BOILERPLATE = {
    "app", "chaos_type", "namespace", "system", "system_type",
    "time_offset", "duration",
}


def _fault_params(entry: dict) -> str:
    """Render one engine_config entry's intensity/config as ``k=v, …``."""
    return ", ".join(
        f"{k}={v}" for k, v in entry.items()
        if k not in _FAULT_BOILERPLATE and v not in (None, "")
    )


def get_injections(data_dir: Path) -> list[dict[str, str]]:
    """Extract ``[{target, chaos_type, params}]`` from injection.json."""
    injection = json.loads((data_dir / "injection.json").read_text())

    eng = injection.get("engine_config")
    if isinstance(eng, list) and eng and isinstance(eng[0], dict):
        return [
            {
                "target": e["app"],
                "chaos_type": e.get("chaos_type", "unknown"),
                "params": _fault_params(e),
            }
            for e in eng
        ]

    display = injection.get("display_config")
    if isinstance(display, str):
        try:
            display = json.loads(display)
        except Exception:
            display = {}
    if not isinstance(display, dict):
        display = {}

    point = display.get("injection_point", {})
    target = None
    if isinstance(point, dict):
        target = point.get("source_service") or point.get("app_label")
    if not target:
        gt = injection.get("ground_truth")
        if isinstance(gt, dict):
            svcs = gt.get("service") or []
            target = svcs[0] if svcs else None
        elif isinstance(gt, list) and gt and isinstance(gt[0], dict):
            svcs = gt[0].get("service") or []
            target = svcs[0] if svcs else None

    if not target:
        return []

    chaos_type = str(injection.get("fault_type", "unknown"))
    try:
        from rcabench_platform.v3.sdk.evaluation.v2.fault_kind import (
            chaos_type_from_index,
            map_chaos_type,
        )
        chaos_type = str(map_chaos_type(
            chaos_type_from_index(int(chaos_type))
        ).value)
    except Exception:
        pass
    return [{"target": target, "chaos_type": chaos_type, "params": ""}]


def get_target_evidence(data_dir: Path, target: str) -> TargetEvidence:
    """Quick SQL check: latency comparison for the injection target."""
    conn = _duckdb_conn(data_dir)
    # duration is microseconds — convert to ms with /1e3
    rows = conn.execute(
        "SELECT 'normal' AS win, AVG(duration)/1e3, COUNT(*) "
        "FROM normal_traces WHERE service_name = ? "
        "UNION ALL "
        "SELECT 'abnormal', AVG(duration)/1e3, COUNT(*) "
        "FROM abnormal_traces WHERE service_name = ?",
        [target, target],
    ).fetchall()
    conn.close()

    if len(rows) == 2 and rows[0][1] and rows[1][1]:
        ratio = rows[1][1] / rows[0][1] if rows[0][1] > 0 else 0
        return {
            "normal_avg_ms": round(rows[0][1], 3),
            "abnormal_avg_ms": round(rows[1][1], 3),
            "ratio": round(ratio, 1),
        }
    return {}


# chaos_type names whose doc filename abbreviates differently than a plain
# normalization would catch (same fault, shorter slug).
_FAULT_DOC_ALIAS = {
    "memorystress": "memstress",
    "jvmlatency": "jvmmethodlatency",
    "podkill": "podfailure",
    "containerkill": "podfailure",
}


def _norm_fault(name: str) -> str:
    """Lowercase, strip non-alphanumerics — so ``NetworkLoss`` matches
    ``network_loss``, ``PodFailure`` matches ``pod_failure``, etc."""
    key = re.sub(r"[^a-z0-9]", "", name.lower())
    return _FAULT_DOC_ALIAS.get(key, key)


def _load_fault_doc(fault_kind: str) -> str:
    """Read the per-fault-kind reference doc, or return empty.

    chaos_type arrives CamelCase (``NetworkLoss``) while the docs are
    snake_case (``network_loss.md``); match on a normalized name so the
    reference actually loads instead of silently missing.
    """
    if not fault_kind:
        return ""
    p = FAULT_KINDS_DIR / f"{fault_kind}.md"
    if p.is_file():
        return p.read_text().strip()
    target = _norm_fault(fault_kind)
    for doc in FAULT_KINDS_DIR.glob("*.md"):
        if _norm_fault(doc.stem) == target:
            return doc.read_text().strip()
    return ""
