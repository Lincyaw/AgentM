"""Injection metadata parsing from case injection.json."""
from __future__ import annotations

import json
import re
from pathlib import Path

_FAULT_BOILERPLATE = {
    "app", "chaos_type", "namespace", "system", "system_type",
    "time_offset", "duration",
}


def _fault_params(entry: dict) -> str:
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


# ---------------------------------------------------------------
# Fault doc loading
# ---------------------------------------------------------------

_FAULT_DOC_ALIAS = {
    "memorystress": "memstress",
    "jvmlatency": "jvmmethodlatency",
    "jvmexception": "jvmmethodexception",
    "podkill": "podfailure",
    "containerkill": "podfailure",
}


def _norm_fault(name: str) -> str:
    key = re.sub(r"[^a-z0-9]", "", name.lower())
    return _FAULT_DOC_ALIAS.get(key, key)


def load_fault_doc(fault_kind: str, fault_kinds_dir: Path) -> str:
    """Read the per-fault-kind reference doc, or return empty."""
    if not fault_kind:
        return ""
    p = fault_kinds_dir / f"{fault_kind}.md"
    if p.is_file():
        return p.read_text().strip()
    target = _norm_fault(fault_kind)
    for doc in fault_kinds_dir.glob("*.md"):
        if _norm_fault(doc.stem) == target:
            return doc.read_text().strip()
    return ""
