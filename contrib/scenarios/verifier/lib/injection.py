"""Injection metadata parsing from case injection.json."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from loguru import logger

_FAULT_BOILERPLATE = {
    "app", "chaos_type", "namespace", "system", "system_type",
    "time_offset", "duration",
}

_LINK_SCOPED_FAULTS = {
    "httpaborted",
    "httpabort",
    "httppayloadmodified",
    "httpresponsestatusmodified",
    "httpresponsestatuscodemodified",
    "httpslow",
    "networkbandwidth",
    "networkbandwidthlimit",
    "networkcorrupt",
    "networkdelay",
    "networkduplicate",
    "networkloss",
    "networkpartition",
}


def _fault_key(name: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def is_link_fault(entry: Mapping[str, Any]) -> bool:
    """Return whether an injection entry targets a service-to-service link."""
    return (
        _fault_key(entry.get("chaos_type", "")) in _LINK_SCOPED_FAULTS
        and bool(entry.get("target_service"))
    )


def fault_parameter_dict(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Return fault parameters with metadata boilerplate removed."""
    return {
        k: v for k, v in entry.items()
        if k not in _FAULT_BOILERPLATE and v not in (None, "")
    }


def _fault_params(entry: Mapping[str, Any]) -> str:
    return ", ".join(
        f"{k}={v}" for k, v in fault_parameter_dict(entry).items()
    )


def _link_endpoints(entry: Mapping[str, Any]) -> tuple[str, str] | None:
    app = str(entry.get("app") or entry.get("target") or "")
    peer = str(entry.get("target_service") or "")
    if not app or not peer:
        return None

    direction = str(entry.get("direction", "to")).lower()
    if direction == "from":
        return peer, app
    return app, peer


def enrich_injection_entry(entry: Mapping[str, Any]) -> dict[str, str]:
    """Normalize injection metadata for verifier workflow and fpg output.

    Service faults keep the historical service target. Network and HTTP proxy
    faults that specify ``target_service`` additionally get a reified link
    entity such as ``link:search->rate``. The seed agent still verifies the
    rule-bearing service side because that is where the dataset contains
    direct evidence.
    """
    app = str(entry.get("app") or entry.get("target") or "")
    chaos_type = str(entry.get("chaos_type", "unknown"))
    normalized: dict[str, str] = {
        "target": app,
        "chaos_type": chaos_type,
        "params": _fault_params(entry),
        "node_id": app,
        "subject": f"svc:{app}",
        "target_entity": f"svc:{app}",
        "effect_target": app,
    }

    endpoints = _link_endpoints(entry) if is_link_fault(entry) else None
    if endpoints:
        src, dst = endpoints
        link_ref = f"link:{src}->{dst}"
        normalized.update(
            {
                "node_id": link_ref,
                "subject": link_ref,
                "target_entity": link_ref,
                "effect_target": app,
                "edge_source": src,
                "edge_target": dst,
            }
        )
    return normalized


def get_injections(data_dir: Path) -> list[dict[str, str]]:
    """Extract normalized injection entries from injection.json."""
    injection = json.loads((data_dir / "injection.json").read_text())

    eng = injection.get("engine_config")
    if isinstance(eng, list) and eng and isinstance(eng[0], dict):
        return [enrich_injection_entry(e) for e in eng if e.get("app")]

    display = injection.get("display_config")
    if isinstance(display, str):
        try:
            display = json.loads(display)
        except Exception:
            logger.debug("Failed to parse display_config JSON")
            display = {}
    if not isinstance(display, dict):
        display = {}

    point = display.get("injection_point", {})
    target = None
    peer = None
    params: dict[str, Any] = {}
    if isinstance(point, dict):
        target = (
            point.get("source_service")
            or point.get("app_label")
            or point.get("app_name")
        )
        peer = point.get("target_service") or point.get("server_address")
        params = {
            k: point.get(k)
            for k in ("method", "route", "path", "server_port")
            if point.get(k) not in (None, "")
        }
    for k in (
        "direction",
        "latency",
        "jitter",
        "loss",
        "correlation",
        "bandwidth",
        "rate",
        "limit",
    ):
        if display.get(k) not in (None, ""):
            params[k] = display[k]
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
    except Exception as exc:
        # rcabench_platform unavailable or unmapped index — keep the raw
        # fault_type string rather than failing injection parsing.
        logger.debug("verifier injection: chaos-type mapping failed, using raw: {}", exc)
    return [
        enrich_injection_entry(
            {
                "app": target,
                "chaos_type": chaos_type,
                "target_service": peer,
                **params,
            }
        )
    ]


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
