"""Helpers for persisting and restoring session identity metadata."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from agentm.core.abi.catalog import ActiveSetFingerprint
from agentm.core.abi.session_api import ResolvedSessionSpec, SessionContext
from agentm.core.abi.store import SessionMeta

MetaConfigValue = str | int | float | bool | None


def session_meta_config(
    ctx: SessionContext,
    *,
    resolved_spec: ResolvedSessionSpec | None = None,
    active_set: ActiveSetFingerprint | None = None,
) -> dict[str, MetaConfigValue]:
    """Return the minimal SessionContext fields needed to resume a session."""

    config: dict[str, MetaConfigValue] = {
        "root_session_id": ctx.root_session_id,
        "depth": ctx.depth,
    }
    if ctx.scenario is not None:
        config["scenario"] = ctx.scenario
    if ctx.scenario_dir is not None:
        config["scenario_dir"] = ctx.scenario_dir
    if resolved_spec is not None:
        spec_record = _resolved_spec_record(resolved_spec)
        config["resolved_spec_digest"] = _digest_json(spec_record)
        config["resolved_spec_provenance_json"] = _stable_json(
            _json_safe(resolved_spec.provenance)
        )
    if active_set is not None:
        config["active_set_algorithm"] = active_set.algorithm
        config["active_set_digest"] = active_set.digest
        config["active_set_atom_count"] = len(active_set.atoms)
    return config


def context_from_session_meta(session_id: str, meta: SessionMeta) -> SessionContext:
    """Reconstruct the resumable SessionContext subset from SessionMeta."""

    config = meta.config
    root_session_id = _config_str(config, "root_session_id") or session_id
    return SessionContext(
        session_id=session_id,
        root_session_id=root_session_id,
        parent_session_id=meta.parent_id,
        depth=_config_int(config, "depth", default=1 if meta.parent_id else 0),
        cwd=meta.cwd,
        purpose=meta.purpose,
        scenario=_config_str(config, "scenario"),
        scenario_dir=_config_str(config, "scenario_dir"),
    )


def _config_str(config: dict[str, MetaConfigValue], key: str) -> str | None:
    value = config.get(key)
    return value if isinstance(value, str) and value else None


def _config_int(
    config: dict[str, MetaConfigValue],
    key: str,
    *,
    default: int,
) -> int:
    value = config.get(key)
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _resolved_spec_record(spec: ResolvedSessionSpec) -> dict[str, Any]:
    return {
        "scenario": spec.scenario,
        "extensions": [
            {"module": module, "config": _json_safe(config)}
            for module, config in spec.extensions
        ],
        "atom_config": {
            module: _json_safe(config)
            for module, config in sorted(spec.atom_config.items())
        },
        "provider": (
            None
            if spec.provider is None
            else {
                "module": spec.provider[0],
                "config": _json_safe(spec.provider[1]),
            }
        ),
        "provenance": _json_safe(spec.provenance),
    }


def _digest_json(value: Any) -> str:
    payload = _stable_json(value).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _stable_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return repr(value)


__all__ = [
    "context_from_session_meta",
    "session_meta_config",
]
