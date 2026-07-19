"""Helpers for persisting and restoring session identity metadata."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from agentm.core.abi.catalog import ActiveSetFingerprint
from agentm.core.abi.provider import ProviderSessionIdentity
from agentm.core.abi.session_api import ResolvedSessionSpec, SessionContext
from agentm.core.abi.store import SessionMeta

MetaConfigValue = str | int | float | bool | None


def session_meta_config(
    ctx: SessionContext,
    *,
    resolved_spec: ResolvedSessionSpec | None = None,
    active_set: ActiveSetFingerprint | None = None,
    provider_identity: ProviderSessionIdentity | None = None,
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
    if provider_identity is not None:
        config["provider_name"] = provider_identity.name
        if provider_identity.model_id is not None:
            config["provider_model_id"] = provider_identity.model_id
        if provider_identity.active_set_digest is not None:
            config["provider_active_set_digest"] = (
                provider_identity.active_set_digest
            )
        if provider_identity.frozen_after_turn_index is not None:
            config["provider_frozen_after_turn_index"] = (
                provider_identity.frozen_after_turn_index
            )
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


def provider_identity_from_session_meta(
    meta: SessionMeta,
) -> ProviderSessionIdentity | None:
    """Restore provider/model binding metadata from ``SessionMeta``."""

    name = _config_str(meta.config, "provider_name")
    if name is None:
        return None
    return ProviderSessionIdentity(
        name=name,
        model_id=_config_str(meta.config, "provider_model_id"),
        active_set_digest=_config_str(meta.config, "provider_active_set_digest"),
        frozen_after_turn_index=_config_int_optional(
            meta.config,
            "provider_frozen_after_turn_index",
        ),
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


def _config_int_optional(
    config: dict[str, MetaConfigValue],
    key: str,
) -> int | None:
    value = config.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


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
        "provider_identity": (
            None
            if spec.provider_identity is None
            else {
                "name": spec.provider_identity.name,
                "model_id": spec.provider_identity.model_id,
                "active_set_digest": spec.provider_identity.active_set_digest,
                "frozen_after_turn_index": (
                    spec.provider_identity.frozen_after_turn_index
                ),
                "metadata": _json_safe(spec.provider_identity.metadata),
            }
        ),
        "value_provenance": [
            {
                "path": item.path,
                "source": item.source,
                "source_ref": item.source_ref,
                "value_fingerprint": item.value_fingerprint,
            }
            for item in spec.value_provenance
        ],
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
    "provider_identity_from_session_meta",
    "session_meta_config",
]
