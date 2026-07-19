"""Helpers for persisting and restoring session identity metadata."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
import math
from typing import Any

from agentm.core.abi.catalog import ActiveSetFingerprint
from agentm.core.abi.provider import ProviderSessionIdentity
from agentm.core.abi.session_api import (
    ExtensionSpec,
    ResolvedSessionSpec,
    SessionContext,
)
from agentm.core.abi.store import SessionMeta

MetaConfigValue = str | int | float | bool | None


class ResumeIdentityError(RuntimeError):
    """Raised when resume resolves a different executable session identity."""


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
        config["resolved_spec_digest"] = resolved_spec_digest(resolved_spec)
        config["resolved_spec_provenance_json"] = _stable_json(
            _metadata_record(resolved_spec.provenance, path="provenance")
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


def validate_resume_identity(
    meta: SessionMeta,
    *,
    resolved_spec: ResolvedSessionSpec | None,
    active_set: ActiveSetFingerprint | None,
) -> None:
    """Fail if current resolution does not match the stored session identity."""

    expected_spec = _config_str(meta.config, "resolved_spec_digest")
    if expected_spec is not None:
        if resolved_spec is None:
            raise ResumeIdentityError(
                "session was created from a resolved spec but resume did not resolve one"
            )
        actual_spec = resolved_spec_digest(resolved_spec)
        if actual_spec != expected_spec:
            raise ResumeIdentityError(
                "resolved session spec changed since creation: "
                f"{actual_spec} != {expected_spec}"
            )

    expected_active_set = _config_str(meta.config, "active_set_digest")
    if expected_active_set is not None:
        if active_set is None:
            raise ResumeIdentityError(
                "session has a stored active-set identity but resume produced none"
            )
        if active_set.digest != expected_active_set:
            raise ResumeIdentityError(
                "active atom set changed since session creation: "
                f"{active_set.digest} != {expected_active_set}"
            )


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


def _config_str(config: Mapping[str, MetaConfigValue], key: str) -> str | None:
    value = config.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ResumeIdentityError(f"stored session config {key!r} must be a string")
    return value


def _config_int(
    config: Mapping[str, MetaConfigValue],
    key: str,
    *,
    default: int,
) -> int:
    value = config.get(key)
    if value is None:
        return default
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ResumeIdentityError(
            f"stored session config {key!r} must be a non-negative integer"
        )
    return value


def _config_int_optional(
    config: Mapping[str, MetaConfigValue],
    key: str,
) -> int | None:
    value = config.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ResumeIdentityError(
            f"stored session config {key!r} must be a non-negative integer"
        )
    return value


def resolved_spec_digest(spec: ResolvedSessionSpec) -> str:
    """Fingerprint resolver structure without persisting credential material."""

    return _digest_json(_resolved_spec_record(spec))


def _resolved_spec_record(spec: ResolvedSessionSpec) -> dict[str, Any]:
    return {
        "scenario": spec.scenario,
        "extensions": [
            _extension_identity_record(extension)
            for extension in spec.extensions
        ],
        "atom_config_modules": sorted(spec.atom_config),
        "provider": (
            None
            if spec.provider is None
            else _extension_identity_record(spec.provider)
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
            }
        ),
        "value_provenance": _provenance_record(spec),
        "provenance": _metadata_record(spec.provenance, path="provenance"),
    }


def _extension_identity_record(spec: ExtensionSpec) -> dict[str, Any]:
    return {
        "source": {
            "kind": spec.source.kind,
            "location": spec.source.location,
            "digest": spec.source.digest,
        },
        "config_keys": sorted(spec.config),
    }


def _provenance_record(spec: ResolvedSessionSpec) -> list[dict[str, Any]]:
    return [
        {
            "path": item.path,
            "source": item.source,
            "source_ref": item.source_ref,
        }
        for item in spec.value_provenance
    ]


def _metadata_record(value: object, *, path: str) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{path} must contain finite numbers")
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError(f"{path} must use string object keys")
        return {
            key: _metadata_record(item, path=f"{path}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [
            _metadata_record(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(
        f"{path} is not JSON-safe: {type(value).__name__}; "
        "resolver provenance must contain source metadata, not runtime objects"
    )


def _digest_json(value: Any) -> str:
    payload = _stable_json(value).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _stable_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


__all__ = [
    "ResumeIdentityError",
    "context_from_session_meta",
    "provider_identity_from_session_meta",
    "resolved_spec_digest",
    "session_meta_config",
    "validate_resume_identity",
]
