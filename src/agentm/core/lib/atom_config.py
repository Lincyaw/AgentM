"""Resolve per-atom config from manifest + env + explicit overrides.

An atom declares the config it accepts in ``MANIFEST.config_schema`` (a
JSON-Schema dict). That single declaration is the source of truth for *which*
keys exist and *what type* each is. This module reuses it to bind two extra
sources on top of the manifest-supplied ``config:`` dict:

- **Environment** — by convention ``AGENTM_<ATOM>_<KEY>`` (both upper-cased).
  Names are *generated* from the schema's declared keys, never reverse-parsed,
  so underscores in atom names (``cost_budget`` → ``AGENTM_COST_BUDGET_*``)
  are unambiguous.
- **Explicit overrides** — a ``{atom_name: {key: value}}`` map, populated by
  the CLI ``--set <atom>.<key>=<value>`` flag (or any embedder).

Precedence (highest wins)::

    overrides  >  env (AGENTM_<ATOM>_<KEY>)  >  manifest config:

This is pure *mechanism*: it only ever touches keys an atom declared (env) or a
caller named (overrides), and it does not invent defaults — an atom's own
``config.get(key, default)`` remains the single place defaults live. String
values (always the case for env, usually for CLI) are coerced to the schema's
declared type; already-typed values supplied programmatically pass through
verbatim.
"""

from __future__ import annotations

import importlib
import json
import math
import os
from collections.abc import Mapping
from typing import Any

from agentm.core.abi import ExtensionManifest

_TRUE_TOKENS = frozenset({"1", "true", "yes", "on"})
_FALSE_TOKENS = frozenset({"0", "false", "no", "off"})


class AtomConfigError(ValueError):
    """A ``--set`` / env value could not be coerced to its declared type.

    Carries the atom name and key so the session factory can surface a
    user-actionable diagnostic instead of an anonymous ``ValueError``.
    """


def resolve_atom_configs(
    extensions: list[tuple[str, dict[str, Any]]],
    *,
    env: Mapping[str, str] | None = None,
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    """Overlay env vars and explicit overrides onto each atom's config.

    ``extensions`` is the ``(module_path, config)`` list the session factory
    has assembled (scenario + extras + user atoms). Returns a new list with
    fresh config dicts; the inputs are not mutated. Atoms whose module cannot
    be imported, or that expose no ``MANIFEST.config_schema``, are passed
    through unchanged (no declared keys ⇒ nothing to bind from env; explicit
    overrides still apply by name).
    """

    env = os.environ if env is None else env
    overrides = overrides or {}

    resolved: list[tuple[str, dict[str, Any]]] = []
    for module_path, config in extensions:
        manifest = _load_manifest(module_path)
        atom_name = manifest.name if manifest is not None else None
        schema = manifest.config_schema if manifest is not None else None
        properties = _properties(schema)

        merged = dict(config)

        # 1) env, keyed by the schema's declared property names only.
        if atom_name is not None:
            for key, prop in properties.items():
                env_name = f"AGENTM_{atom_name.upper()}_{key.upper()}"
                raw = env.get(env_name)
                if raw is not None:
                    merged[key] = _coerce(atom_name, key, raw, prop)

        # 2) explicit overrides win over env + manifest.
        atom_overrides = overrides.get(atom_name) if atom_name is not None else None
        if atom_overrides:
            for key, value in atom_overrides.items():
                merged[key] = _coerce(
                    atom_name or module_path, key, value, properties.get(key)
                )

        resolved.append((module_path, merged))
    return resolved


def _load_manifest(module_path: str) -> ExtensionManifest | None:
    try:
        module = importlib.import_module(module_path)
    except Exception:  # noqa: BLE001 — unimportable atoms pass through untouched.
        return None
    manifest = getattr(module, "MANIFEST", None)
    return manifest if isinstance(manifest, ExtensionManifest) else None


def _properties(schema: Any) -> dict[str, Any]:
    from pydantic import BaseModel as _PydanticBaseModel

    if isinstance(schema, type) and issubclass(schema, _PydanticBaseModel):
        json_schema = schema.model_json_schema()
        props = json_schema.get("properties")
        return props if isinstance(props, dict) else {}
    if not isinstance(schema, dict):
        return {}
    props = schema.get("properties")
    return props if isinstance(props, dict) else {}


def _coerce(atom: str, key: str, value: Any, prop: Any) -> Any:
    """Coerce a (usually string) value to the schema property's type.

    Non-string values pass through verbatim — an embedder that supplies a
    typed override (``{"limit": 5.0}``) should not be re-parsed. Strings are
    coerced by the declared ``type``. A key with no declared type (undeclared
    ``additionalProperties`` keys, or a property lacking ``type``) keeps the
    raw string: free text stays text rather than being silently re-typed
    (a ``"123"`` label must not become an int). Malformed values for a typed
    key raise :class:`AtomConfigError` naming the atom + key.
    """

    if not isinstance(value, str):
        return value

    declared = prop.get("type") if isinstance(prop, dict) else None
    try:
        if declared == "integer":
            return int(value)
        if declared == "number":
            parsed = float(value)
            if not math.isfinite(parsed):
                raise ValueError(f"expected a finite number, got {value!r}")
            return parsed
        if declared == "boolean":
            token = value.strip().lower()
            if token in _TRUE_TOKENS:
                return True
            if token in _FALSE_TOKENS:
                return False
            raise ValueError(f"expected a boolean, got {value!r}")
        if declared in {"array", "object"}:
            return json.loads(value)
    except (ValueError, json.JSONDecodeError) as exc:
        raise AtomConfigError(
            f"config {atom}.{key}: cannot coerce {value!r} to {declared}: {exc}"
        ) from exc
    # ``string`` or no declared type: keep the raw string verbatim.
    return value


__all__ = ["AtomConfigError", "resolve_atom_configs"]
