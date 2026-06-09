"""Convert a Pydantic model into a provider-neutral JSON tool schema.

Pydantic's :meth:`pydantic.BaseModel.model_json_schema` produces output
with ``$ref`` / ``$defs`` indirection and auto-generated ``title`` keys.
This helper normalises the output: inlines ``$defs`` and strips metadata
``title`` keys (preserving user-defined properties named "title"), so
atoms can use a single Pydantic source of truth for both runtime
validation AND the schema advertised to the LLM — instead of hand-writing
a parallel JSON schema constant that drifts.

Provider-specific constraints (e.g. OpenAI strict mode requiring
``additionalProperties: false`` and full ``required``) are applied by the
provider adapter layer, not here. See :func:`_force_strict` for the
mutation applied by the OpenAI adapter.

Why this lives in ``core.lib``: it is a pure function used by atoms.
``core.lib`` is the canonical home for stateless helpers shared across
atoms and core code (see :mod:`agentm.core.lib.frontmatter`,
:mod:`agentm.core.lib.redact`).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


def pydantic_to_tool_schema(
    model_cls: type[BaseModel],
) -> dict[str, Any]:
    """Return a provider-neutral JSON schema for ``model_cls``.

    Performs ``model_json_schema()`` → resolve ``$ref`` / ``$defs`` →
    strip metadata ``title`` keys (preserving user properties named
    "title").

    Provider-specific constraints (OpenAI strict mode, etc.) are NOT
    applied here — they belong in the provider adapter layer.
    """

    raw = model_cls.model_json_schema()
    defs = raw.get("$defs", {})
    return _resolve_refs({k: v for k, v in raw.items() if k != "$defs"}, defs)


# Backward-compatible alias — deprecated, use pydantic_to_tool_schema.
pydantic_to_openai_tool_schema = pydantic_to_tool_schema


def _resolve_refs(node: Any, defs: dict[str, Any], *, _inside_properties: bool = False) -> Any:
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            target = defs.get(ref[len("#/$defs/") :])
            if isinstance(target, dict):
                resolved = _resolve_refs(target, defs)
                sibling = {k: v for k, v in node.items() if k != "$ref"}
                if sibling:
                    return {**resolved, **_resolve_refs(sibling, defs)}
                return resolved
        out: dict[str, Any] = {}
        for k, v in node.items():
            # Strip Pydantic's auto-generated metadata keys:
            # - "title": noisy in wire payloads, not needed by LLMs.
            # - "additionalProperties": validation concern, not schema
            #   description; _force_strict adds it back for OpenAI strict
            #   mode in the provider adapter.
            # Preserve user-defined *properties* named "title" (they live
            # inside a "properties" dict, not at the schema-metadata level).
            if k == "title" and not _inside_properties:
                continue
            if k == "additionalProperties":
                continue
            child_inside = k == "properties"
            out[k] = _resolve_refs(v, defs, _inside_properties=child_inside)
        return out
    if isinstance(node, list):
        return [_resolve_refs(item, defs) for item in node]
    return node


def _force_strict(node: Any) -> Any:
    if isinstance(node, dict):
        if node.get("type") == "object" or "properties" in node:
            node["additionalProperties"] = False
            props = list((node.get("properties") or {}).keys())
            node["required"] = props
        for value in node.values():
            _force_strict(value)
    elif isinstance(node, list):
        for value in node:
            _force_strict(value)
    return node


__all__ = ["_force_strict", "pydantic_to_openai_tool_schema", "pydantic_to_tool_schema"]
