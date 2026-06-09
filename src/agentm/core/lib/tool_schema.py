"""Convert a Pydantic model into an OpenAI-strict-mode JSON tool schema.

OpenAI's structured-outputs / function-calling strict mode imposes three
constraints on the JSON schema attached to a tool:

1. every ``object`` has ``additionalProperties: false``;
2. every key in ``properties`` is listed in ``required`` — there are no
   "optional" properties (optionality is expressed via nullable type
   unions, e.g. ``str | None`` → ``{"anyOf": [{"type": "string"},
   {"type": "null"}]}``);
3. no ``$ref`` / ``$defs`` indirection.

Pydantic's :meth:`pydantic.BaseModel.model_json_schema` produces output
that violates 1 and 3 by default and only partially satisfies 2 (fields
with default values are dropped from ``required``). This helper
normalises the output so atoms can use a single Pydantic source of truth
for both runtime validation AND the schema advertised to the LLM —
instead of hand-writing a parallel JSON schema constant that drifts.

Why this lives in ``core.lib``: it is a pure function used by atoms.
``core.lib`` is the canonical home for stateless helpers shared across
atoms and core code (see :mod:`agentm.core.lib.frontmatter`,
:mod:`agentm.core.lib.redact`).

The normaliser also strips Pydantic's auto-generated ``title`` keys
(harmless to OpenAI but visually noisy in the wire payload).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


def pydantic_to_openai_tool_schema(
    model_cls: type[BaseModel],
    *,
    strict: bool = True,
) -> dict[str, Any]:
    """Return a JSON schema for ``model_cls``, optionally OpenAI-strict.

    Args:
        model_cls: a :class:`pydantic.BaseModel` subclass.
        strict: when True (default), forces ``additionalProperties: false``
            and all-properties-required on every nested object — required
            by OpenAI's strict structured-outputs mode. Set to False for
            providers whose constrained-decoding engines reject these
            constraints (e.g. doubao).

    Returns:
        A ``$ref``-free, ``title``-free dict. With ``strict=True`` it also
        has ``additionalProperties: false`` on every object node.
    """

    raw = model_cls.model_json_schema()
    defs = raw.get("$defs", {})
    flat = _resolve_refs({k: v for k, v in raw.items() if k != "$defs"}, defs)
    if strict:
        return _force_strict(flat)
    return flat


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
            # Strip Pydantic's auto-generated "title" metadata, but
            # preserve user-defined properties named "title" (they live
            # inside a "properties" dict, not at the schema-metadata level).
            if k == "title" and not _inside_properties:
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


__all__ = ["pydantic_to_openai_tool_schema"]
