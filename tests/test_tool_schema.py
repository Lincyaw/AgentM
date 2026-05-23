"""Unit tests for ``pydantic_to_openai_tool_schema``.

Fail-stop position: this helper is the single source of truth for every
atom that derives its tool schema from a Pydantic model. If the
normaliser drops a required-field rewrite or fails to inline ``$defs``,
every consuming atom silently advertises a non-strict schema that some
backends compile-reject.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from agentm.core.lib import pydantic_to_openai_tool_schema


def _walk(node: Any, path: str = "$") -> list[str]:
    violations: list[str] = []
    if isinstance(node, dict):
        if node.get("type") == "object" or "properties" in node:
            if node.get("additionalProperties") is not False:
                violations.append(f"{path}: additionalProperties!=false")
            props = list((node.get("properties") or {}).keys())
            required = set(node.get("required") or [])
            missing = [p for p in props if p not in required]
            if missing:
                violations.append(f"{path}: not in required: {missing}")
        if "title" in node:
            violations.append(f"{path}: title not stripped")
        if "$ref" in node:
            violations.append(f"{path}: $ref not inlined")
        for k, v in node.items():
            violations.extend(_walk(v, f"{path}.{k}"))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            violations.extend(_walk(v, f"{path}[{i}]"))
    return violations


class _Leaf(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sql: str
    claim: str = Field(description="<=20 words")


class _Root(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Literal["a", "b"]
    items: list[_Leaf]
    note: str = Field(description="free text")


def test_no_strict_mode_violations() -> None:
    schema = pydantic_to_openai_tool_schema(_Root)
    assert _walk(schema) == []


def test_defs_inlined() -> None:
    schema = pydantic_to_openai_tool_schema(_Root)
    assert "$defs" not in schema
    leaf_items = schema["properties"]["items"]["items"]
    assert leaf_items["type"] == "object"
    assert set(leaf_items["properties"].keys()) == {"sql", "claim"}


def test_all_properties_required_even_with_defaults() -> None:
    class WithDefault(BaseModel):
        model_config = ConfigDict(extra="forbid")
        keep: str
        also: list[str] = Field(default_factory=list)

    schema = pydantic_to_openai_tool_schema(WithDefault)
    # default-bearing fields are still required in the strict schema
    assert set(schema["required"]) == {"keep", "also"}
