"""Unit tests for ``pydantic_to_tool_schema`` and ``_force_strict``.

Fail-stop position: this helper is the single source of truth for every
atom that derives its tool schema from a Pydantic model. If the
normaliser fails to inline ``$defs`` or strip ``title`` keys,
every consuming atom silently advertises a broken schema. If
``_force_strict`` (used by the OpenAI adapter) stops enforcing
``additionalProperties: false`` and full ``required``, OpenAI strict
mode rejects the tool.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from agentm.core.lib import pydantic_to_tool_schema
from agentm.core.lib.tool_schema import _force_strict


def _walk_base(node: Any, path: str = "$") -> list[str]:
    """Check provider-neutral invariants: no title, no default, no $ref."""
    violations: list[str] = []
    if isinstance(node, dict):
        if "title" in node:
            violations.append(f"{path}: title not stripped")
        if "default" in node:
            violations.append(f"{path}: default not stripped")
        if "$ref" in node:
            violations.append(f"{path}: $ref not inlined")
        for k, v in node.items():
            violations.extend(_walk_base(v, f"{path}.{k}"))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            violations.extend(_walk_base(v, f"{path}[{i}]"))
    return violations


def _walk_strict(node: Any, path: str = "$") -> list[str]:
    """Check strict-mode invariants on top of base."""
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
        for k, v in node.items():
            violations.extend(_walk_strict(v, f"{path}.{k}"))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            violations.extend(_walk_strict(v, f"{path}[{i}]"))
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


def test_base_schema_refs_inlined_and_titles_stripped() -> None:
    schema = pydantic_to_tool_schema(_Root)
    violations = _walk_base(schema)
    assert violations == [], violations


def test_force_strict_adds_strict_constraints() -> None:
    import copy

    schema = copy.deepcopy(pydantic_to_tool_schema(_Root))
    _force_strict(schema)
    violations = _walk_strict(schema)
    assert violations == [], violations


def test_force_strict_all_properties_required_even_with_defaults() -> None:
    import copy

    class WithDefault(BaseModel):
        model_config = ConfigDict(extra="forbid")
        keep: str
        also: list[str] = Field(default_factory=list)

    schema = copy.deepcopy(pydantic_to_tool_schema(WithDefault))
    _force_strict(schema)
    assert set(schema["required"]) == {"keep", "also"}


def test_default_metadata_stripped_from_schema() -> None:
    class WithDefaults(BaseModel):
        name: str
        note: str = Field(default="hello")
        count: int = Field(default=0)

    schema = pydantic_to_tool_schema(WithDefaults)
    assert "default" not in schema["properties"]["note"]
    assert "default" not in schema["properties"]["count"]
    assert set(schema["required"]) == {"name"}


def test_property_named_default_preserved() -> None:
    class HasDefaultProp(BaseModel):
        default: str = Field(description="The default value")
        name: str

    schema = pydantic_to_tool_schema(HasDefaultProp)
    assert "default" in schema["properties"]
    assert schema["properties"]["default"]["type"] == "string"


def test_backward_compat_alias() -> None:
    from agentm.core.lib import pydantic_to_openai_tool_schema

    assert pydantic_to_openai_tool_schema is pydantic_to_tool_schema
