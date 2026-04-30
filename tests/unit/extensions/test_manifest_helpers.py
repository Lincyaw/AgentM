"""Tests for ``agentm.extensions`` public surface (manifest + tag parser)."""

from __future__ import annotations

import pytest

from agentm.extensions import (
    VALID_REGISTER_KINDS,
    ExtensionManifest,
    parse_register_tag,
)


def test_register_tag_parses_kind_and_id() -> None:
    """Well-formed tag splits into (kind, id) without surprises."""

    assert parse_register_tag("tool:read") == ("tool", "read")
    assert parse_register_tag("event:tool_call") == ("event", "tool_call")


def test_register_tag_rejects_unknown_kind() -> None:
    """Unknown kinds must fail loudly so typos in MANIFEST surface in
    review, not in production behavior."""

    with pytest.raises(ValueError, match="kind"):
        parse_register_tag("widget:foo")


@pytest.mark.parametrize("malformed", ["read", ":read", "tool:", ":"])
def test_register_tag_rejects_malformed(malformed: str) -> None:
    """Tags missing kind or id are rejected with a stable error message."""

    with pytest.raises(ValueError):
        parse_register_tag(malformed)


def test_valid_register_kinds_is_immutable_snapshot() -> None:
    """The kind set is part of the contract — code shouldn't mutate it."""

    assert isinstance(VALID_REGISTER_KINDS, frozenset)
    assert {"tool", "event", "command", "provider", "renderer"} <= (
        VALID_REGISTER_KINDS
    )


def test_manifest_defaults_for_optional_fields() -> None:
    """``requires`` / ``conflicts`` default to empty tuples; schema to None."""

    m = ExtensionManifest(
        name="x", description="y", registers=("tool:x",)
    )
    assert m.config_schema is None
    assert m.requires == ()
    assert m.conflicts == ()
