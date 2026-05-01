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


# ---------------------------------------------------------------------------
# manifest-schema task: defaults for the new fields (api_version, affects, tier)
#
# The manifest grew three new fields in plan/2026-05-01-self-mod-mvp R4:
#   - api_version: int = 1   — the §11.4.9 versioned-API gate
#   - affects:     tuple[str, ...] = () — Phase 2 indexer hint
#   - tier:        int = 1   — §11.4.10 reload-cost classification
#
# All three are defaulted so the existing ~25 atoms keep building without
# source changes. These tests pin the defaults so a future dataclass
# refactor cannot silently drop them.
# ---------------------------------------------------------------------------


def test_extension_manifest_default_api_version_is_1() -> None:
    """``api_version`` defaults to 1 — matches ``extension_api.current`` in
    ``core-manifest.yaml``. Without this default every existing atom would
    need to gain an explicit ``api_version=1`` line on the same PR.
    """

    m = ExtensionManifest(name="x", description="y", registers=())
    assert m.api_version == 1


def test_extension_manifest_default_affects_is_empty_tuple() -> None:
    """``affects`` defaults to an empty tuple — matches the pattern used by
    ``registers`` / ``requires`` (frozen, hashable, value-typed). MVP atoms
    do not need to declare what they affect; Phase 2 will require it when
    ``compare()`` consumes the structure.
    """

    m = ExtensionManifest(name="x", description="y", registers=())
    assert m.affects == ()
    # Pin the type — a list default would defeat the frozen/hashable
    # invariant the dataclass was designed around.
    assert isinstance(m.affects, tuple)


def test_extension_manifest_default_tier_is_1() -> None:
    """``tier`` defaults to 1 — only the five named atoms in
    ``core-manifest.yaml::reload.tier_2_atoms`` opt in to ``tier=2``. The
    default keeps the catalog mechanically backward compatible.
    """

    m = ExtensionManifest(name="x", description="y", registers=())
    assert m.tier == 1
