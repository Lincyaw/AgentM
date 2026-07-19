"""Behavior contracts for portable authoring artifacts."""

from __future__ import annotations

from types import MappingProxyType

import pytest

from agentm.authoring import AuthoringArtifact, FrontmatterDocument


def test_frontmatter_round_trips_typed_yaml_scalars() -> None:
    document = FrontmatterDocument(
        metadata={
            "title": "value: with colon",
            "enabled": True,
            "attempts": 2,
        },
        body="artifact body",
    )

    restored = FrontmatterDocument.parse(document.render())

    assert restored.metadata == document.metadata
    assert restored.body == document.body
    assert isinstance(restored.metadata, MappingProxyType)


def test_authoring_rejects_ambiguous_metadata_and_escaping_paths() -> None:
    with pytest.raises(ValueError, match="closing delimiter"):
        FrontmatterDocument.parse("---\ntitle: incomplete")
    with pytest.raises(TypeError, match="must be a scalar"):
        FrontmatterDocument.parse("---\nnested:\n  key: value\n---\nbody")
    with pytest.raises(ValueError, match="stay relative"):
        AuthoringArtifact(path="../outside.md", body="body")
