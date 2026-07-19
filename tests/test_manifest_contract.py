"""Behavioral contract for extension manifest validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agentm.core.abi.manifest import ExtensionManifest


def test_extension_manifest_rejects_unknown_contract_fields() -> None:
    with pytest.raises(ValidationError, match="api_version"):
        ExtensionManifest.model_validate(
            {
                "name": "invalid_manifest",
                "description": "Unknown fields must not look effective.",
                "api_version": 1,
            }
        )
