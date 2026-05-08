"""Scenario-local wrapper that lazily loads the RCA hypothesis tools."""

from __future__ import annotations

from typing import Any

from agentm.extensions import ExtensionManifest

MANIFEST = ExtensionManifest(
    name="hypothesis_tools",
    description="Lazy wrapper for the RCA shared-artifact hypothesis tools.",
    registers=(),
)


def install(api: Any, config: dict[str, Any]) -> Any:
    from agentm_rca.tools.hypothesis_tools import install as package_install

    return package_install(api, config)
