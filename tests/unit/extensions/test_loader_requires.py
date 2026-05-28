"""Fail-stop tests for ``sort_extensions_by_requires`` floor-atom handling.

The scenario requires-validator runs on the manifest's own extension list,
before the session factory injects floor atoms (compaction_prompts,
prompt_templates, slash_commands, system_prompt). An atom that ``requires`` a
floor atom must therefore validate WITHOUT the manifest redundantly listing
it — otherwise the validator raises, the factory falls back to an empty load,
and the session comes up with no Operations bundle. Regression guard for the
gateway-breaking bug where mounting ``llm_compaction`` (which requires the
``compaction_prompts`` floor atom) blew up scenario load.

A genuinely-missing non-floor dependency must still raise.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from agentm.extensions import ExtensionManifest
from agentm.extensions.loader import ScenarioLoadError, sort_extensions_by_requires


def _register(module_path: str, manifest: ExtensionManifest) -> str:
    mod = types.ModuleType(module_path)
    mod.MANIFEST = manifest  # type: ignore[attr-defined]

    def install(api: Any, config: Any) -> None:  # pragma: no cover - never called
        return None

    mod.install = install  # type: ignore[attr-defined]
    sys.modules[module_path] = mod
    return module_path


def test_requires_floor_atom_without_manifest_listing_is_ok() -> None:
    # ``llm_compaction`` requires the ``compaction_prompts`` floor atom plus
    # ``prompt_templates``. Listing only ``prompt_templates`` must not raise —
    # the factory auto-mounts ``compaction_prompts``.
    result = sort_extensions_by_requires(
        [
            ("agentm.extensions.builtin.prompt_templates", {}),
            ("agentm.extensions.builtin.llm_compaction", {}),
        ]
    )
    modules = [m for m, _ in result]
    assert "agentm.extensions.builtin.llm_compaction" in modules


def test_missing_non_floor_requires_still_raises() -> None:
    _register(
        "_test_requirer_atom",
        ExtensionManifest(
            name="_test_requirer",
            description="",
            registers=(),
            requires=("_test_absent_non_floor_dep",),
        ),
    )
    with pytest.raises(ScenarioLoadError):
        sort_extensions_by_requires([("_test_requirer_atom", {})])
