"""Loader-level smoke tests for the ``<scenario>:<variant>`` syntax.

The loader resolves bare scenario names to
``contrib/scenarios/<name>/manifest.yaml``. Variant manifests live alongside:
``rca:baseline`` -> ``contrib/scenarios/rca/manifest.baseline.yaml``. This
keeps multiple scenario flavors in one directory (sharing prompts, atoms,
and the package) instead of duplicating the directory.
"""

from __future__ import annotations

import pytest

from agentm.extensions.loader import ScenarioLoadError, load_scenario_with_meta


def test_bare_scenario_name_loads_default_manifest() -> None:
    atoms, _ = load_scenario_with_meta("rca")
    assert len(atoms) > 0


def test_variant_syntax_loads_suffixed_manifest() -> None:
    atoms, meta = load_scenario_with_meta("rca:baseline")
    assert len(atoms) > 0
    # The baseline manifest declares its own task_class so production traces
    # are filterable separately from the multi-agent rca scenario.
    assert meta.get("task_class") == "rca_baseline"


def test_subdirectory_syntax_still_works() -> None:
    """Pre-existing convention: ``rca/tuner`` -> ``rca/tuner/manifest.yaml``."""
    atoms, meta = load_scenario_with_meta("rca/tuner")
    assert len(atoms) > 0
    assert meta.get("task_class") == "rca_baseline_tuner"


@pytest.mark.parametrize(
    "name", ["rca:", ":baseline", "rca:nonexistent_variant"]
)
def test_invalid_variant_names_raise(name: str) -> None:
    with pytest.raises(ScenarioLoadError):
        load_scenario_with_meta(name)
