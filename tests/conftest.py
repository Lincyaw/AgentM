"""Shared test fixtures for AgentM test suite."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

# The harness configures the constitution-boundary manifest path on session
# start; for unit tests that exercise the loader directly (or tests that
# spin up sessions in tmpdirs and so leave the module-global pointing
# somewhere transient) we re-pin the manifest path before every test.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_REPO_MANIFEST = _REPO_ROOT / "core-manifest.yaml"


if _REPO_MANIFEST.exists():
    from agentm.core._internal.catalog import manifest as _manifest_mod

    _manifest_mod._MANIFEST_PATH = _REPO_MANIFEST


@pytest.fixture(autouse=True)
def _reset_repo_manifest_path() -> Iterator[None]:
    """Ensure each test starts with the repo-root ``core-manifest.yaml``.

    Tests that monkeypatch the path get their changes restored by
    ``MonkeyPatch.undo`` (because they use ``pytest.MonkeyPatch.setattr``);
    sessions started under :func:`AgentSession.create` mutate the global
    directly via ``configure_manifest_path`` and need this safety net.
    """

    if _REPO_MANIFEST.exists():
        from agentm.core._internal.catalog import manifest as _manifest_mod

        _manifest_mod._MANIFEST_PATH = _REPO_MANIFEST
        _manifest_mod.reload_manifest()
    yield
    if _REPO_MANIFEST.exists():
        from agentm.core._internal.catalog import manifest as _manifest_mod

        _manifest_mod._MANIFEST_PATH = _REPO_MANIFEST
        _manifest_mod.reload_manifest()
