"""Shared test fixtures for AgentM test suite."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

os.environ.setdefault("OTEL_EXPORTER_OTLP_ENDPOINT", "")


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Tear down OTel global providers so atexit never retries against 4317."""
    try:
        from agentm.core.observability.otel_export import shutdown_process_telemetry

        shutdown_process_telemetry()
    except Exception as exc:  # noqa: BLE001
        # Best-effort teardown; a failure here must not mask test results.
        print(f"conftest: telemetry shutdown failed: {exc}")

# The harness configures the constitution-boundary manifest path on session
# start; for unit tests that exercise the loader directly (or tests that
# spin up sessions in tmpdirs and so leave the ContextVar pointing
# somewhere transient) we re-pin the manifest path before every test.
_REPO_ROOT = Path(__file__).parents[1]
_REPO_MANIFEST = _REPO_ROOT / "core-manifest.yaml"


if _REPO_MANIFEST.exists():
    from agentm.core._internal.catalog import manifest as _manifest_mod

    _manifest_mod.configure_manifest_path(_REPO_MANIFEST)


@pytest.fixture(autouse=True)
def _reset_repo_manifest_path() -> Iterator[None]:
    """Ensure each test starts and ends with the repo-root manifest bound.

    The manifest path lives on a :class:`ContextVar`; tests that rebind it
    via :func:`override_manifest_path` automatically restore on context
    exit, and tests that drive a session through ``AgentSession.create``
    leave the ContextVar pointing at a tmpdir manifest. Re-pin before and
    after every test so cases stay isolated.
    """

    if _REPO_MANIFEST.exists():
        from agentm.core._internal.catalog import manifest as _manifest_mod

        _manifest_mod.configure_manifest_path(_REPO_MANIFEST)
    yield
    if _REPO_MANIFEST.exists():
        from agentm.core._internal.catalog import manifest as _manifest_mod

        _manifest_mod.configure_manifest_path(_REPO_MANIFEST)
