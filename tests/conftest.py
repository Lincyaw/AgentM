"""Shared test fixtures for the AgentM SDK test suite."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("OTEL_EXPORTER_OTLP_ENDPOINT", "")


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Tear down process-level OTel providers after tests."""
    del session, exitstatus
    try:
        from agentm.extensions.observability.otel_export import (
            shutdown_process_telemetry,
        )

        shutdown_process_telemetry()
    except Exception as exc:  # noqa: BLE001
        print(f"conftest: telemetry shutdown failed: {exc}")
