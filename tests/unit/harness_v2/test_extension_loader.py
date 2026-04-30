"""Tests for ``agentm.harness.extension.load_extension``."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from agentm.harness.extension import ExtensionLoadError, load_extension


class _StubAPI:
    """Minimal stand-in for ExtensionAPI; the loader doesn't introspect it."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []


def test_sync_install_called_with_api_and_config() -> None:
    """A module exporting ``install`` is invoked with the provided api and
    config; the loader returns ``None`` to signal "no awaitable required"."""

    from tests.unit.harness_v2._fixtures import sample_ext

    sample_ext.INSTALLED.clear()
    api = _StubAPI()
    result = load_extension(
        "tests.unit.harness_v2._fixtures.sample_ext",
        api,  # type: ignore[arg-type]
        {"k": "v"},
    )

    assert result is None
    assert len(sample_ext.INSTALLED) == 1
    record = sample_ext.INSTALLED[0]
    assert record["api"] is api
    assert record["config"] == {"k": "v"}


def test_async_install_returns_awaitable_that_runs_install() -> None:
    """An async ``install`` returns a coroutine; awaiting it executes the
    body."""

    from tests.unit.harness_v2._fixtures import sample_async_ext

    sample_async_ext.INSTALLED.clear()
    api = _StubAPI()
    result = load_extension(
        "tests.unit.harness_v2._fixtures.sample_async_ext",
        api,  # type: ignore[arg-type]
        {"x": 1},
    )

    assert result is not None
    assert asyncio.iscoroutine(result)

    asyncio.run(result)  # type: ignore[arg-type]
    assert len(sample_async_ext.INSTALLED) == 1
    assert sample_async_ext.INSTALLED[0]["config"] == {"x": 1}


def test_missing_module_raises_extension_load_error() -> None:
    """Importing a non-existent module path surfaces as ``ExtensionLoadError``
    naming the offending path."""

    api = _StubAPI()
    with pytest.raises(ExtensionLoadError) as exc_info:
        load_extension(
            "no.such.module.exists.anywhere",
            api,  # type: ignore[arg-type]
            {},
        )
    assert "no.such.module.exists.anywhere" in str(exc_info.value)
    assert exc_info.value.cause is not None


def test_module_without_install_raises_extension_load_error() -> None:
    """Importable module without an ``install`` symbol must error clearly."""

    api = _StubAPI()
    with pytest.raises(ExtensionLoadError) as exc_info:
        load_extension(
            "tests.unit.harness_v2._fixtures.no_install_ext",
            api,  # type: ignore[arg-type]
            {},
        )
    assert "no_install_ext" in str(exc_info.value)
    assert "install" in str(exc_info.value)


def test_install_raising_is_wrapped() -> None:
    """An exception thrown by ``install`` is wrapped in ``ExtensionLoadError``
    while preserving the cause."""

    from tests.unit.harness_v2._fixtures.raising_ext import BoomError

    api = _StubAPI()
    with pytest.raises(ExtensionLoadError) as exc_info:
        load_extension(
            "tests.unit.harness_v2._fixtures.raising_ext",
            api,  # type: ignore[arg-type]
            {},
        )
    assert isinstance(exc_info.value.cause, BoomError)
