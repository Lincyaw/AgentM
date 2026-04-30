"""Async sample extension used by ``test_extension_loader``."""

from __future__ import annotations

from typing import Any


INSTALLED: list[dict[str, Any]] = []


async def install(api: Any, config: dict[str, Any]) -> None:
    INSTALLED.append({"api": api, "config": dict(config)})
