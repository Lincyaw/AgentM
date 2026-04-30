"""Fixture: an extension whose ``install`` raises."""

from __future__ import annotations

from typing import Any


class BoomError(RuntimeError):
    pass


def install(api: Any, config: dict[str, Any]) -> None:
    raise BoomError("kapow")
