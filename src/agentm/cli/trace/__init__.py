"""Lazy public boundary for the trace-query command group."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from typer import Typer

    app: Typer


def main() -> None:
    """Load and run the trace command tree."""

    from agentm.cli.trace.commands import main as run

    run()


def __getattr__(name: str) -> Any:
    if name == "app":
        from agentm.cli.trace.commands import app

        return app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["app", "main"]
