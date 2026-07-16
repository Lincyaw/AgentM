"""Lazy public boundary for the AgentM command-line presenter."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agentm.env import autoload_dotenv

if TYPE_CHECKING:
    from typer import Typer

    app: Typer


def main() -> None:
    """Load and run the root command tree."""

    from agentm.cli.main import main as run

    run()


def __getattr__(name: str) -> Any:
    if name == "app":
        from agentm.cli.main import app

        return app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["app", "autoload_dotenv", "main"]
