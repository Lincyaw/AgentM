"""AgentM package root — v2 kernel + harness + extension catalog.

The ``agentm`` console-script entry point delegates to :func:`agentm.cli.main`.
"""

from __future__ import annotations

import os


def _configure_native_logging_defaults() -> None:
    """Keep native dependencies quiet unless the operator opts in."""

    os.environ.setdefault("GRPC_VERBOSITY", "ERROR")


def main() -> None:
    """Console-script entry point — defers to ``agentm.cli.main``."""

    _configure_native_logging_defaults()

    from agentm.cli import main as cli_main

    cli_main()
