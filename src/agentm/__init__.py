"""AgentM package root — v2 kernel + harness + extension catalog.

The ``agentm`` console-script entry point delegates to :func:`agentm.cli.main`.
"""

from __future__ import annotations


def main() -> None:
    """Console-script entry point — defers to ``agentm.cli.main``."""

    from agentm.cli import main as cli_main

    cli_main()
