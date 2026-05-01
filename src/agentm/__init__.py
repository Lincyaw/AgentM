"""AgentM package root.

Phase 2.5 reduces the package to the v2 kernel + harness + extension catalog.
The CLI used to live in ``agentm.cli`` (deleted with the legacy tree); the
``agentm`` console-script now points at :func:`agentm.cli.main` defined in
``cli.py``.
"""

from __future__ import annotations


def main() -> None:
    """Console-script entry point — defers to ``agentm.cli.main``."""

    from agentm.cli import main as cli_main

    cli_main()
