"""Smoke tests for the ``--format textual`` frontend.

We do NOT spin up a real Textual app here (it grabs the terminal,
which fights pytest's stdout capture). Instead we verify the import
graph is healthy, the CSS file ships, and the entrypoint is wired —
enough to catch packaging / refactor regressions without flaky TUI
runs.

Full interactive testing is operator-side (`agentm-terminal --format
textual` against a live gateway).
"""

from __future__ import annotations

import importlib
from pathlib import Path


def test_textual_module_imports_cleanly() -> None:
    mod = importlib.import_module("agentm_terminal.ui.textual")
    assert hasattr(mod, "run_textual"), "run_textual entrypoint missing"


def test_css_file_ships_next_to_module() -> None:
    import agentm_terminal.ui.textual as mod

    css_path = Path(mod.__file__).parent / "textual_app.tcss"
    assert css_path.is_file(), f"CSS not found at {css_path}"
    body = css_path.read_text()
    # Spot-check: the three load-bearing widget classes have rules.
    assert "UserTurn" in body
    assert "AssistantTextBlock" in body
    assert "ApprovalBlock" in body


def test_cli_accepts_textual_format() -> None:
    from agentm_terminal.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args(["--connect", "unix:///tmp/x.sock", "--format", "textual"])
    assert args.format == "textual"
