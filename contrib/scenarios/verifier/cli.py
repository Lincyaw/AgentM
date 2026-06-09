#!/usr/bin/env python3
"""Verifier CLI — fault propagation verification.

Thin entry point that adds eval/ to sys.path and delegates to the
audit_case_propagate module.

Usage:
    uv run python contrib/scenarios/verifier/cli.py run <case_dir>
    uv run python contrib/scenarios/verifier/cli.py batch <dataset_dir> --run-dir ...
    uv run python contrib/scenarios/verifier/cli.py judge <case_dir> --run-dir ...
    uv run python contrib/scenarios/verifier/cli.py diff <dataset_dir> --run-dir ...
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "eval"))

from audit_case_propagate import app  # noqa: E402

if __name__ == "__main__":
    app()
