"""Benchmark adapters.

Importing this package registers the adapters it ships, so a bench name on the
command line resolves without the caller naming a module.
"""

from __future__ import annotations

from . import senior_swe

__all__ = ["senior_swe"]
