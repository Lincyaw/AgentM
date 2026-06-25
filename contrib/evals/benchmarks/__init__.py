"""Benchmark adapter registry."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .harbor import HarborAdapter
from .swebench import SWEBenchAdapter
from .terminal_bench import TerminalBenchAdapter

if TYPE_CHECKING:
    from .base import BenchAdapter

ADAPTERS: dict[str, type | object] = {
    "terminal-bench": TerminalBenchAdapter,
    "tb1": TerminalBenchAdapter,  # legacy alias
    "harbor": HarborAdapter,
    "tb2": HarborAdapter,  # legacy alias
    "swebench-verified": lambda: SWEBenchAdapter(variant="verified"),
    "swebench-pro": lambda: SWEBenchAdapter(variant="pro"),
}

BENCH_CHOICES = list(ADAPTERS.keys())


def get_adapter(name: str) -> BenchAdapter:
    """Instantiate and return the adapter for the given bench name."""
    entry = ADAPTERS.get(name)
    if entry is None:
        msg = f"Unknown bench format: {name!r}. Choose from: {', '.join(ADAPTERS)}"
        raise ValueError(msg)
    if callable(entry) and isinstance(entry, type):
        return entry()  # type: ignore[return-value]
    if callable(entry):
        return entry()  # type: ignore[return-value]
    return entry  # type: ignore[return-value]
