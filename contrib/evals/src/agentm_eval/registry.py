"""Benchmark registry — discovers and registers eval benchmarks."""

from __future__ import annotations

from typing import Any

import typer

_BENCHMARKS: dict[str, tuple[str, callable]] = {}


def register(name: str, description: str, factory: callable) -> None:
    _BENCHMARKS[name] = (description, factory)


def get_benchmark(name: str) -> Any:
    if name not in _BENCHMARKS:
        available = ", ".join(sorted(_BENCHMARKS))
        raise KeyError(f"Unknown benchmark {name!r}. Available: {available}")
    _, factory = _BENCHMARKS[name]
    return factory()


def list_benchmarks() -> list[tuple[str, str]]:
    return [(name, desc) for name, (desc, _) in sorted(_BENCHMARKS.items())]


def get_cli(name: str) -> typer.Typer:
    bench = get_benchmark(name)
    return bench.create_cli()


def discover() -> None:
    """Import all benchmark modules to trigger registration."""
    import importlib

    modules = [
        "agentm_eval.benchmarks.ale",
        "agentm_eval.benchmarks.sandbox",
        "agentm_eval.benchmarks.aftraj",
        "agentm_eval.benchmarks.tau2",
        "agentm_eval.benchmarks.rescue_window",
        "agentm_eval.benchmarks.index_eval",
        "agentm_eval.benchmarks.auditor_eval",
    ]
    for mod in modules:
        try:
            importlib.import_module(mod)
        except ImportError:
            pass
