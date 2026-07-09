"""Benchmark adapter registry — discovers and registers eval adapters."""

from __future__ import annotations

from typing import Any

import typer

_ADAPTERS: dict[str, tuple[str, callable]] = {}


def register(name: str, description: str, factory: callable) -> None:
    _ADAPTERS[name] = (description, factory)


def get_adapter(name: str) -> Any:
    if name not in _ADAPTERS:
        available = ", ".join(sorted(_ADAPTERS))
        raise KeyError(f"Unknown benchmark {name!r}. Available: {available}")
    _, factory = _ADAPTERS[name]
    return factory()


def list_adapters() -> list[tuple[str, str]]:
    return [(name, desc) for name, (desc, _) in sorted(_ADAPTERS.items())]


def get_cli(name: str) -> typer.Typer:
    adapter = get_adapter(name)
    return adapter.create_cli()


def discover() -> None:
    """Import all adapter modules to trigger registration."""
    import importlib

    modules = [
        "agentm_eval.adapters.sandbox",
        "agentm_eval.adapters.aftraj_auditor",
        "agentm_eval.adapters.aftraj_grounding",
        "agentm_eval.adapters.tau2",
        "agentm_eval.adapters.telbench",
        "agentm_eval.adapters.rescue_window_adapter",
    ]
    for mod in modules:
        try:
            importlib.import_module(mod)
        except ImportError:
            pass
