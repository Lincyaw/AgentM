"""Domain-specific scenarios built on the AgentM SDK.

Each scenario sub-package exposes a ``register()`` function that
populates the SDK registries (state, strategy, answer schemas,
output schemas) with its domain-specific types.

Call ``discover()`` once at application startup to auto-register
all built-in scenarios.  This is the *only* place that knows which
scenarios exist — the SDK core never imports from ``scenarios/``
directly.
"""

from __future__ import annotations

_discovered = False


def discover() -> None:
    """Import and register all built-in scenarios.

    Safe to call multiple times — only executes once.
    New scenarios only need to add their ``register()`` call here.
    """
    global _discovered
    if _discovered:
        return
    _discovered = True

    from agentm.scenarios.rca import register as register_rca
    from agentm.scenarios.memory_extraction import register as register_mem

    register_rca()
    register_mem()
