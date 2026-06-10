"""Registry-based OTel dispatch for kernel events.

Replaces the monkey-patch pattern where ``abi/events.py`` overwrote
``Event.to_otel`` on concrete subclasses. The registry lives in ``lib/``
(not ``abi/`` or ``runtime/``) so both layers can reach it without creating
a circular dependency.
"""

from __future__ import annotations

from typing import Any, Callable

OtelTranslator = Callable[[Any, Any], None]

_TRANSLATORS: dict[type, OtelTranslator] = {}


def register_otel(event_type: type, fn: OtelTranslator) -> None:
    _TRANSLATORS[event_type] = fn


def dispatch_otel(event: Any, telemetry: Any) -> None:
    fn = _TRANSLATORS.get(type(event))
    if fn is not None:
        fn(event, telemetry)
