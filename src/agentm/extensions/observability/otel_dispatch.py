"""Registry-based OTel dispatch for session events.

This is part of the OTel observability implementation, not the SDK substrate.
The builtin observability atom calls :func:`dispatch_otel`; backend-specific
translator modules register handlers here.
"""

# code-health: ignore-file[AM022] -- dispatches heterogeneous events to OTel adapters

from __future__ import annotations

from collections.abc import Callable
from typing import Any

OtelTranslator = Callable[[Any, Any], None]

_TRANSLATORS: dict[type, OtelTranslator] = {}


def register_otel(event_type: type, fn: OtelTranslator) -> None:
    _TRANSLATORS[event_type] = fn


def dispatch_otel(event: Any, telemetry: Any) -> None:
    fn = _TRANSLATORS.get(type(event))
    if fn is not None:
        fn(event, telemetry)
