"""Permission-demo extension: passively logs every tool_call.

Designed for the smoke test: confirms that an extension can subscribe to
``tool_call`` and observe the call without blocking it.
"""

from __future__ import annotations

from typing import Any


CALLS_OBSERVED: list[dict[str, Any]] = []


def install(api: Any, config: dict[str, Any]) -> None:
    def _on_tool_call(event: Any) -> None:
        CALLS_OBSERVED.append(
            {
                "tool_call_id": event.tool_call_id,
                "tool_name": event.tool_name,
                "args": dict(event.args),
            }
        )

    api.on("tool_call", _on_tool_call)
