"""Shared serialization utilities for AgentM.

Provides safe access to langgraph's JsonPlusSerializer with unified error handling.
"""

from __future__ import annotations

from typing import Any, Tuple

try:
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    _SERIALIZER = JsonPlusSerializer()
except ImportError:
    _SERIALIZER = None  # type: ignore[assignment]


def deserialize_typed(data: Tuple[str, bytes]) -> Any:
    """Deserialize typed data using JsonPlusSerializer.

    Args:
        data: A tuple of (type_name, serialized_bytes) from checkpoint storage.

    Returns:
        The deserialized Python object.

    Raises:
        RuntimeError: If JsonPlusSerializer is not available (langgraph not installed).
        Exception: If deserialization fails (propagated from serde).
    """
    if _SERIALIZER is None:
        raise RuntimeError("JsonPlusSerializer not available — langgraph not installed")
    return _SERIALIZER.loads_typed(data)
