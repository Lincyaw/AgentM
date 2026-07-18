"""Shared helper for forwarding child session trajectories onto the parent wire."""

from __future__ import annotations

from typing import Any

from loguru import logger

from agentm.core.abi import ExtensionAPI
from agentm.core.abi.roles import WIRE_CHILD_FORWARDER_SERVICE


def forward_child_to_wire(api: ExtensionAPI, child: Any) -> None:
    """Fan a spawned child's trajectory onto the parent wire.

    No-op when running outside the gateway (no ``wire_driver``, so no
    ``child_wire_forwarder`` service) — the child still runs, its
    trajectory just isn't streamed to a chat peer.
    """
    forwarder = api.get_service(WIRE_CHILD_FORWARDER_SERVICE)
    if forwarder is None:
        return
    try:
        forwarder(child)
    except Exception as exc:  # noqa: BLE001
        logger.debug("child wire forwarding failed: {}", exc)
