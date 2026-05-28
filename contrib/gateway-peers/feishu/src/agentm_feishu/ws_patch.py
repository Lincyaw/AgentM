"""Workaround for the upstream lark-oapi WS event-loop bug.

``lark_oapi.ws.client`` captures ``asyncio.get_event_loop()`` at
module-import time. When the module is imported from inside an
already-running event loop (our case — ``agentm-feishu`` runs the
adapter under ``asyncio.run``), that captured reference *is* the main
loop; ``WSClient.start`` then calls ``loop.run_until_complete(...)``
from a background thread against the still-running main loop and
raises ``RuntimeError: This event loop is already running``. Replacing
the module-level reference with a fresh dedicated loop makes
``run_until_complete`` legal again — the WS client is the only
consumer of that module global.

TODO: drop this once an upstream fix lands. File the issue on
https://github.com/larksuite/oapi-sdk-python with a reproducer and
delete this module plus its caller in ``cli._arun``.
"""

from __future__ import annotations

import asyncio


def apply_ws_patch() -> None:
    """Replace ``lark_oapi.ws.client.loop`` with a fresh event loop.

    Idempotent only in the sense that calling it twice in the same
    process just installs a second fresh loop; the WS client picks up
    whatever is current at ``WSClient.start`` time. Call once at CLI
    startup, before constructing the lark ``Client``.
    """
    import lark_oapi.ws.client as _ws_client  # noqa: PLC0415 — import-on-demand

    _ws_client.loop = asyncio.new_event_loop()


__all__ = ["apply_ws_patch"]
