"""Shared cooperative-shutdown defaults.

Single source of truth for the grace window every atom/session bus drain
gives still-running asyncio tasks before forcing cancellation. Lives in
``core.lib`` so atoms (``background_exec``, ``sub_agent``, ``monitor``) and
the substrate (``Session.shutdown``) import the same number — drifting
copies of ``5.0`` was a recurring boundary-review finding.
"""

from __future__ import annotations

from typing import Final

# Seconds the shutdown drain waits for tasks to finish cooperatively before
# cancelling them. Atoms expose this as a ``shutdown_grace_seconds`` config
# knob (default = this constant); the substrate session driver uses it
# directly (no config knob — substrate-private).
DEFAULT_SHUTDOWN_GRACE_SECONDS: Final[float] = 5.0


__all__ = ["DEFAULT_SHUTDOWN_GRACE_SECONDS"]
