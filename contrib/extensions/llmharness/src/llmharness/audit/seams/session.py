"""Session-orchestration utilities shared by the live adapter and the
offline replay runner.

Terminal-tool argument scraping now lives in
:func:`llmharness.child_collect.terminal_tool_arguments` (shared
across the live and embedded paths); this module keeps the
extractor-state binding and the best-effort child-shutdown helper used
by the embedded host-driver.
"""

from __future__ import annotations

import logging
from typing import Any

from ...agents.extractor.state import ExtractionState
from ..toolkit.atom_constants import (
    EXTRACTOR_STATE_SERVICE_KEY,
    EXTRACTOR_TOOLS_MODULE,
)

_logger = logging.getLogger(__name__)


async def safe_shutdown(session: Any) -> None:
    """Swallow shutdown errors — audit children are best-effort cleanup."""
    try:
        shutdown = getattr(session, "shutdown", None)
        if shutdown is not None:
            await shutdown()
    except Exception:
        _logger.debug("audit-child shutdown failed", exc_info=True)


def bind_extractor_state(
    base_extensions: list[tuple[str, dict[str, Any]]],
    *,
    state: ExtractionState,
) -> list[tuple[str, dict[str, Any]]]:
    """Inject a fresh ``ExtractionState`` into the extractor extensions.

    Returns a copy; the input list and its config dicts are never mutated.
    Called once per extractor firing, both in live spawning and in replay.
    The new-turn window is delivered as the child's user message (see
    the adapter / replay runner) rather than substituted into the system
    prompt, so this helper has nothing to do with prompt text.

    Note. v18's ``witness_retry_budget`` knob is gone in v19; each
    upsert gets per-edit validation feedback so there is no batch to
    bounce back. This helper only injects the per-firing state.
    """
    out: list[tuple[str, dict[str, Any]]] = []
    for module, cfg in base_extensions:
        new_cfg = dict(cfg)
        if module == EXTRACTOR_TOOLS_MODULE:
            new_cfg["state"] = state
            new_cfg.setdefault(EXTRACTOR_STATE_SERVICE_KEY, state)
        out.append((module, new_cfg))
    return out
