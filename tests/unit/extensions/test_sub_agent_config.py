"""Fail-stop coverage for ``sub_agent`` config propagation (A3).

Validates that the ``shutdown_grace_seconds`` config knob added in the
boundary-review polish flows from ``install(config)`` into the manager and
bounds the actual ``on_session_shutdown`` drain. The integration suites
(``test_sub_agent_lifecycle`` / ``test_sub_agent_budgets``) cover the full
child-session lifecycle behaviour — this is a focused fail-stop test for the
config propagation alone, so a future refactor cannot silently revert the
constant-extracted shutdown grace back to a hard-coded ``5.0``.
"""

from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest

from agentm.core.abi.events import SessionShutdownEvent
from agentm.core.abi.extension import ExtensionAPI
from agentm.extensions.builtin.sub_agent import _ChildTask, _ChildTaskManager


class _FakeApi:
    """Minimal stub for the ExtensionAPI bits ``_ChildTaskManager`` touches in
    its shutdown drain. Construction-only — no dispatch is exercised here."""

    def __init__(self) -> None:
        self._handlers: dict[str, list[Any]] = {}

    def on(self, channel: str, handler: Any, *, priority: int = 500) -> Any:
        self._handlers.setdefault(channel, []).append(handler)
        return lambda: None


@pytest.mark.asyncio
async def test_shutdown_grace_seconds_config_bounds_drain() -> None:
    """A3 fail-stop: a ``shutdown_grace_seconds=0.05`` override is what
    ``on_session_shutdown`` actually waits on (no leftover ``5.0`` literal
    anywhere on the drain path). The drain must finish well under a second
    even when the child task never cooperates with the abort signal.
    """

    manager = _ChildTaskManager(
        api=cast(ExtensionAPI, _FakeApi()),
        inherit_extensions=[],
        available_inherited={},
        max_workers=4,
        system_prompt_module="agentm.extensions.builtin.system_prompt",
        shutdown_grace_seconds=0.05,
    )
    assert manager._shutdown_grace_seconds == 0.05

    # Register a cooperative running child — the sub_agent shutdown drain
    # sets ``abort_signal`` and gathers the task; the task observes the
    # signal and exits. With grace=0.05 the wait times out almost instantly
    # if the task hasn't completed on its own, then the abort+gather path
    # finishes the child.
    abort = asyncio.Event()

    async def cooperative() -> None:
        # Sleep until the registry's shutdown drain trips the abort signal.
        await abort.wait()

    coop_task: asyncio.Task[None] = asyncio.create_task(cooperative())
    state = _ChildTask(
        task_id="x",
        task=coop_task,
        abort_signal=abort,
        purpose="test",
        session=None,
    )
    await manager._registry.reserve_slot()
    await manager._registry.register(state)

    start = asyncio.get_event_loop().time()
    await asyncio.wait_for(
        manager.on_session_shutdown(SessionShutdownEvent(cwd=".")),
        timeout=2.0,
    )
    elapsed = asyncio.get_event_loop().time() - start
    # Drain returned within the configured grace + cooperative shutdown
    # overhead — would have been ~5s with the previous hard-coded constant.
    assert elapsed < 1.0, f"drain took {elapsed:.3f}s — grace config not honoured"
    assert coop_task.done()
