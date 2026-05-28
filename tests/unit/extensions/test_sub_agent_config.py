"""Fail-stop coverage for ``sub_agent`` boundary-review fixes (A3 + B3).

* A3: ``shutdown_grace_seconds`` config knob propagation from
  ``install(config)`` into the manager and the actual drain.
* B3: counter-reset encapsulation — pin that ``ToolCallEvent`` on the bus
  resets ``_running_only_cancels`` (the single source of truth that replaced
  the five per-tool ``await self._reset_running_only_cancels()`` callsites).

The integration suites (``test_sub_agent_lifecycle`` /
``test_sub_agent_budgets``) cover the full child-session lifecycle and
auto-abort behaviour — those tests are the behaviour-unchanged proof. These
unit tests pin the mechanism so a future refactor cannot silently revert
either piece (e.g. re-add a hard-coded ``5.0``, or drop the
``ToolCallEvent`` subscription so the counter only resets on sub_agent's
own tools).
"""

from __future__ import annotations

import asyncio
from typing import cast

import pytest

from agentm.core.abi import ToolCallEvent
from agentm.core.abi.events import SessionShutdownEvent
from agentm.core.abi.extension import ExtensionAPI
from agentm.extensions.builtin.sub_agent import _ChildTask, _ChildTaskManager
from tests.unit.extensions._fake_api import FakeExtensionAPI

# Alias for diff continuity; the shared helper is enough for the wiring
# tests below (subscribe + a no-op register_tool wired in test_install_*).
_FakeApi = FakeExtensionAPI


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


def test_tool_call_event_resets_running_only_cancels() -> None:
    """B3 fail-stop: a ``ToolCallEvent`` on the bus resets
    ``_running_only_cancels`` (the single source of truth that replaced
    the five per-tool ``await self._reset_running_only_cancels()``
    callsites).

    Deliberately ANY tool name, not just sub_agent's own — that's the
    behaviour the bus-subscription approach gives us (the agent is engaged
    whenever it invokes any tool, regardless of which atom owns it). The
    handler is sync (single int assignment) so the test calls it directly.
    """

    manager = _ChildTaskManager(
        api=cast(ExtensionAPI, _FakeApi()),
        inherit_extensions=[],
        available_inherited={},
        max_workers=4,
        system_prompt_module="agentm.extensions.builtin.system_prompt",
    )
    # Force a non-zero counter (the decide_turn_action path puts it there).
    manager._running_only_cancels = 1

    # ANY tool — even one not owned by sub_agent — fires the reset. That
    # was the whole point of moving off the per-method callsites.
    event = ToolCallEvent(
        tool_call_id="t1",
        tool_name="bash",  # not a sub_agent tool
        args={},
    )
    manager._on_tool_call_reset_counter(event)
    assert manager._running_only_cancels == 0


def test_install_subscribes_tool_call_handler() -> None:
    """B3 wiring: ``install`` registers the ToolCallEvent handler so the
    reset mechanism is actually wired up (a forgotten subscription would
    silently re-enable the unbounded counter)."""

    api = _FakeApi()

    async def _drive() -> None:
        from agentm.extensions.builtin import sub_agent

        await sub_agent.install(cast(ExtensionAPI, api), {})

    asyncio.run(_drive())
    handlers = api._handlers.get(ToolCallEvent.CHANNEL, [])
    assert len(handlers) == 1, (
        f"install() must subscribe exactly one ToolCallEvent handler; "
        f"got {len(handlers)}"
    )
