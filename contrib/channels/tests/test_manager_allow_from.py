"""Empty ``allow_from`` must fail-fast at manager construction.

A channel that is ``enabled: true`` with ``allow_from: []`` denies
everyone — combined with online status, that looks like a silently
broken bot. Refuse to start so the operator knows immediately.
"""

from __future__ import annotations

import pytest

from agentm_channels.bus import MessageBus
from agentm_channels.manager import ChannelManager


def test_empty_allow_from_raises_systemexit() -> None:
    bus = MessageBus()
    with pytest.raises(SystemExit) as excinfo:
        ChannelManager(
            {"stub": {"enabled": True, "allow_from": []}}, bus
        )
    assert "stub" in str(excinfo.value)
    assert "allow_from" in str(excinfo.value)


def test_non_list_allow_from_raises_systemexit() -> None:
    bus = MessageBus()
    with pytest.raises(SystemExit):
        ChannelManager(
            {"stub": {"enabled": True, "allow_from": "u1"}}, bus  # type: ignore[dict-item]
        )




