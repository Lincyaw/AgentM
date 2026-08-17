from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from agentm.control import SessionControlServer, send_interrupt
from agentm.core.abi.trigger import TriggerPriority


class _FakeSession:
    session_id = "control-session"

    def __init__(self) -> None:
        self.prompts: list[tuple[str, TriggerPriority, str | None, str]] = []

    async def prompt(
        self,
        text: str,
        *,
        priority: TriggerPriority,
        origin: str | None,
        mode: str,
    ) -> object:
        self.prompts.append((text, priority, origin, mode))
        return object()


@pytest.mark.asyncio
async def test_session_control_delivers_immediate_interrupt() -> None:
    with tempfile.TemporaryDirectory(dir="/tmp") as directory:
        inbox = Path(directory)
        session = _FakeSession()
        server = SessionControlServer(session, inbox_root=inbox)
        await server.start()
        try:
            await send_interrupt(
                session.session_id,
                "reconsider the task",
                inbox_root=inbox,
            )
        finally:
            await server.stop()

        assert session.prompts == [("reconsider the task", "now", "human", "interrupt")]
        assert not server.path.exists()


def test_the_control_socket_follows_agentm_home(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Where this installation keeps its state is one rule, written once.

    ``$AGENTM_HOME`` is the runtime state directory: config, scenarios, skills,
    trajectories and observability all move when it is set. The control socket
    did not -- it was the seventh copy of a rule written out six times, and the
    one that was written without it. Every socket went to ``~/.agentm/inbox``
    however the deployment was configured, so a container that mounts
    ``$AGENTM_HOME`` and leaves ``$HOME`` read-only could not open one at all.
    """

    from agentm.control import control_socket_path

    monkeypatch.setenv("AGENTM_HOME", str(tmp_path / "state"))
    assert control_socket_path("abc") == tmp_path / "state" / "inbox" / "abc.sock"

    monkeypatch.delenv("AGENTM_HOME", raising=False)
    assert control_socket_path("abc") == Path.home() / ".agentm" / "inbox" / "abc.sock"

    # An explicit root still wins, because a caller that named one meant it.
    assert control_socket_path("abc", tmp_path) == tmp_path / "abc.sock"
