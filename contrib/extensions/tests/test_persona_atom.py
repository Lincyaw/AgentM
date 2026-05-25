"""Coverage for the ``persona`` contrib atom.

The atom's value is a single contract: user-authored identity files on
disk become the cache-stable head of the system prompt, in order, while a
bare workspace (no files) injects nothing. If that breaks, the chatbot
scenario silently loses its character or — worse — a sandbox with no
persona files starts emitting an empty/garbled block.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from contrib.extensions import persona as persona_atom
from agentm.core.abi.events import BeforeAgentStartEvent


class _DiskFileOps:
    async def read_file(self, path: str) -> bytes:
        return Path(path).read_bytes()

    async def access(self, path: str) -> bool:
        return Path(path).exists()


class _DiskWriter:
    """ResourceWriter stub writing cwd-relative paths under ``root``."""

    def __init__(self, root: Path) -> None:
        self._root = root

    async def write(self, path: str, content: bytes, *, rationale: str) -> Any:
        del rationale
        target = self._root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        return SimpleNamespace(error=None)


class _FakeAPI:
    def __init__(self, root: Path) -> None:
        self.cwd = str(root)
        self.handlers: dict[str, Any] = {}
        self._file_ops = _DiskFileOps()
        self._writer = _DiskWriter(root)

    def get_operations(self) -> Any:
        return SimpleNamespace(file=self._file_ops)

    def get_resource_writer(self) -> Any:
        return self._writer

    def on(self, channel: str, handler: Any) -> None:
        self.handlers[channel] = handler


async def _fire(api: _FakeAPI, system: str) -> str | None:
    handler = api.handlers[BeforeAgentStartEvent.CHANNEL]
    event = BeforeAgentStartEvent(messages=[], system=system)
    ret = await handler(event)
    # The handler must also *return* its replacement (order-independence),
    # and that return must agree with the mutated event.
    if event.system and event.system != system:
        assert ret == {"system": event.system}
    return event.system


@pytest.mark.asyncio
async def test_present_files_compose_in_order_above_base_prompt(
    tmp_path: Path,
) -> None:
    (tmp_path / "SOUL.md").write_text("Warm, terse, never sycophantic.\n")
    (tmp_path / "USER.md").write_text("Boxi — prefers Chinese, dislikes fluff.\n")
    # IDENTITY.md intentionally absent — must be skipped without a gap.

    api = _FakeAPI(tmp_path)
    persona_atom.install(cast(Any, api), {})

    out = await _fire(api, "base prompt")
    assert out is not None
    assert out.startswith("# Persona")
    # Both present files surface, each under its derived heading.
    assert "## Soul" in out and "Warm, terse" in out
    assert "## User" in out and "prefers Chinese" in out
    # Absent file leaves no heading.
    assert "## Identity" not in out
    # Configured order is honoured: SOUL before USER.
    assert out.index("## Soul") < out.index("## User")
    # The base prompt is preserved, after the persona block.
    assert out.rstrip().endswith("base prompt")








@pytest.mark.asyncio
async def test_seeding_happens_once(tmp_path: Path) -> None:
    api = _FakeAPI(tmp_path)
    persona_atom.install(cast(Any, api), {"defaults": {"SOUL.md": "v1\n"}})

    await _fire(api, "")
    # Simulate the agent editing its own persona after the first turn.
    (tmp_path / "SOUL.md").write_text("# Soul\n\nv2 — edited by the agent.\n")
    out = await _fire(api, "")
    # Re-seeding must NOT overwrite the edit back to the preset; the next
    # turn re-reads the edited file (this is the "auto-reload" contract).
    assert out is not None
    assert "v2 — edited by the agent." in out
    assert "v1" not in out
