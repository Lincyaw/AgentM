"""Fail-stop coverage for the ``memory`` builtin atom.

Single round-trip test: save a memory, read it back (access counter
increments), and confirm MEMORY.md index is regenerated. If this breaks,
the evolution layer can no longer trust persisted memories to survive
session restarts — that's the load-bearing position this guards.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from agentm.extensions.builtin import memory as memory_atom
from agentm.harness.events import BeforeAgentStartEvent


class _DiskFileOps:
    """Real-disk FileOperations stub so we exercise the writer/reader pair
    end-to-end without mocking the storage primitives the atom relies on."""

    def __init__(self, root: Path) -> None:
        self._root = root

    async def read_file(self, path: str) -> bytes:
        return Path(path).read_bytes()

    async def access(self, path: str) -> bool:
        return Path(path).exists()

    async def is_dir(self, path: str) -> bool:
        return Path(path).is_dir()

    async def list_dir(self, path: str) -> list[str]:
        p = Path(path)
        if not p.is_dir():
            return []
        return sorted(entry.name for entry in p.iterdir())


class _DiskWriter:
    """ResourceWriter stub that writes to a real tmp dir. The atom passes
    cwd-relative paths; we anchor them under ``root``."""

    def __init__(self, root: Path) -> None:
        self._root = root

    async def write(self, path: str, content: bytes, *, rationale: str) -> Any:
        del rationale
        target = self._root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        return SimpleNamespace(error=None)

    async def delete(self, path: str, *, rationale: str) -> Any:
        del rationale
        target = self._root / path
        if target.exists():
            target.unlink()
        return SimpleNamespace(error=None)


class _FakeAPI:
    def __init__(self, root: Path) -> None:
        self.cwd = str(root)
        self.tools: list[Any] = []
        self.handlers: dict[str, Any] = {}
        self._file_ops = _DiskFileOps(root)
        self._writer = _DiskWriter(root)

    def register_tool(self, tool: Any) -> None:
        self.tools.append(tool)

    def get_operations(self) -> Any:
        return SimpleNamespace(file=self._file_ops)

    def get_resource_writer(self) -> Any:
        return self._writer

    def on(self, channel: str, handler: Any) -> None:
        self.handlers[channel] = handler


@pytest.mark.asyncio
async def test_memory_roundtrip_persists_index_and_access_counter(
    tmp_path: Path,
) -> None:
    api = _FakeAPI(tmp_path)
    memory_atom.install(cast(Any, api), {})

    by_name = {tool.name: tool for tool in api.tools}
    save = by_name["memory_save"]
    read = by_name["memory_read"]
    search = by_name["memory_search"]

    save_result = await save.execute(
        {
            "type": "feedback",
            "name": "no_mocks",
            "description": "integration tests hit the real db",
            "content": "We got burned mocking the migration last quarter.",
        }
    )
    assert save_result.is_error is False, save_result.content[0].text

    mem_dir = tmp_path / ".agentm" / "memory"
    saved_file = mem_dir / "feedback_no_mocks.md"
    assert saved_file.exists(), "memory file must land on disk"
    body = saved_file.read_text(encoding="utf-8")
    assert "name: no_mocks" in body
    assert "type: feedback" in body
    assert "We got burned" in body

    index = (mem_dir / "MEMORY.md").read_text(encoding="utf-8")
    assert "[feedback/no_mocks]" in index
    assert "integration tests hit the real db" in index

    # Read twice; access counter must reach 2.
    read_result = await read.execute({"name": "no_mocks"})
    assert read_result.is_error is False
    assert "We got burned" in read_result.content[0].text

    second = await read.execute({"name": "no_mocks"})
    assert second.is_error is False

    stats_raw = (mem_dir / "access_stats.json").read_text(encoding="utf-8")
    stats = json.loads(stats_raw)
    assert stats["no_mocks"]["count"] == 2
    assert "last_access" in stats["no_mocks"]

    # MEMORY.md content must surface through the BeforeAgentStartEvent hook.
    handler = api.handlers[BeforeAgentStartEvent.CHANNEL]
    event = BeforeAgentStartEvent(messages=[], system="base prompt")
    await handler(event)
    assert event.system is not None
    assert "<memory_index>" in event.system
    assert "[feedback/no_mocks]" in event.system
    assert event.system.endswith("base prompt")

    # Search finds the description by substring.
    found = await search.execute({"query": "real db"})
    assert found.is_error is False
    assert "no_mocks" in found.content[0].text


@pytest.mark.asyncio
async def test_memory_save_rejects_invalid_name(tmp_path: Path) -> None:
    api = _FakeAPI(tmp_path)
    memory_atom.install(cast(Any, api), {})
    save = next(tool for tool in api.tools if tool.name == "memory_save")

    result = await save.execute(
        {
            "type": "user",
            "name": "bad name with spaces",
            "description": "irrelevant",
            "content": "x",
        }
    )
    assert result.is_error is True
    assert "invalid name" in result.content[0].text
