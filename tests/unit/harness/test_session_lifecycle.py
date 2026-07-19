"""Session lifecycle: cold-start, prompt, shutdown, and provider-missing guard."""

from __future__ import annotations

import importlib
import uuid
from pathlib import Path

import pytest

from agentm.core.abi import AssistantMessage
from agentm.core.abi.extension import ExtensionLoadError
from agentm.core.abi.session_api import AgentSessionConfig
from agentm.core.runtime.resource_loader import InMemoryResourceLoader
from agentm.core.runtime.session import AgentSession


_PROVIDER_SOURCE = '''
from __future__ import annotations
from collections.abc import AsyncIterator
from typing import Any
from agentm.core.abi import AssistantMessage, MessageEnd, Model, TextContent, ProviderConfig

class _Stream:
    def __call__(self, *, messages: list[Any], model: Model, tools: list[Any], system: str | None = None, signal: Any = None, thinking: str = "off") -> AsyncIterator[Any]:
        return self._iter()
    async def _iter(self) -> AsyncIterator[Any]:
        yield MessageEnd(message=AssistantMessage(role="assistant", content=[TextContent(type="text", text="hello from stub")], timestamp=1.0, stop_reason="end_turn"))

def install(api, config):
    api.register_provider("stub", ProviderConfig(stream_fn=_Stream(), model=Model(id="stub", provider="stub", context_window=4096, max_output_tokens=512), name="stub"))
'''


async def _make_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    include_provider: bool = True,
) -> AgentSession:
    pkg = f"lifecycle_{uuid.uuid4().hex[:8]}"
    pkg_dir = tmp_path / pkg
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("", encoding="utf-8")
    (pkg_dir / "provider.py").write_text(_PROVIDER_SOURCE, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()

    extensions = [("agentm.extensions.builtin.operations", {})]
    provider = (f"{pkg}.provider", {}) if include_provider else None

    return await AgentSession.create(AgentSessionConfig(
        cwd=str(tmp_path),
        extensions=extensions,
        provider=provider,
        resource_loader=InMemoryResourceLoader(),
    ))


@pytest.mark.asyncio
async def test_session_cold_start_and_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = await _make_session(tmp_path, monkeypatch)
    try:
        messages = await session.prompt("hello")
        assert len(messages) >= 2
        has_assistant = any(isinstance(m, AssistantMessage) for m in messages)
        assert has_assistant
    finally:
        await session.shutdown()


@pytest.mark.asyncio
async def test_session_shutdown_is_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = await _make_session(tmp_path, monkeypatch)
    await session.prompt("hi")
    await session.shutdown()
    # second shutdown must not raise
    await session.shutdown()


@pytest.mark.asyncio
async def test_session_without_provider_fails_fast(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ExtensionLoadError, match="(?i)provider"):
        await _make_session(tmp_path, monkeypatch, include_provider=False)
