"""Issue #76 fail-stops: tool-metadata file_op routing, prompt registry,
entry-materializer registry. Covers the de-scenario contract in three
behavioural beats.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest

from agentm.core.abi import (
    AssistantMessage,
    FunctionTool,
    TextContent,
    ToolCallBlock,
    ToolResult,
)
from agentm.core.abi.session import (
    ENTRY_MATERIALIZERS,
    SessionEntry,
)
from agentm.extensions.builtin.llm_compaction import (
    create_file_ops,
    extract_file_ops_from_message,
)


async def _stub_summarizer(system: str, prompt: str, max_tokens: int) -> str:
    # Echo the system prompt + a marker so tests can verify the body the
    # engine threaded through actually reached the summarizer.
    return f"[system:{system}]\n[len:{len(prompt)}]"


def _make_function_tool(name: str, file_op: str | None) -> FunctionTool:
    async def _execute(args: dict[str, Any]) -> ToolResult:
        return ToolResult(content=[TextContent(type="text", text="ok")])

    metadata: dict[str, Any] = {}
    if file_op is not None:
        metadata["file_op"] = file_op
    return FunctionTool(
        name=name,
        description="",
        parameters={"type": "object"},
        fn=_execute,
        metadata=metadata,
    )


def _assistant_with_tool_call(name: str, path: str) -> AssistantMessage:
    return AssistantMessage(
        role="assistant",
        content=[
            ToolCallBlock(
                type="tool_call",
                id="call-1",
                name=name,
                arguments={"path": path},
            )
        ],
        timestamp=0.0,
        stop_reason="tool_use",
    )


# --- 1. Tool metadata drives file-op extraction ---------------------------


def test_extract_file_ops_uses_tool_metadata() -> None:
    file_ops = create_file_ops()
    message = _assistant_with_tool_call("vendor_read_x", "/tmp/a.txt")
    tools = [_make_function_tool("vendor_read_x", "read")]

    extract_file_ops_from_message(message, file_ops, tools)

    assert file_ops.read == {"/tmp/a.txt"}
    assert file_ops.written == set()
    assert file_ops.edited == set()




# --- 2. Entry-materializer registry ---------------------------------------


@dataclass(frozen=True)
class _RecordingMaterializer:
    calls: list[str] = field(default_factory=list)

    def to_message(self, entry: SessionEntry) -> AssistantMessage | None:
        self.calls.append(entry.type)
        return AssistantMessage(
            role="assistant",
            content=[TextContent(type="text", text=entry.type)],
            timestamp=entry.timestamp,
            stop_reason="end_turn",
        )


@pytest.fixture
def isolated_materializers() -> Any:
    """Snapshot and restore ``ENTRY_MATERIALIZERS`` around each test."""

    snapshot = dict(ENTRY_MATERIALIZERS)
    yield ENTRY_MATERIALIZERS
    ENTRY_MATERIALIZERS.clear()
    ENTRY_MATERIALIZERS.update(snapshot)








# --- 3. Compaction with and without the atom ------------------------------






# --- Stubs ---------------------------------------------------------------


class _StubPromptTemplatesService:
    def __init__(self) -> None:
        self._registry: dict[str, str] = {}

    def load_prompt_templates(self, **_: Any) -> list[Any]:  # pragma: no cover
        return []

    def expand_prompt_template(self, *_: Any, **__: Any) -> str | None:  # pragma: no cover
        return None

    def register_prompt(self, name: str, body: str) -> None:
        self._registry[name] = body

    def get_prompt(self, name: str) -> str | None:
        return self._registry.get(name)


class _StubExtensionAPI:
    def __init__(self) -> None:
        self._services: dict[str, Any] = {
            "prompt_templates": _StubPromptTemplatesService()
        }

    def set_service(self, name: str, obj: Any) -> None:
        self._services[name] = obj

    def get_service(self, name: str) -> Any:
        return self._services.get(name)

    @property
    def prompt_templates(self) -> Any:
        # Back-compat shim for tests that still read api.prompt_templates
        # directly to inspect the registered bodies.
        return self._services.get("prompt_templates")


# Ensure async tests run.
@pytest.fixture
def event_loop() -> Any:  # pragma: no cover
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
