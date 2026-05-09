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
    UserMessage,
)
from agentm.core.abi.compaction import (
    CompactionPrompts,
    CompactionSettings,
)
from agentm.core.abi.session import (
    ENTRY_MATERIALIZERS,
    ENTRY_TYPE_BRANCH_SUMMARY,
    ENTRY_TYPE_COMPACTION,
    ENTRY_TYPE_MESSAGE,
    SessionEntry,
    branch_summary_entry,
    compaction_entry,
    message_entry,
)
from agentm.core._internal.compaction import (
    compact,
    create_file_ops,
    extract_file_ops_from_message,
    get_message_from_entry,
    prepare_compaction,
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


def test_extract_file_ops_without_registry_is_empty() -> None:
    """Graceful degradation: no registry, no file ops, no crash."""

    file_ops = create_file_ops()
    message = _assistant_with_tool_call("read", "/tmp/a.txt")

    extract_file_ops_from_message(message, file_ops, None)

    assert file_ops.read == set()


def test_extract_file_ops_ignores_tools_without_file_op_metadata() -> None:
    file_ops = create_file_ops()
    message = _assistant_with_tool_call("custom_tool", "/tmp/a.txt")
    tools = [_make_function_tool("custom_tool", file_op=None)]

    extract_file_ops_from_message(message, file_ops, tools)

    assert file_ops.read == set()
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


def test_get_message_from_entry_uses_registry(isolated_materializers: dict) -> None:
    isolated_materializers.clear()
    materializer = _RecordingMaterializer()
    isolated_materializers["custom"] = materializer

    entry = SessionEntry(
        type="custom",
        id="e1",
        parent_id=None,
        timestamp=0.0,
        payload=None,
    )
    message = get_message_from_entry(entry)

    assert isinstance(message, AssistantMessage)
    assert materializer.calls == ["custom"]


def test_get_message_from_entry_returns_none_when_unregistered(
    isolated_materializers: dict,
) -> None:
    isolated_materializers.clear()
    entry = SessionEntry(
        type=ENTRY_TYPE_MESSAGE,
        id="e2",
        parent_id=None,
        timestamp=0.0,
        payload=UserMessage(
            role="user",
            content=[TextContent(type="text", text="hi")],
            timestamp=0.0,
        ),
    )
    assert get_message_from_entry(entry) is None


def test_compaction_prompts_atom_populates_registry(
    isolated_materializers: dict,
) -> None:
    """End-to-end: installing ``compaction_prompts`` registers all three
    canonical materializers so ``get_message_from_entry`` works for the
    kernel-defined entry types."""

    isolated_materializers.clear()
    api = _StubExtensionAPI()
    from agentm.extensions.builtin import compaction_prompts

    compaction_prompts.install(api, {})

    assert set(isolated_materializers) >= {
        ENTRY_TYPE_MESSAGE,
        ENTRY_TYPE_BRANCH_SUMMARY,
        ENTRY_TYPE_COMPACTION,
    }

    user_entry = message_entry(
        UserMessage(
            role="user",
            content=[TextContent(type="text", text="hi")],
            timestamp=0.0,
        ),
        parent_id=None,
    )
    assert isinstance(get_message_from_entry(user_entry), UserMessage)

    summary_entry = branch_summary_entry("did things", parent_id=None)
    msg = get_message_from_entry(summary_entry)
    assert isinstance(msg, AssistantMessage)
    block = msg.content[0]
    assert isinstance(block, TextContent)
    assert "did things" in block.text

    compaction_entry_value = compaction_entry({"summary": "done"}, parent_id=None)
    msg = get_message_from_entry(compaction_entry_value)
    assert isinstance(msg, AssistantMessage)
    block = msg.content[0]
    assert isinstance(block, TextContent)
    assert block.text == "done"


# --- 3. Compaction with and without the atom ------------------------------


@pytest.mark.asyncio
async def test_compact_with_prompts_atom_threads_system_prompt(
    isolated_materializers: dict,
) -> None:
    isolated_materializers.clear()
    api = _StubExtensionAPI()
    from agentm.extensions.builtin import compaction_prompts

    compaction_prompts.install(api, {})

    user_entry = message_entry(
        UserMessage(
            role="user",
            content=[TextContent(type="text", text="explain X")],
            timestamp=0.0,
        ),
        parent_id=None,
    )
    assistant_entry = message_entry(
        AssistantMessage(
            role="assistant",
            content=[TextContent(type="text", text="here is X")],
            timestamp=1.0,
            stop_reason="end_turn",
        ),
        parent_id=user_entry.id,
    )

    settings = CompactionSettings(
        enabled=True, reserve_tokens=512, keep_recent_tokens=1
    )
    preparation = prepare_compaction([user_entry, assistant_entry], settings)
    assert preparation is not None

    summarization_body = api.prompt_templates.get_prompt("compaction.summarization")
    assert isinstance(summarization_body, str) and summarization_body

    prompts = CompactionPrompts(
        summarization_system=api.prompt_templates.get_prompt(
            "compaction.summarization_system"
        )
        or "",
        update_summarization=api.prompt_templates.get_prompt(
            "compaction.update_summarization"
        )
        or "",
        turn_prefix_summarization=api.prompt_templates.get_prompt(
            "compaction.turn_prefix_summarization"
        )
        or "",
    )

    result = await compact(
        preparation,
        _stub_summarizer,
        summarization_body,
        custom_instructions=None,
        prompts=prompts,
    )
    # The stub summarizer echoes the system prompt — verifying it really
    # threads through the engine and into the summarizer.
    assert prompts.summarization_system[:20] in result.summary


@pytest.mark.asyncio
async def test_compact_without_prompts_atom_falls_back_quietly() -> None:
    """The kernel must not crash when the prompts atom is absent — it
    accepts ``prompts=None`` and threads empty strings through.
    """

    user_entry = message_entry(
        UserMessage(
            role="user",
            content=[TextContent(type="text", text="hi")],
            timestamp=0.0,
        ),
        parent_id=None,
    )
    assistant_entry = message_entry(
        AssistantMessage(
            role="assistant",
            content=[TextContent(type="text", text="ok")],
            timestamp=1.0,
            stop_reason="end_turn",
        ),
        parent_id=user_entry.id,
    )

    settings = CompactionSettings(
        enabled=True, reserve_tokens=512, keep_recent_tokens=1
    )
    preparation = prepare_compaction([user_entry, assistant_entry], settings)
    assert preparation is not None

    result = await compact(
        preparation,
        _stub_summarizer,
        "",
        custom_instructions=None,
        prompts=None,
    )
    # Empty system prompt threaded through.
    assert "[system:]" in result.summary


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


@dataclass
class _StubExtensionAPI:
    prompt_templates: Any = field(default_factory=_StubPromptTemplatesService)


# Ensure async tests run.
@pytest.fixture
def event_loop() -> Any:  # pragma: no cover
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
