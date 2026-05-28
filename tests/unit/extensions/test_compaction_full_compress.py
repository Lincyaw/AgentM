"""Fail-stop tests for the full-compress compaction model + read_history.

Protected positions:
- ``enumerate_turns`` numbers turns 1-based, skips non-message entries, and
  stays stable across compactions — the summary's ``[Turn N]`` tags must keep
  pointing at the same content ``read_history`` later resolves.
- ``prepare_compaction`` is incremental: it folds only turns after the
  previous compaction's ``covered_through_turn`` and carries the prior summary
  forward (no O(n^2) re-summarization, no lost early context).
- After a compaction entry, ``build_session_context`` rebuilds the context as
  a single ``user(summary)`` message — a provider completion request must not
  end on an assistant turn, and nothing pre-compaction may leak back in.
- The original turns survive in ``get_branch`` so ``read_history`` can recover
  them verbatim.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from agentm.core.abi import (
    AssistantMessage,
    TextContent,
    UserMessage,
)
from agentm.core.abi.session import (
    ENTRY_MATERIALIZERS,
    compaction_entry,
    message_entry,
)
from agentm.core.lib import enumerate_turns
from agentm.core.runtime.session_manager import SessionManager
from agentm.extensions.builtin import compaction_prompts, read_history
from agentm.extensions.builtin.llm_compaction import (
    CompactionSettings,
    prepare_compaction,
)


def _user(text: str) -> UserMessage:
    return UserMessage(role="user", content=[TextContent(type="text", text=text)], timestamp=0.0)


def _assistant(text: str) -> AssistantMessage:
    return AssistantMessage(
        role="assistant",
        content=[TextContent(type="text", text=text)],
        timestamp=0.0,
        stop_reason="end_turn",
    )


def _first_text(message: Any) -> str:
    for block in message.content:
        if isinstance(block, TextContent):
            return block.text
    return ""


# --- 1. Turn numbering (shared by compaction tags + read_history) ----------


def test_enumerate_turns_numbers_and_skips_compaction() -> None:
    branch = [
        message_entry(_user("u1"), None),
        message_entry(_assistant("a1"), None),
        message_entry(_user("u2"), None),
        compaction_entry({"summary": "s", "covered_through_turn": 2}, None),
        message_entry(_user("u3"), None),
        message_entry(_assistant("a3"), None),
    ]
    turns = enumerate_turns(branch)

    assert [t.index for t in turns] == [1, 2, 3]
    assert len(turns[0].messages) == 2  # user + assistant
    assert _first_text(turns[0].messages[0]) == "u1"
    assert len(turns[1].messages) == 1  # user only
    # The compaction entry is skipped: u3 still starts turn 3.
    assert _first_text(turns[2].messages[0]) == "u3"


def test_turn_indices_are_stable_when_branch_grows() -> None:
    branch = [message_entry(_user("u1"), None), message_entry(_assistant("a1"), None)]
    assert [t.index for t in enumerate_turns(branch)] == [1]
    branch.append(message_entry(_user("u2"), None))
    turns = enumerate_turns(branch)
    assert [t.index for t in turns] == [1, 2]
    assert _first_text(turns[0].messages[0]) == "u1"  # turn 1 unchanged


# --- 2. Incremental chaining ----------------------------------------------


def test_prepare_compaction_folds_only_uncovered_turns() -> None:
    settings = CompactionSettings()

    first = [
        message_entry(_user("u1"), None),
        message_entry(_assistant("a1"), None),
        message_entry(_user("u2"), None),
        message_entry(_assistant("a2"), None),
    ]
    prep1 = prepare_compaction(first, settings)
    assert prep1 is not None
    assert [t.index for t in prep1.turns_to_summarize] == [1, 2]
    assert prep1.covered_through_turn == 2
    assert prep1.previous_summary is None

    chained = first + [
        compaction_entry({"summary": "PREV", "covered_through_turn": 2}, None),
        message_entry(_user("u3"), None),
        message_entry(_assistant("a3"), None),
    ]
    prep2 = prepare_compaction(chained, settings)
    assert prep2 is not None
    assert [t.index for t in prep2.turns_to_summarize] == [3]
    assert prep2.covered_through_turn == 3
    assert prep2.previous_summary == "PREV"


def test_prepare_compaction_returns_none_when_nothing_new() -> None:
    settings = CompactionSettings()
    branch = [
        message_entry(_user("u1"), None),
        message_entry(_assistant("a1"), None),
        compaction_entry({"summary": "PREV", "covered_through_turn": 1}, None),
    ]
    assert prepare_compaction(branch, settings) is None


# --- 3. Full-compress rebuild ---------------------------------------------


@pytest.fixture
def materializers() -> Any:
    """Register the default entry materializers and restore afterwards."""

    snapshot = dict(ENTRY_MATERIALIZERS)
    compaction_prompts.install(_PromptApi(), {})  # type: ignore[arg-type]
    yield
    ENTRY_MATERIALIZERS.clear()
    ENTRY_MATERIALIZERS.update(snapshot)


def test_full_compress_rebuilds_to_single_user_summary(materializers: Any) -> None:
    mgr = SessionManager.in_memory(cwd="")
    mgr.new_session(id="s1")
    mgr.append_message(_user("hello 1"))
    mgr.append_message(_assistant("reply 1"))
    mgr.append_message(_user("hello 2"))
    mgr.append_message(_assistant("reply 2"))

    mgr.append_custom_entry("compaction", {"summary": "CHECKPOINT", "covered_through_turn": 2})

    msgs = mgr.get_messages()
    assert len(msgs) == 1
    assert isinstance(msgs[0], UserMessage)  # not assistant — valid trailing prompt
    assert _first_text(msgs[0]) == "CHECKPOINT"

    # The raw turns survive for read_history.
    turns = enumerate_turns(mgr.get_branch())
    assert [t.index for t in turns] == [1, 2]


# --- 4. read_history recovers a turn verbatim ------------------------------


def test_read_history_returns_turn_content() -> None:
    branch = [
        message_entry(_user("first question"), None),
        message_entry(_assistant("first answer"), None),
        message_entry(_user("second question"), None),
        message_entry(_assistant("second answer"), None),
    ]
    api = _ToolApi(branch)
    read_history.install(api, {})  # type: ignore[arg-type]

    res = asyncio.run(api.tool.fn({"start": 2}))
    assert not res.is_error
    text = _first_text(res)
    assert "Turn 2" in text
    assert "second question" in text
    assert "second answer" in text
    assert "first question" not in text

    out_of_range = asyncio.run(api.tool.fn({"start": 9}))
    assert out_of_range.is_error


# --- stubs -----------------------------------------------------------------


class _PromptRegistry:
    def __init__(self) -> None:
        self._d: dict[str, str] = {}

    def register_prompt(self, name: str, body: str) -> None:
        self._d[name] = body

    def get_prompt(self, name: str) -> str | None:
        return self._d.get(name)


class _PromptApi:
    def __init__(self) -> None:
        self._svc: dict[str, Any] = {"prompt_templates": _PromptRegistry()}

    def get_service(self, name: str) -> Any:
        return self._svc.get(name)


class _ToolSession:
    def __init__(self, branch: list[Any]) -> None:
        self._branch = branch

    def get_branch(self) -> list[Any]:
        return self._branch


class _ToolApi:
    def __init__(self, branch: list[Any]) -> None:
        self.session = _ToolSession(branch)
        self.tool: Any = None

    def register_tool(self, tool: Any) -> None:
        self.tool = tool
