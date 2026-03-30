"""Tests for builder._to_messages() — input conversion logic.

_to_messages() is the single point where AgentInput dicts are converted
into a list[Message] before being passed to the agent loop. Bugs here
would silently drop user instructions or crash the loop.
"""
from __future__ import annotations

from typing import Any

from agentm.builder import _to_messages


class TestToMessagesFromMessagesField:
    """When input_data contains a 'messages' key, use it directly."""

    def test_should_return_messages_list_when_present(self) -> None:
        """Messages field present and non-empty -> returned as-is.

        Bug prevented: messages field ignored, falling through to
        task_description or returning [].
        """
        input_data: dict[str, Any] = {
            "messages": [{"role": "human", "content": "hello"}],
        }
        result = _to_messages(input_data)

        assert result == [{"role": "human", "content": "hello"}]

    def test_should_return_copy_not_original_reference(self) -> None:
        """Returned list is a copy, so mutations don't affect the original.

        Bug prevented: Caller's input dict silently mutated when the loop
        appends system/tool messages to the returned list.
        """
        original_messages = [{"role": "human", "content": "hi"}]
        input_data: dict[str, Any] = {"messages": original_messages}

        result = _to_messages(input_data)
        result.append({"role": "system", "content": "extra"})

        assert len(original_messages) == 1  # original unchanged

    def test_should_handle_multiple_messages(self) -> None:
        """Multiple messages are all preserved in order.

        Bug prevented: Only first message returned, or messages reversed.
        """
        msgs = [
            {"role": "human", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "human", "content": "second"},
        ]
        input_data: dict[str, Any] = {"messages": msgs}
        result = _to_messages(input_data)

        assert len(result) == 3
        assert result[0]["content"] == "first"
        assert result[2]["content"] == "second"


class TestToMessagesFromTaskDescription:
    """When input_data has no 'messages' but has 'task_description'."""

    def test_should_wrap_task_description_as_human_message(self) -> None:
        """task_description string -> wrapped as [{"role": "human", "content": ...}].

        Bug prevented: task_description passed as raw string to loop.stream()
        instead of message dict, causing type errors.
        """
        input_data: dict[str, Any] = {"task_description": "do something"}
        result = _to_messages(input_data)

        assert len(result) == 1
        assert result[0] == {"role": "human", "content": "do something"}

    def test_should_handle_task_description_as_list_legacy(self) -> None:
        """Legacy edge: task_description is a list -> extracts first element.

        Bug prevented: TypeError when str() is called on a list, producing
        ugly "[item1, item2]" as content, or IndexError on empty list.
        """
        input_data: dict[str, Any] = {"task_description": ["first task", "second task"]}
        result = _to_messages(input_data)

        assert len(result) == 1
        assert result[0]["content"] == "first task"

    def test_should_handle_empty_list_task_description(self) -> None:
        """Legacy edge: task_description is an empty list -> returns [].

        Bug prevented: IndexError accessing task[0] on empty list.
        """
        input_data: dict[str, Any] = {"task_description": []}
        result = _to_messages(input_data)

        assert result == []


class TestToMessagesMessagesPriority:
    """When both 'messages' and 'task_description' are present."""

    def test_should_prefer_messages_over_task_description(self) -> None:
        """messages field wins when both are present.

        Bug prevented: task_description overrides explicit messages, causing
        the caller's carefully constructed conversation history to be lost.
        """
        input_data: dict[str, Any] = {
            "messages": [{"role": "human", "content": "from messages"}],
            "task_description": "from task",
        }
        result = _to_messages(input_data)

        assert len(result) == 1
        assert result[0]["content"] == "from messages"


class TestToMessagesEmptyInput:
    """When neither 'messages' nor 'task_description' is set."""

    def test_should_return_empty_list_when_neither_set(self) -> None:
        """No messages, no task -> returns [].

        Bug prevented: KeyError or None returned instead of empty list,
        crashing downstream code that iterates the result.
        """
        result = _to_messages({})
        assert result == []

    def test_should_return_empty_list_for_empty_messages(self) -> None:
        """messages=[] (empty list, falsy) -> falls through to task_description
        or returns [].

        Bug prevented: Empty list treated as truthy, returned as-is when
        task_description could provide a fallback.
        """
        input_data: dict[str, Any] = {"messages": []}
        result = _to_messages(input_data)
        assert result == []

    def test_should_return_empty_list_for_empty_task_description(self) -> None:
        """task_description="" (empty string) -> returns [].

        Bug prevented: Empty string wrapped as a human message, sending
        a blank instruction to the LLM.
        """
        input_data: dict[str, Any] = {"task_description": ""}
        result = _to_messages(input_data)
        assert result == []
