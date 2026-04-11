"""Focused regression tests for `_to_messages` input conversion."""
from __future__ import annotations

from typing import Any

from agentm.builder import _to_messages


def test_messages_field_is_preferred_and_copied() -> None:
    original = [{"role": "human", "content": "hello"}]
    result = _to_messages({"messages": original, "task_description": "ignored"})
    assert result == original
    result.append({"role": "system", "content": "extra"})
    assert len(original) == 1


def test_task_description_string_becomes_single_human_message() -> None:
    result = _to_messages({"task_description": "do something"})
    assert result == [{"role": "human", "content": "do something"}]


def test_task_description_list_legacy_uses_first_element() -> None:
    result = _to_messages({"task_description": ["first", "second"]})
    assert result == [{"role": "human", "content": "first"}]


def test_empty_input_variants_return_empty_list() -> None:
    assert _to_messages({}) == []
    assert _to_messages({"messages": []}) == []
    assert _to_messages({"task_description": ""}) == []
    assert _to_messages({"task_description": []}) == []
