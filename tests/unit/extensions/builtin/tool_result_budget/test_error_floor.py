"""Bug guard: an aggressive ``max_chars`` would otherwise chop block-reason
strings like "Tool call blocked: <reason>" down to a single character,
hiding from the model why a tool call was denied.
"""

from __future__ import annotations

from typing import Any, cast

from agentm.core.kernel import TextContent, ToolResult, ToolResultEvent

from agentm.extensions.builtin import tool_result_budget


class _API:
    def __init__(self) -> None:
        self.handlers: dict[str, Any] = {}

    def on(self, channel: str, handler: Any) -> Any:
        self.handlers[channel] = handler
        return lambda: None


def _event(text: str, *, is_error: bool) -> ToolResultEvent:
    return ToolResultEvent(
        tool_call_id="c1",
        tool_name="any_tool",
        result=ToolResult(
            content=[TextContent(type="text", text=text)],
            is_error=is_error,
        ),
    )


def test_error_floor_preserves_short_block_reasons_but_not_normal_results() -> None:
    api = _API()
    tool_result_budget.install(cast(Any, api), {"max_chars": 5})

    # An error payload longer than max_chars but well under the floor must
    # pass through untouched so the block reason remains intact.
    error_text = "Tool call blocked: permission denied"
    assert len(error_text) > 5
    assert len(error_text) < tool_result_budget._ERROR_FLOOR

    error_result = api.handlers["tool_result"](_event(error_text, is_error=True))
    assert error_result is None  # no truncation applied

    # A non-error payload of the same length still gets truncated to max_chars,
    # confirming the floor only relaxes the rule for error results.
    normal_result = api.handlers["tool_result"](_event(error_text, is_error=False))
    assert isinstance(normal_result, ToolResult)
    assert isinstance(normal_result.content[0], TextContent)
    assert normal_result.content[0].text == error_text[:5]
