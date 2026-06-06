"""Fail-stop: the ``before_submission`` fork selector must recognise the
RCA orchestrator's *live* submission tool.

The selector originally matched only ``submit_investigation`` — a name no RCA
tool has ever registered (the orchestrator submits with
``submit_final_report``). With no test, that drift silently degraded
``before_submission`` to "fork before the last message" for every real
trajectory. This locks the selector to the live submission vocabulary.
"""

from __future__ import annotations

from agentm.core.abi.messages import (
    AssistantMessage,
    TextContent,
    ToolCallBlock,
    UserMessage,
)

from agentm_rca.eval.replay_fork.strategy import (
    _SUBMISSION_TOOL_NAMES,
    before_submission,
)


def _assistant_submitting(tool_name: str) -> AssistantMessage:
    return AssistantMessage(
        role="assistant",
        content=[
            ToolCallBlock(
                type="tool_call",
                id="c1",
                name=tool_name,
                arguments={"root_causes": []},
            )
        ],
        timestamp=0.0,
    )


def _user(text: str) -> UserMessage:
    return UserMessage(
        role="user", content=[TextContent(type="text", text=text)], timestamp=0.0
    )


def test_before_submission_forks_at_live_submit_final_report() -> None:
    # The orchestrator's real terminator is submit_final_report; the selector
    # must return that assistant turn's index, not the generic last-message
    # fallback.
    messages = [
        _user("investigate"),
        _assistant_submitting("submit_final_report"),
        _user("anything after"),
    ]
    assert before_submission(messages) == 1


def test_before_submission_still_matches_legacy_name() -> None:
    # Old recordings used submit_investigation; keep them resolving.
    messages = [_user("go"), _assistant_submitting("submit_investigation")]
    assert before_submission(messages) == 1


def test_submission_vocab_includes_live_and_legacy_names() -> None:
    assert "submit_final_report" in _SUBMISSION_TOOL_NAMES
    assert "submit_investigation" in _SUBMISSION_TOOL_NAMES
