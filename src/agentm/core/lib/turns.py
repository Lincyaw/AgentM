"""Turn enumeration over a session branch.

A *turn* begins at a :class:`UserMessage` and includes the assistant and
tool-result messages that follow it until the next user message. Non-message
entries (compaction, branch_summary) are skipped.

This is the single source of turn numbering shared by two atoms that must
agree: the ``llm_compaction`` engine tags its summary with ``[Turn N]``
markers, and the ``read_history`` tool resolves a turn index back to its
original messages. forbids atom-to-atom imports, so the shared numbering
lives here in ``core.lib`` where both atoms reach it as a pure utility.

Indices are 1-based and **stable across compactions**: the session tree only
ever appends (compaction adds an entry, it never removes the original message
entries), so a turn's index never shifts once assigned.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentm.core.abi.messages import (
    AgentMessage,
    AssistantMessage,
    ToolResultMessage,
    UserMessage,
)
from agentm.core.abi.session import ENTRY_TYPE_MESSAGE, SessionEntry


@dataclass(frozen=True, slots=True)
class Turn:
    """One conversation turn: a user message plus its assistant/tool replies."""

    index: int  # 1-based, stable across compactions
    messages: list[AgentMessage]


def enumerate_turns(branch: list[SessionEntry]) -> list[Turn]:
    """Group a session branch into turns.

    Message entries carry their :class:`AgentMessage` directly as ``payload``,
    so no materializer is needed. Compaction / branch-summary entries are
    skipped. A new turn starts at every :class:`UserMessage` (and at the first
    message, defensively, in case a branch does not open with a user message).
    """

    groups: list[list[AgentMessage]] = []
    for entry in branch:
        if entry.type != ENTRY_TYPE_MESSAGE:
            continue
        payload = entry.payload
        if not isinstance(payload, (UserMessage, AssistantMessage, ToolResultMessage)):
            continue
        if isinstance(payload, UserMessage) or not groups:
            groups.append([])
        groups[-1].append(payload)
    return [Turn(index=i + 1, messages=msgs) for i, msgs in enumerate(groups)]
