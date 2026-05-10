"""Card templates.

Feishu interactive cards use a JSON schema with ``header`` / ``elements``
sections. Buttons carry a ``value`` payload that comes back on the
``cardAction`` callback — we encode our own ``card_id`` + ``action``
strings there so the approval bridge can look the card up. Tests rely
only on the *structure* (presence of a button with the right name);
they do not assert the exact Feishu schema.
"""

from __future__ import annotations

from typing import Any


def approval_card(
    *,
    card_id: str,
    tool_name: str,
    summary: str,
    requested_by: str | None = None,
) -> dict[str, Any]:
    """Card asking the requester to approve / deny a pending tool call.

    ``card_id`` is the gateway-assigned id; the buttons echo it back via
    ``value.card_id`` so :class:`approval.ApprovalBridge` can resolve
    the right pending future.
    """

    title = f"Approve `{tool_name}`?"
    if requested_by:
        title = f"{title} (requested by {requested_by})"

    return {
        "schema": "2.0",
        "header": {"title": {"tag": "plain_text", "content": title}},
        "body": {
            "elements": [
                {"tag": "markdown", "content": summary or "*(no summary)*"},
                {
                    "tag": "action",
                    "actions": [
                        {
                            "tag": "button",
                            "text": {"tag": "plain_text", "content": "Approve"},
                            "type": "primary",
                            "name": "approve",
                            "value": {"card_id": card_id, "action": "approve"},
                        },
                        {
                            "tag": "button",
                            "text": {"tag": "plain_text", "content": "Deny"},
                            "type": "danger",
                            "name": "deny",
                            "value": {"card_id": card_id, "action": "deny"},
                        },
                    ],
                },
            ]
        },
    }


def resolved_card(*, tool_name: str, decision: str, by_user: str) -> dict[str, Any]:
    """Replacement card body shown after approval is recorded."""

    icon = "✅" if decision == "approve" else "🛑"
    verb = "approved" if decision == "approve" else "denied"
    return {
        "schema": "2.0",
        "header": {
            "title": {
                "tag": "plain_text",
                "content": f"{icon} `{tool_name}` {verb}",
            }
        },
        "body": {
            "elements": [
                {"tag": "markdown", "content": f"Resolved by **{by_user}**."}
            ]
        },
    }


def assistant_card(*, text: str, in_progress: bool) -> dict[str, Any]:
    """Card body the gateway streams assistant text into.

    Each delta replaces the body with a fresh card via ``update_card``;
    when the turn ends ``in_progress=False`` removes the spinner cue.
    """

    elements: list[dict[str, Any]] = [{"tag": "markdown", "content": text}]
    if in_progress:
        elements.append(
            {
                "tag": "markdown",
                "content": "_…thinking_",
            }
        )
    return {"schema": "2.0", "body": {"elements": elements}}
