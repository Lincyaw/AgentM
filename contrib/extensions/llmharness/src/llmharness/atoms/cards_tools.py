"""§11 single-file extension: expose AFC cards as ``cards_list`` /
``cards_get`` tools to the cognitive-audit diagnostic agent.

Wraps :func:`llmharness.cards.cards_list` and :func:`llmharness.cards.cards_get`
(REQ-016) as AgentM ``FunctionTool`` instances so the ``harness_monitor``
scenario can register them via a single ``module: llmharness.atoms.cards_tools``
extension entry. See design ``.claude/designs/llmharness-cognitive-audit.md``
§4.4 — cards live as **tools**, not as static prompt content; the agent's
retrieval choices are observable and become V1 training signal.

Both tool wrappers return JSON text. ``cards_get`` propagates an
``is_error=True`` ToolResult on unknown ids rather than raising — the
diagnostic agent can recover by re-listing.
"""

from __future__ import annotations

import json
from typing import Any

from agentm.core.abi import FunctionTool, TextContent, ToolResult
from agentm.core.abi.extension import ExtensionAPI
from agentm.extensions import ExtensionManifest

from ..cards import cards_get as _cards_get
from ..cards import cards_list as _cards_list

MANIFEST = ExtensionManifest(
    name="cards_tools",
    description=(
        "Register cards_list / cards_get tools so the harness_monitor "
        "diagnostic agent can retrieve AFC failure cards on demand."
    ),
    registers=(
        "tool:cards_list",
        "tool:cards_get",
    ),
    config_schema={
        "type": "object",
        "properties": {},
        "additionalProperties": True,
    },
    api_version=1,
    tier=1,
)


_LIST_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}


_GET_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "card_id": {
            "type": "string",
            "description": (
                "AFC card id, e.g. 'AFC-0016'. Must match an id surfaced by cards_list."
            ),
        },
    },
    "required": ["card_id"],
    "additionalProperties": False,
}


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    """Register both tool wrappers on ``api``."""

    del config  # no configuration knobs in V0

    async def _list(args: dict[str, Any]) -> ToolResult:
        del args
        try:
            summaries = _cards_list()
        except Exception as exc:  # pragma: no cover - surfaces to agent
            return _error(f"cards_list failed: {exc}")
        payload = [s.to_dict() for s in summaries]
        return _ok_json(payload)

    async def _get(args: dict[str, Any]) -> ToolResult:
        card_id_raw = args.get("card_id", "")
        card_id = str(card_id_raw).strip()
        if not card_id:
            return _error("cards_get requires a non-empty 'card_id' string.")
        try:
            card = _cards_get(card_id)
        except KeyError:
            return _error(
                f"Unknown card id: {card_id!r}. Call cards_list to see every available id."
            )
        except Exception as exc:  # pragma: no cover - surfaces to agent
            return _error(f"cards_get failed: {exc}")
        return _ok_json(card.to_dict())

    api.register_tool(
        FunctionTool(
            name="cards_list",
            description=(
                "List all AFC failure cards as compact summaries (id, name, "
                "axis_hint, one_line_mechanism). Call once per audit firing "
                "before drilling into specific cards via cards_get."
            ),
            parameters=_LIST_PARAMETERS,
            fn=_list,
        )
    )
    api.register_tool(
        FunctionTool(
            name="cards_get",
            description=(
                "Fetch the full YAML payload for one AFC card (mechanism, "
                "activation, observable, downstream_effects, evidence). "
                "Cite the returned id in the audit Verdict's cited_cards "
                "field when the card materially shaped the finding."
            ),
            parameters=_GET_PARAMETERS,
            fn=_get,
        )
    )


def _ok_json(payload: Any) -> ToolResult:
    return ToolResult(
        content=[
            TextContent(
                type="text",
                text=json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            )
        ]
    )


def _error(message: str) -> ToolResult:
    return ToolResult(
        content=[TextContent(type="text", text=message)],
        is_error=True,
    )
