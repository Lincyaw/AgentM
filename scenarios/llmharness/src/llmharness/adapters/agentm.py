"""Adapter: AgentM bus events ↔ llmharness cognitive-audit pipeline.

V0 realization of ``.claude/designs/llmharness-cognitive-audit.md`` §4.5.
The adapter wires two harness events into the audit pipeline:

- :class:`agentm.core.abi.TurnEndEvent` — fires after every assistant turn.
  Pulls the full conversation via ``api.session.get_messages()``, spawns a
  child AgentM session whose extension list comes from
  :func:`llmharness.audit.compose_extensions`, drives the child to completion,
  parses the trailing JSON via :class:`llmharness.audit.RawAuditOutput`,
  appends the verdict + extracted events to the session entry tree via
  ``api.session.append_entry``, and on drift caches a single pending Reminder
  in adapter-local state.
- :class:`agentm.harness.events.BeforeAgentStartEvent` — fires at the top of
  every ``AgentSession.prompt``. Consumes the pending reminder (if any) and
  *appends* its text to ``event.system`` in place with the configured
  ``reminder_prefix`` (default ``"\\n\\n[harness] "``).

V0 design decisions (do NOT modify here without revisiting §4.5):

- Trigger is **every** ``TurnEndEvent`` — no throttling. The diagnostic
  agent stays silent on healthy trajectories.
- Injection is **system-prompt mutation**, not ``api.send_user_message``.
  Quieter, advisor-shaped, aligned with the soft-tone authority constraints
  in design §2.

Persistence:

All audit state lives on the session entry tree, not on disk. Verdicts and
audit events are appended via ``api.session.append_entry`` under the entry
types ``llmharness.verdict`` / ``llmharness.audit_event`` and read back by
filtering ``api.session.get_branch()``. The pending reminder is held in a
single closure slot — at most one reminder ever queues per session, by
design (§5.3).

Plug-and-play config knobs (all optional):

- ``K_history_events`` (int, default ``50``) — tail length of prior audit
  events fed back as ``prior_events`` to the child.
- ``N_recent_alerts`` (int, default ``5``) — number of recent verdicts fed
  back as ``recent_alerts`` (§5.2 step 1).
- ``child_purpose`` (str, default ``"cognitive_audit"``) — surfaced on
  ``ChildSessionStartEvent`` for observability/cost rollups.
- ``reminder_prefix`` (str, default ``"\\n\\n[harness] "``) — what gets
  prepended to the reminder body during ``BeforeAgentStartEvent`` injection.
- ``prompt_override`` (str, optional) — replace the default audit system
  prompt. Forwarded to :func:`llmharness.audit.compose_extensions`.
- ``cards_tools_config`` (dict | ``None``, default ``{}``) — config for the
  ``llmharness.atoms.cards_tools`` extension. ``null`` drops AFC retrieval.
- ``observability_config`` (dict | ``None``, default ``{}``) — config for the
  ``agentm.extensions.builtin.observability`` extension. ``null`` drops it.
"""

from __future__ import annotations

import json
from typing import Any

from agentm.core.abi import TurnEndEvent
from agentm.core.abi.messages import (
    AgentMessage,
    AssistantMessage,
    TextContent,
    ToolCallBlock,
    ToolResultMessage,
    UserMessage,
)
from agentm.core.abi.session import SessionEntry
from agentm.extensions import ExtensionManifest
from agentm.harness.events import BeforeAgentStartEvent
from agentm.harness.extension import ExtensionAPI, ProviderConfig
from agentm.harness.session_config import AgentSessionConfig

from ..audit import RawAuditOutput, compose_extensions
from ..audit.submit_tool import SUBMIT_AUDIT_TOOL_NAME
from ..schema import Event, Reminder, Verdict

MANIFEST = ExtensionManifest(
    name="agentm",
    description=(
        "Cognitive-audit adapter: spawn diagnostic child session on TurnEndEvent "
        "and inject pending reminders on BeforeAgentStartEvent."
    ),
    registers=(
        "event:turn_end",
        "event:before_agent_start",
    ),
    config_schema={
        "type": "object",
        "properties": {
            "K_history_events": {"type": "integer", "minimum": 0},
            "N_recent_alerts": {"type": "integer", "minimum": 0},
            "child_purpose": {"type": "string"},
            "reminder_prefix": {"type": "string"},
            "prompt_override": {"type": "string"},
            "cards_tools_config": {"type": ["object", "null"]},
            "observability_config": {"type": ["object", "null"]},
        },
        "additionalProperties": True,
    },
    affects=(
        "event:turn_end",
        "event:before_agent_start",
    ),
    api_version=1,
    tier=1,
)


_DEFAULT_K_HISTORY_EVENTS = 50
_DEFAULT_N_RECENT_ALERTS = 5
_DEFAULT_CHILD_PURPOSE = "cognitive_audit"
_DEFAULT_REMINDER_PREFIX = "\n\n[harness] "

# Entry types used on the session tree. Namespaced so consumers querying
# the tree can filter unambiguously.
_VERDICT_ENTRY_TYPE = "llmharness.verdict"
_AUDIT_EVENT_ENTRY_TYPE = "llmharness.audit_event"

# Marker config key the bridge-provider branch keys on. Internal contract
# between this adapter and itself when it loads as a child provider; users
# should never set it.
_BRIDGE_PROVIDER_KEY = "_bridge_provider"


def _entries_of_type(branch: list[SessionEntry], type_: str) -> list[SessionEntry]:
    return [e for e in branch if e.type == type_]


def _read_prior_events(api: ExtensionAPI, k: int) -> list[Event]:
    if k <= 0:
        return []
    branch = api.session.get_branch()
    entries = _entries_of_type(branch, _AUDIT_EVENT_ENTRY_TYPE)[-k:]
    return [Event.from_dict(e.payload) for e in entries]


def _read_recent_verdicts(api: ExtensionAPI, n: int) -> list[Verdict]:
    if n <= 0:
        return []
    branch = api.session.get_branch()
    entries = _entries_of_type(branch, _VERDICT_ENTRY_TYPE)[-n:]
    return [Verdict.from_dict(e.payload) for e in entries]


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    # Bridge-provider branch: when this module is loaded as a child session's
    # provider extension (via ``provider=(__name__, {"_bridge_provider":
    # parent_provider})``), re-publish the parent's ProviderConfig and return.
    # Mirrors ``agentm.extensions.builtin.sub_agent``. The child must NOT
    # re-register turn-end / before-start handlers — that would recurse
    # infinitely. The compose_extensions() default never includes this module,
    # so the only path here is the deliberate one below.
    bridge_provider = config.get(_BRIDGE_PROVIDER_KEY)
    if isinstance(bridge_provider, ProviderConfig):
        api.register_provider(bridge_provider.name, bridge_provider)
        return

    k_history = int(config.get("K_history_events", _DEFAULT_K_HISTORY_EVENTS))
    n_alerts = int(config.get("N_recent_alerts", _DEFAULT_N_RECENT_ALERTS))
    child_purpose = str(config.get("child_purpose", _DEFAULT_CHILD_PURPOSE))
    reminder_prefix = str(config.get("reminder_prefix", _DEFAULT_REMINDER_PREFIX))

    cards_cfg = config.get("cards_tools_config", {})
    obs_cfg = config.get("observability_config", {})
    prompt_override = config.get("prompt_override")
    child_extensions = compose_extensions(
        prompt_override=prompt_override if isinstance(prompt_override, str) else None,
        cards_tools_config=cards_cfg if isinstance(cards_cfg, dict) else None,
        observability_config=obs_cfg if isinstance(obs_cfg, dict) else None,
    )

    # Single-slot pending reminder, scoped to this session. List used as a
    # mutable closure cell for type-checker friendliness; len ≤ 1 by design.
    pending: list[Reminder] = []

    def _child_provider() -> tuple[str, dict[str, Any]]:
        provider = api.provider
        if provider is None:
            raise RuntimeError(
                "agentm cognitive-audit adapter: no active provider on parent "
                "session; cannot spawn diagnostic child."
            )
        return (__name__, {_BRIDGE_PROVIDER_KEY: provider})

    async def _on_turn_end(event: TurnEndEvent) -> None:
        del event  # full conversation pulled from session.get_messages() below

        full_messages = api.session.get_messages()
        prior_events = _read_prior_events(api, k_history)
        recent_alerts = [v.to_dict() for v in _read_recent_verdicts(api, n_alerts)]

        payload = {
            "trajectory": _messages_to_dicts(full_messages),
            "prior_events": [e.to_dict() for e in prior_events],
            "recent_alerts": recent_alerts,
        }

        try:
            child_provider = _child_provider()
        except RuntimeError:
            return

        child_config = AgentSessionConfig(
            cwd=api.cwd,
            provider=child_provider,
            extensions=child_extensions,
            purpose=child_purpose,
        )
        try:
            child = await api.spawn_child_session(child_config)
        except Exception:
            # Audit is best-effort — never let a diagnostic failure crash the
            # main agent's turn end.
            return

        try:
            messages = await child.prompt(json.dumps(payload, ensure_ascii=False))
        except Exception:
            await _safe_shutdown(child)
            return

        events, verdict = _audit_output_from_messages(messages, prior_events)
        await _safe_shutdown(child)

        for ev in events:
            api.session.append_entry(_AUDIT_EVENT_ENTRY_TYPE, ev.to_dict())
        api.session.append_entry(_VERDICT_ENTRY_TYPE, verdict.to_dict())

        # Reminder gating: must be drift, must have body, must have a
        # parseable type. Dropping a reminder when ``verdict.type is None``
        # surfaces LLM-side schema violations instead of papering over them
        # with a default ``TASK_DRIFT`` (which would poison §5.2 step 9
        # same-type suppression on next firing).
        if verdict.drift and verdict.reminder and verdict.type is not None:
            pending[:] = [
                Reminder(
                    type=verdict.type,
                    confidence=verdict.confidence,
                    text=verdict.reminder,
                )
            ]

    def _on_before_agent_start(event: BeforeAgentStartEvent) -> None:
        if not pending:
            return
        reminder = pending.pop()
        existing = event.system or ""
        event.system = existing + reminder_prefix + reminder.text

    api.on(TurnEndEvent.CHANNEL, _on_turn_end)
    api.on(BeforeAgentStartEvent.CHANNEL, _on_before_agent_start)


def _audit_output_from_messages(
    messages: list[AgentMessage], prior_events: list[Event]
) -> tuple[list[Event], Verdict]:
    """Pull (new events, verdict) off the audit child's ``submit_audit`` tool call.

    The audit child terminates by calling ``submit_audit(events=..., verdict=...)``.
    The kernel records the call as a :class:`ToolCallBlock` on the final
    assistant message; we walk newest-first looking for that block and read
    its ``arguments`` dict directly — already structured, schema-validated
    by the LLM provider's tool-use surface. Falls back to ``([], silent
    verdict)`` when no submit_audit call is found (V0 default: stay quiet
    per design §2).
    """

    next_id = max((e.id for e in prior_events), default=-1) + 1
    for msg in reversed(messages):
        if not isinstance(msg, AssistantMessage):
            continue
        for block in reversed(msg.content):
            if not isinstance(block, ToolCallBlock):
                continue
            if block.name != SUBMIT_AUDIT_TOOL_NAME:
                continue
            parsed = RawAuditOutput.from_dict(block.arguments)
            if parsed is None:
                return [], Verdict(drift=False)
            return parsed.to_events(next_id=next_id), parsed.to_verdict()
    return [], Verdict(drift=False)


def _messages_to_dicts(messages: list[AgentMessage]) -> list[dict[str, Any]]:
    """Serialize a full conversation history for the audit child.

    ``isinstance`` dispatch over the kernel's message dataclasses preserves
    role + content semantics. Drops thinking and image blocks — the audit
    operates on pure-text trajectory material (design §3.3). Each message
    carries an ``index`` so the audit can reference specific positions in
    its ``source_turns`` field.
    """

    out: list[dict[str, Any]] = []
    for i, msg in enumerate(messages):
        if isinstance(msg, UserMessage):
            blocks = _serialize_user(msg)
        elif isinstance(msg, AssistantMessage):
            blocks = _serialize_assistant(msg)
        elif isinstance(msg, ToolResultMessage):
            blocks = _serialize_tool_result(msg)
        else:
            continue
        if blocks:
            out.append({"index": i, "role": msg.role, "content": blocks})
    return out


def _serialize_user(msg: UserMessage) -> list[dict[str, Any]]:
    return [
        {"type": "text", "text": b.text}
        for b in msg.content
        if isinstance(b, TextContent) and b.text
    ]


def _serialize_assistant(msg: AssistantMessage) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for b in msg.content:
        if isinstance(b, TextContent):
            if b.text:
                out.append({"type": "text", "text": b.text})
        elif isinstance(b, ToolCallBlock):
            out.append(
                {
                    "type": "tool_call",
                    "id": b.id,
                    "name": b.name,
                    "arguments": dict(b.arguments),
                }
            )
    return out


def _serialize_tool_result(msg: ToolResultMessage) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for b in msg.content:
        text_parts = [
            inner.text for inner in b.content if isinstance(inner, TextContent)
        ]
        out.append(
            {
                "type": "tool_result",
                "tool_call_id": b.tool_call_id,
                "content": "\n".join(text_parts),
                "is_error": b.is_error,
            }
        )
    return out


async def _safe_shutdown(child: Any) -> None:
    shutdown = getattr(child, "shutdown", None)
    if shutdown is None:
        return
    try:
        result = shutdown()
        if hasattr(result, "__await__"):
            await result
    except Exception:
        return


__all__ = [
    "MANIFEST",
    "install",
]
