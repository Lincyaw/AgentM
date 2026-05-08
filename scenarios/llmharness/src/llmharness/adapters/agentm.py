"""Adapter: AgentM bus events ↔ llmharness cognitive-audit pipeline.

V0 realization of ``.claude/designs/llmharness-cognitive-audit.md`` §4.5.
The adapter wires two harness events into the audit pipeline:

- :class:`agentm.core.abi.TurnEndEvent` — fires after every assistant turn.
  The handler pulls the full conversation via ``api.session.get_messages()``,
  spawns a child AgentM session whose extension list is composed by
  :func:`llmharness.audit.compose_extensions`, drives the child to completion,
  parses the trailing JSON via :class:`llmharness.audit.RawAuditOutput`,
  persists the verdict + extracted events to :class:`HarnessStore`, and on
  drift writes a single pending Reminder.
- :class:`agentm.harness.events.BeforeAgentStartEvent` — fires at the top of
  every ``AgentSession.prompt``. The handler pops one pending reminder (at
  most) and *appends* its text to ``event.system`` in place with the
  configured ``reminder_prefix`` (default ``"\\n\\n[harness] "``). No
  reminder ⇒ ``event.system`` is left untouched.

V0 design decisions (do NOT modify here without revisiting §4.5):

- Trigger is **every** ``TurnEndEvent`` — no throttling. The diagnostic
  agent itself stays silent on healthy trajectories.
- Injection is **system-prompt mutation**, not ``api.send_user_message``.
  Quieter, advisor-shaped, aligned with the soft-tone authority constraints
  in design §2.

Plug-and-play config knobs (all optional):

- ``root`` (str, default ``".harness"``) — relative to ``api.cwd``; the
  :class:`HarnessStore` root that owns inbox/events/verdicts/reminders.
- ``K_history_events`` (int, default ``50``) — tail length of
  ``store.read_events(sid)`` passed as ``prior_events``.
- ``N_recent_alerts`` (int, default ``5``) — number of recent verdicts fed
  back as ``recent_alerts`` (§5.2 step 1: the diagnostic agent uses these
  plus their ``downstream_reaction`` to decide whether prior reminders were
  heeded).
- ``child_purpose`` (str, default ``"cognitive_audit"``) — surfaced on
  ``ChildSessionStartEvent`` for observability/cost rollups.
- ``reminder_prefix`` (str, default ``"\\n\\n[harness] "``) — what gets
  prepended to the reminder body during ``BeforeAgentStartEvent`` injection.
  Design §5.3 mandates the literal ``[harness] ``; this knob exists so
  multi-harness deployments can disambiguate sources.
- ``audit_extensions`` (list of ``[module_path, config_dict]``, optional) —
  full override of the diagnostic child's extension list. When present,
  ``prompt_override`` / ``cards_tools_config`` / ``observability_config``
  are ignored. Use this to plug in a completely custom audit logic.
- ``prompt_override`` (str, optional) — replace the default audit system
  prompt body. Forwarded to :func:`llmharness.audit.compose_extensions`.
- ``cards_tools_config`` (dict | ``None``, default ``{}``) — config for the
  ``llmharness.atoms.cards_tools`` extension. Set to ``null`` in YAML (or
  pass ``None``) to drop AFC card retrieval entirely.
- ``observability_config`` (dict | ``None``, default ``{}``) — config for the
  ``agentm.extensions.builtin.observability`` extension. Set to ``null`` to
  drop child-session trace capture (loses §7 training-data signal).
"""

from __future__ import annotations

import json
from pathlib import Path
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
from agentm.extensions import ExtensionManifest
from agentm.harness.events import BeforeAgentStartEvent
from agentm.harness.extension import ExtensionAPI, ProviderConfig
from agentm.harness.session_config import AgentSessionConfig

from .. import audit
from ..audit import RawAuditOutput
from ..audit import extract_json as _extract_json
from ..schema import Event, Reminder, Verdict
from ..store import HarnessStore

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
            "root": {"type": "string"},
            "K_history_events": {"type": "integer", "minimum": 0},
            "N_recent_alerts": {"type": "integer", "minimum": 0},
            "child_purpose": {"type": "string"},
            "reminder_prefix": {"type": "string"},
            "audit_extensions": {"type": "array"},
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


# ----------------------------------------------------------------------------
# Defaults
# ----------------------------------------------------------------------------

_DEFAULT_ROOT = ".harness"
_DEFAULT_K_HISTORY_EVENTS = 50
_DEFAULT_N_RECENT_ALERTS = 5
_DEFAULT_CHILD_PURPOSE = "cognitive_audit"
_DEFAULT_REMINDER_PREFIX = "\n\n[harness] "

# Sentinel for ``Reminder.created_at_event_id`` when V0 audit hasn't wired
# up event-id correlation yet. Replaced with real ids in V1.
_NO_EVENT_ID = -1

# Marker config key the bridge-provider branch keys on. Internal contract
# between this adapter and itself when it loads as a child provider; users
# should never set it through ``audit_extensions``.
_BRIDGE_PROVIDER_KEY = "_bridge_provider"


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _recent_alerts(store: HarnessStore, sid: str, n: int) -> list[dict[str, Any]]:
    """Last ``n`` verdicts as plain dicts. The diagnostic agent's self-monitor
    stage (§5.2 step 2) reads each entry's ``downstream_reaction`` to decide
    whether prior reminders were heeded."""

    return [v.to_dict() for v in store.recent_verdicts(sid, n=n)]


def _assistant_text(message: AssistantMessage) -> str:
    """Concatenate every ``TextContent`` block in an assistant message."""

    return "\n".join(b.text for b in message.content if isinstance(b, TextContent))


def _resolve_audit_extensions(
    config: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    """Pick the diagnostic child's extension list from adapter config.

    Resolution precedence:
    1. ``audit_extensions`` — full override, used as-is (with shape validation).
    2. Otherwise, :func:`audit.compose_extensions` with the four knob defaults.
    """

    override = config.get("audit_extensions")
    if override is not None:
        return _coerce_extension_list(override)

    cards_cfg = config.get("cards_tools_config", {})
    obs_cfg = config.get("observability_config", {})
    prompt_override = config.get("prompt_override")
    return audit.compose_extensions(
        prompt_override=prompt_override if isinstance(prompt_override, str) else None,
        cards_tools_config=cards_cfg if isinstance(cards_cfg, dict) else None,
        observability_config=obs_cfg if isinstance(obs_cfg, dict) else None,
    )


def _coerce_extension_list(raw: Any) -> list[tuple[str, dict[str, Any]]]:
    """Validate a user-supplied ``audit_extensions`` value into the typed
    ``list[tuple[module_path, config]]`` shape AgentSessionConfig expects.

    Accepts both ``[module, config]`` two-element lists (YAML) and the
    canonical 2-tuple form (Python callers). Drops malformed entries —
    audit is best-effort, and a typo'd extension entry should not crash
    every turn end.
    """

    if not isinstance(raw, list):
        return []
    out: list[tuple[str, dict[str, Any]]] = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        module_path, ext_config = item
        if not isinstance(module_path, str) or not module_path:
            continue
        if not isinstance(ext_config, dict):
            continue
        out.append((module_path, dict(ext_config)))
    return out


# ----------------------------------------------------------------------------
# install
# ----------------------------------------------------------------------------


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    # Bridge-provider branch: when this module is loaded as a child session's
    # provider extension (via ``provider=(__name__, {"_bridge_provider": parent_provider})``),
    # we just re-publish the parent's ProviderConfig and return. Mirrors
    # ``agentm.extensions.builtin.sub_agent``. The child must NOT re-register
    # turn-end / before-start handlers — that would recurse infinitely.
    #
    # Recursion safety also relies on the audit-extensions composition
    # (:func:`audit.compose_extensions`) NOT including this module. The
    # default composition lists observability + cards_tools + system_prompt;
    # if a user passes ``audit_extensions=[..., ("llmharness.adapters.agentm",
    # {})]`` they'd hit the no-config path below and reinstall handlers on
    # the child — infinite spawn. The factory therefore never inserts this
    # module on its own, and the typed config_schema documents it.
    bridge_provider = config.get(_BRIDGE_PROVIDER_KEY)
    if isinstance(bridge_provider, ProviderConfig):
        api.register_provider(bridge_provider.name, bridge_provider)
        return

    root_str = str(config.get("root", _DEFAULT_ROOT))
    k_history = int(config.get("K_history_events", _DEFAULT_K_HISTORY_EVENTS))
    n_alerts = int(config.get("N_recent_alerts", _DEFAULT_N_RECENT_ALERTS))
    child_purpose = str(config.get("child_purpose", _DEFAULT_CHILD_PURPOSE))
    reminder_prefix = str(config.get("reminder_prefix", _DEFAULT_REMINDER_PREFIX))
    child_extensions = _resolve_audit_extensions(config)

    cwd = Path(api.cwd)
    root_path = Path(root_str)
    if not root_path.is_absolute():
        root_path = cwd / root_path
    store = HarnessStore(root_path)

    # The child session inherits the parent's active provider via the
    # bridge-provider trick: pass ``(__name__, {"_bridge_provider": ...})``
    # so the child loads THIS module as its provider extension and the
    # branch above re-registers the parent's stream_fn without re-running
    # the real provider's install (which would re-authenticate against the
    # LLM gateway).
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
        sid = api.session_id

        # Pull the full conversation history so the audit child sees the
        # user's original request and every assistant turn + tool result.
        # Without this the audit cannot judge task-spec deviation (axis 2
        # needs the user's task) or content correctness on tool outputs
        # (axis 3 needs tool_result blocks). Design §5.2.
        full_messages = api.session.get_messages()

        # Two-stage audit input (design §5.2): the child performs stage A
        # (extract events from raw trajectory) and stage B (three-axis
        # analysis over prior_events plus stage-A events). ``prior_events`` is
        # what the child wrote to the store on previous firings — empty on
        # the first audit, growing as the session continues. Tail-truncated
        # to keep prompt budget bounded.
        prior_events = store.read_events(sid)[-k_history:] if k_history > 0 else []

        payload = {
            "trajectory": _messages_to_dicts(full_messages),
            "prior_events": [e.to_dict() for e in prior_events],
            "recent_alerts": _recent_alerts(store, sid, n=n_alerts),
        }

        try:
            child_provider = _child_provider()
        except RuntimeError:
            return

        child_config = AgentSessionConfig(
            cwd=str(cwd),
            provider=child_provider,
            extensions=child_extensions,
            purpose=child_purpose,
        )
        try:
            child = await api.spawn_child_session(child_config)
        except Exception:
            # Audit is best-effort — never let a diagnostic failure crash
            # the main agent's turn end. Lifecycle errors surface on the
            # parent bus via the install/error event channels for ops.
            return

        try:
            messages = await child.prompt(json.dumps(payload, ensure_ascii=False))
        except Exception:
            await _safe_shutdown(child)
            return

        events, verdict = _audit_output_from_messages(messages, prior_events)
        await _safe_shutdown(child)

        if events:
            store.append_events(sid, events)
        store.append_verdict(sid, verdict)

        # Reminder gating: must be drift, must have body, must have a
        # parseable type. Dropping a reminder when ``verdict.type is None``
        # surfaces LLM-side schema violations instead of papering over them
        # with a default ``TASK_DRIFT`` (which would poison §5.2 step 9
        # same-type suppression on next firing).
        if verdict.drift and verdict.reminder and verdict.type is not None:
            reminder = Reminder(
                session_id=sid,
                type=verdict.type,
                confidence=verdict.confidence,
                text=verdict.reminder,
                created_at_event_id=_NO_EVENT_ID,
            )
            store.write_reminder(reminder)

    def _on_before_agent_start(event: BeforeAgentStartEvent) -> None:
        pending = store.pop_reminder(api.session_id)
        if pending is None:
            return
        # Mutate ``event.system`` in place per design §5.3: prefix + body.
        # Never replace the existing system prompt.
        existing = event.system or ""
        event.system = existing + reminder_prefix + pending.text

    api.on(TurnEndEvent.CHANNEL, _on_turn_end)
    api.on(BeforeAgentStartEvent.CHANNEL, _on_before_agent_start)


# ----------------------------------------------------------------------------
# Verdict + child-session housekeeping
# ----------------------------------------------------------------------------


def _audit_output_from_messages(
    messages: list[AgentMessage], prior_events: list[Event]
) -> tuple[list[Event], Verdict]:
    """Parse the audit child's final assistant text into (new events, verdict).

    Walks newest-first looking for the latest :class:`AssistantMessage` whose
    text contains a parseable JSON block matching :class:`RawAuditOutput`.
    Falls back to ``([], silent verdict)`` when nothing parses — V0 default
    is "stay quiet" (design §2).
    """

    next_id = max((e.id for e in prior_events), default=-1) + 1
    for msg in reversed(messages):
        if not isinstance(msg, AssistantMessage):
            continue
        text = _assistant_text(msg)
        if not text:
            continue
        data = _extract_json(text)
        if not isinstance(data, dict):
            break
        parsed = RawAuditOutput.from_dict(data)
        if parsed is None:
            break
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
    # ImageContent intentionally skipped — pure-text trajectory.
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
        # ThinkingBlock skipped — design §3.3 keeps audit on emitted moves.
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
        # Shutdown errors are non-actionable for the audit pipeline.
        return


__all__ = [
    "MANIFEST",
    "install",
]
