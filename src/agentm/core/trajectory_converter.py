"""Convert TrajectoryCollector events to multi-agent trajectory JSON.

Reconstructs complete OpenAI-format conversation histories for every agent
(orchestrator + workers) from the JSONL event stream, then assembles them
into the ``{"trajectories": [...]}`` format consumed by the rcabench-platform
evaluation pipeline.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Event types relevant to conversation reconstruction
# ---------------------------------------------------------------------------
_CONVERSATION_EVENTS = {"llm_start", "tool_call", "tool_result", "llm_end"}

# Role mapping: LangChain type -> OpenAI role
_ROLE_MAP = {
    "ai": "assistant",
    "human": "user",
    "system": "system",
    "tool": "tool",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_trajectory_from_events(
    run_id: str,
    events: list[dict[str, Any]],
) -> str:
    """Convert TrajectoryCollector events to multi-agent trajectory JSON.

    Args:
        run_id: Identifier for this execution run.
        events: List of TrajectoryEvent dicts from ``TrajectoryCollector.events``.

    Returns:
        JSON string in the rcabench ``{"trajectories": [...]}`` format.
    """
    agent_groups = _group_events_by_agent(events)

    trajectories: list[dict[str, Any]] = []

    # 1. Orchestrator trajectory
    orch_key = ("orchestrator",)
    if orch_key in agent_groups:
        orch_msgs = _reconstruct_messages(agent_groups[orch_key])
        if orch_msgs:
            trajectories.append(
                {
                    "trajectory_id": run_id,
                    "agent_name": "agentm-orchestrator",
                    "messages": orch_msgs,
                }
            )

    # 2. Worker trajectories (ordered by first event seq)
    worker_keys = sorted(
        (k for k in agent_groups if k != orch_key),
        key=lambda k: agent_groups[k][0]["seq"] if agent_groups[k] else 0,
    )

    for key in worker_keys:
        worker_events = agent_groups[key]
        if not worker_events:
            continue

        worker_msgs = _reconstruct_messages(worker_events)
        if not worker_msgs:
            continue

        # Derive agent_name and task_id
        # key is ("orchestrator", agent_id, task_id) or ("orchestrator", agent_id)
        task_id = worker_events[0].get("task_id")
        # agent_id is always at index 1 (index 0 is "orchestrator")
        agent_id = key[1] if len(key) > 1 else key[0]

        entry: dict[str, Any] = {
            "trajectory_id": task_id or agent_id,
            "agent_name": agent_id,
            "messages": worker_msgs,
        }
        if task_id:
            entry["sub_agent_call_id"] = task_id
        trajectories.append(entry)

    result = {"trajectories": trajectories}
    return json.dumps(result, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Internal: event grouping
# ---------------------------------------------------------------------------



def _group_events_by_agent(
    events: list[dict[str, Any]],
) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    """Group conversation events by agent, using task_id to disambiguate
    multiple dispatches to the same agent_id.

    Keys:
    - ``("orchestrator",)`` for the main orchestrator
    - ``("orchestrator", agent_id, task_id)`` for workers (task_id separates
      multiple dispatches to the same agent_id)
    - ``("orchestrator", agent_id)`` for workers without task_id (fallback)

    Returns dict keyed by agent_path tuple, with events sorted by seq.
    """
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)

    for event in events:
        event_type = event.get("event_type", "")
        if event_type not in _CONVERSATION_EVENTS:
            continue

        agent_path = tuple(event.get("agent_path", []))
        if not agent_path:
            continue

        if len(agent_path) > 1:
            # Worker event — disambiguate by task_id
            task_id = event.get("task_id")
            key = (*agent_path, task_id) if task_id else agent_path
        else:
            key = agent_path

        groups[key].append(event)

    # Sort each group by seq
    for key in groups:
        groups[key].sort(key=lambda e: e.get("seq", 0))

    return dict(groups)


# ---------------------------------------------------------------------------
# Internal: message conversion
# ---------------------------------------------------------------------------


def _langchain_dict_to_openai(msg: dict[str, Any]) -> dict[str, Any]:
    """Convert a serialized LangChain message dict to OpenAI format.

    LangChain serialized messages (from JSONL) have:
    - ``type``: "system" | "human" | "ai" | "tool"
    - ``content``: str
    - ``tool_calls``: list[{name, args, id, type}]  (on AI messages)
    - ``tool_call_id``: str  (on tool messages)
    - ``name``: str | None
    """
    msg_type = msg.get("type", "unknown")
    content = msg.get("content", "") or ""
    role = _ROLE_MAP.get(msg_type, msg_type)

    entry: dict[str, Any] = {"role": role, "content": content}

    if msg_type == "ai":
        tool_calls = msg.get("tool_calls") or []
        if tool_calls:
            entry["tool_calls"] = [
                {
                    "id": tc.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": tc.get("name", ""),
                        "arguments": (
                            json.dumps(tc["args"], ensure_ascii=False)
                            if isinstance(tc.get("args"), dict)
                            else str(tc.get("args", ""))
                        ),
                    },
                }
                for tc in tool_calls
            ]

    elif msg_type == "tool":
        tool_call_id = msg.get("tool_call_id")
        if tool_call_id:
            entry["tool_call_id"] = tool_call_id
        name = msg.get("name")
        if name:
            entry["name"] = name

    return entry


# ---------------------------------------------------------------------------
# Internal: conversation reconstruction
# ---------------------------------------------------------------------------


def _reconstruct_messages(
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Reconstruct OpenAI-format messages from a sequence of events for one agent.

    Handles two patterns:

    **Workers** (AI response usually in llm_start, but not always):
        llm_start (system, human)          → round 1 input
        tool_call × N                      → round 1 output
        tool_result × N                    → tool results
        llm_start (ai+tc, tool, tool...)   → round 2 input (has real AI + tools)

    **Orchestrator** (AI response NOT in llm_start, context prefill instead):
        llm_start (system, human, ai*)     → round 1 input (* = context prefill)
        tool_call × N                      → round 1 output (primary source)
        tool_result × N                    → tool results
        llm_start (tool, tool, ai*)        → round 2 input (tool results + prefill)

    Detection: if ``llm_start`` contains an AI message with ``tool_calls``,
    the AI messages are real responses (worker pattern). Otherwise, AI messages
    are context prefills (orchestrator pattern) and the real response comes
    from ``tool_call`` events.
    """
    # First pass: detect which pattern this agent uses
    uses_context_prefill = _detect_context_prefill_pattern(events)

    if uses_context_prefill:
        return _reconstruct_orchestrator_style(events)
    return _reconstruct_worker_style(events)


def _detect_context_prefill_pattern(events: list[dict[str, Any]]) -> bool:
    """Detect if this agent uses context prefill AI messages in llm_start.

    Returns True if llm_start events contain AI messages without tool_calls
    (orchestrator pattern). Returns False if AI messages in llm_start have
    tool_calls (worker pattern).
    """
    for event in events:
        if event.get("event_type") != "llm_start":
            continue
        for msg in event.get("data", {}).get("messages", []):
            if not isinstance(msg, dict):
                continue
            if msg.get("type") == "ai":
                # If any AI message in llm_start has tool_calls,
                # this is a worker pattern (real AI responses in llm_start)
                if msg.get("tool_calls"):
                    return False
    # No AI messages with tool_calls found in llm_start → context prefill pattern
    return True


def _reconstruct_worker_style(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reconstruct messages for agents where AI responses are in llm_start.

    Accumulates messages from llm_start events. Between llm_starts, if there
    are tool_call events but the NEXT llm_start doesn't start with an AI
    message (e.g. due to injected instructions), synthesizes the missing
    assistant message from tool_call events.
    """
    messages: list[dict[str, Any]] = []
    pending_tool_calls: list[dict[str, Any]] = []
    last_llm_end: dict[str, Any] | None = None

    for event in events:
        event_type = event.get("event_type", "")
        data = event.get("data", {})

        if event_type == "llm_start":
            raw_messages = data.get("messages", [])

            # Check if the first message in this llm_start is an AI message
            # with tool_calls — if so, it covers the pending tool_calls
            first_msg = raw_messages[0] if raw_messages else {}
            first_is_ai_with_tc = (
                isinstance(first_msg, dict)
                and first_msg.get("type") == "ai"
                and first_msg.get("tool_calls")
            )

            if pending_tool_calls and not first_is_ai_with_tc:
                # AI response is missing from llm_start — synthesize it
                _flush_tool_calls_with_ids(messages, pending_tool_calls, data)

            pending_tool_calls = []
            last_llm_end = None

            for raw_msg in raw_messages:
                if isinstance(raw_msg, dict):
                    messages.append(_langchain_dict_to_openai(raw_msg))

        elif event_type == "tool_call":
            pending_tool_calls.append(data)

        elif event_type == "llm_end":
            last_llm_end = data
            pending_tool_calls = []

    # Final response (not captured by a subsequent llm_start)
    if pending_tool_calls:
        _flush_pending_tool_calls(messages, pending_tool_calls)
    elif last_llm_end:
        content = last_llm_end.get("content", "")
        if content:
            messages.append({"role": "assistant", "content": content})

    return messages


def _reconstruct_orchestrator_style(
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Reconstruct messages for agents using context prefill (orchestrator).

    AI messages in llm_start are context prefills — skip them.
    The real AI response comes from tool_call events, and tool results
    come from llm_start as tool-role messages.

    Strategy:
    1. From llm_start: extract system, user, and tool messages (skip ai)
    2. From tool_call events: build assistant message with tool_calls
    3. Tool result messages in the next llm_start pair with the assistant
       message via tool_call_id
    """
    messages: list[dict[str, Any]] = []
    pending_tool_calls: list[dict[str, Any]] = []
    last_llm_end: dict[str, Any] | None = None
    seen_llm_start = False

    for event in events:
        event_type = event.get("event_type", "")
        data = event.get("data", {})

        if event_type == "llm_start":
            # Flush pending tool_calls from previous round as an assistant message
            if seen_llm_start and pending_tool_calls:
                _flush_tool_calls_with_ids(messages, pending_tool_calls, data)
                pending_tool_calls = []

            seen_llm_start = True
            last_llm_end = None

            # Extract non-AI messages from llm_start
            for raw_msg in data.get("messages", []):
                if not isinstance(raw_msg, dict):
                    continue
                msg_type = raw_msg.get("type", "")
                if msg_type == "ai":
                    # Skip context prefill AI messages
                    continue
                messages.append(_langchain_dict_to_openai(raw_msg))

        elif event_type == "tool_call":
            pending_tool_calls.append(data)

        elif event_type == "llm_end":
            last_llm_end = data
            pending_tool_calls = []

    # Final response
    if pending_tool_calls:
        _flush_pending_tool_calls(messages, pending_tool_calls)
    elif last_llm_end:
        content = last_llm_end.get("content", "")
        if content:
            messages.append({"role": "assistant", "content": content})

    return messages


def _build_tc_entries(
    tool_calls: list[dict[str, Any]],
    ids: list[str],
) -> list[dict[str, Any]]:
    return [
        {
            "id": ids[i] if i < len(ids) else f"synth_tc_{i}",
            "type": "function",
            "function": {
                "name": tc.get("tool_name", ""),
                "arguments": (
                    json.dumps(tc["args"], ensure_ascii=False)
                    if isinstance(tc.get("args"), dict)
                    else str(tc.get("args", ""))
                ),
            },
        }
        for i, tc in enumerate(tool_calls)
    ]


def _flush_tool_calls_with_ids(
    messages: list[dict[str, Any]],
    tool_calls: list[dict[str, Any]],
    next_llm_start_data: dict[str, Any],
) -> None:
    """Build an assistant message from tool_call events, using tool_call_ids
    from the subsequent llm_start's tool result messages.

    The tool result messages in the next llm_start have ``tool_call_id``
    fields. We match them with tool_call events by position (both are ordered
    the same way).
    """
    tool_result_ids: list[str] = []
    for msg in next_llm_start_data.get("messages", []):
        if isinstance(msg, dict) and msg.get("type") == "tool":
            tcid = msg.get("tool_call_id", "")
            if tcid:
                tool_result_ids.append(tcid)

    tc_entries = _build_tc_entries(tool_calls, tool_result_ids)
    if tc_entries:
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": tc_entries,
            }
        )


def _flush_pending_tool_calls(
    messages: list[dict[str, Any]],
    tool_calls: list[dict[str, Any]],
) -> None:
    """Append a synthetic assistant message with tool_calls.

    This handles the case where the LLM issued tool calls but there's no
    subsequent ``llm_start`` to capture the response (e.g., final round
    before the agent exits).

    Since ``tool_call`` events from TrajectoryMiddleware don't include
    tool_call_id, we generate placeholder IDs.
    """
    ids = [f"pending_tc_{i}" for i in range(len(tool_calls))]
    tc_entries = _build_tc_entries(tool_calls, ids)
    if tc_entries:
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": tc_entries,
            }
        )
