"""Adapter: Claude Code hook payload + transcript JSONL → Harness `Turn`s.

Claude Code hooks pipe a JSON payload on stdin (with ``session_id`` and
``transcript_path``); the transcript file is the canonical record of the
conversation, so we use it as the source of truth instead of trying to
reconstruct turns from each hook event.

This module is intentionally pure / I/O-light so it can be unit tested with
synthetic transcripts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..schema import Turn, TurnRole

# Transcript message types we ignore entirely (no Turn produced).
_SKIP_TYPES = frozenset(
    {
        "system",
        "permission-mode",
        "attachment",
        "file-history-snapshot",
        "last-prompt",
        "summary",
    }
)
# Content block types we ignore inside an assistant message.
_SKIP_BLOCK_TYPES = frozenset({"thinking", "image"})


@dataclass(frozen=True)
class HookPayload:
    """Subset of fields we care about from a Claude Code hook payload."""

    session_id: str
    transcript_path: str | None
    hook_event_name: str | None

    @property
    def has_transcript(self) -> bool:
        return self.transcript_path is not None and Path(self.transcript_path).exists()


def parse_hook_payload(text: str) -> HookPayload | None:
    """Parse a Claude Code hook payload. Return ``None`` if not recognizable.

    Hooks are configured to fail-open: if a non-Claude tool pipes garbage on
    stdin or the payload lacks ``session_id``, the harness silently no-ops.
    """

    if not text or not text.strip():
        return None
    try:
        data: Any = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    sid = data.get("session_id")
    if not isinstance(sid, str) or not sid:
        return None
    tp = data.get("transcript_path")
    return HookPayload(
        session_id=sid,
        transcript_path=tp if isinstance(tp, str) else None,
        hook_event_name=data.get("hook_event_name")
        if isinstance(data.get("hook_event_name"), str)
        else None,
    )


def _result_text(block: dict[str, Any]) -> str:
    """Extract a flat text representation from a tool_result block."""

    content = block.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for inner in content:
            if not isinstance(inner, dict):
                continue
            if inner.get("type") == "text":
                t = inner.get("text")
                if isinstance(t, str):
                    parts.append(t)
        return "\n".join(parts)
    return ""


def _msg_to_partial_turns(msg: dict[str, Any]) -> list[dict[str, Any]]:
    """Translate one transcript message into 0+ partial-turn dicts.

    Indices are not assigned here; the caller numbers them sequentially.
    """

    msg_type = msg.get("type")
    if msg_type in _SKIP_TYPES:
        return []
    inner = msg.get("message")
    if not isinstance(inner, dict):
        return []
    role = inner.get("role")
    content = inner.get("content")

    if isinstance(content, str):
        # Plain string content: a real user prompt or a plain assistant reply.
        if not content.strip():
            return []
        if role == "user":
            return [{"role": TurnRole.USER, "content": content}]
        if role == "assistant":
            return [{"role": TurnRole.ASSISTANT, "content": content}]
        return []

    if not isinstance(content, list):
        return []

    out: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        bt = block.get("type")
        if bt in _SKIP_BLOCK_TYPES:
            continue
        if bt == "text":
            text = block.get("text") or ""
            if not text.strip():
                continue
            turn_role = TurnRole.ASSISTANT if role == "assistant" else TurnRole.USER
            out.append({"role": turn_role, "content": text})
        elif bt == "tool_use":
            name = block.get("name") or "tool"
            raw_args = block.get("input")
            args: dict[str, Any] = raw_args if isinstance(raw_args, dict) else {}
            out.append(
                {
                    "role": TurnRole.ASSISTANT,
                    "content": "",
                    "tool_name": str(name),
                    "tool_args": dict(args),
                }
            )
        elif bt == "tool_result":
            text = _result_text(block)
            # `tool_result` blocks always live inside a "user" wrapper message;
            # we promote them to TOOL turns regardless of the outer role.
            out.append({"role": TurnRole.TOOL, "content": text})
    return out


def read_transcript_turns(transcript_path: str | Path) -> list[Turn]:
    """Read the full transcript and return all derivable Turns in order.

    Indices are sequential, starting at 0. The same transcript always yields
    the same list (deterministic), which is what the inbox-delta logic relies
    on for idempotence.
    """

    path = Path(transcript_path)
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    turns: list[Turn] = []
    counter = 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(msg, dict):
            continue
        for partial in _msg_to_partial_turns(msg):
            turns.append(
                Turn(
                    index=counter,
                    role=partial["role"],
                    content=partial.get("content", "") or "",
                    tool_name=partial.get("tool_name"),
                    tool_args=partial.get("tool_args"),
                )
            )
            counter += 1
    return turns


def delta_against(known: list[Turn], full: list[Turn]) -> list[Turn]:
    """Return turns from ``full`` whose index exceeds the max in ``known``.

    Both lists are assumed to share the same indexing scheme (``read_transcript_turns``
    output), so identity is by index alone.
    """

    if not known:
        return list(full)
    last = max(t.index for t in known)
    return [t for t in full if t.index > last]
