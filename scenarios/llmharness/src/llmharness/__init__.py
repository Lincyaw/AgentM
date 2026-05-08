"""LLM-as-harness: inference-time process supervision for the main agent.

P0 is rule-based and dependency-free; the LLM-backed summarizer and drift
reasoner are introduced in later phases. See `.doc/designs/llm-harness.md`.
"""

from .adapters.claude_code import (
    HookPayload,
    delta_against,
    parse_hook_payload,
    read_transcript_turns,
)
from .detector import detect_drift
from .schema import Event, EventKind, Reminder, Turn, TurnRole, Verdict
from .store import HarnessStore
from .summarizer import summarize_turns
from .worker import tick

__all__ = [
    "Event",
    "EventKind",
    "HarnessStore",
    "HookPayload",
    "Reminder",
    "Turn",
    "TurnRole",
    "Verdict",
    "delta_against",
    "detect_drift",
    "parse_hook_payload",
    "read_transcript_turns",
    "summarize_turns",
    "tick",
]
