# code-health: ignore-file[AM025] -- tool-call payloads and model JSON are untyped
"""Side-car tagger: one growing conversation over the agent's whole trajectory.

The tagger used to be a stateless per-turn call that saw exactly one turn. That
made half the vocabulary unanswerable: predicates quantified over the session
("*all* executed test commands narrow scope", "*only* agent-authored tests",
"the same failure in *at least two* runs", "differs from a *previous* run")
cannot be decided from a single step, so the tagger either stayed silent or
guessed. What it did emit was mostly redundant — "agent_has_edited" re-asserted
on every editing turn.

So it is now one conversation that grows by one exchange per turn. The system
prompt and every prior turn stay byte-identical at the head, which is the shape
prompt caching rewards, and each turn only has to report what became true *this*
step.

Turns are summarised as events, not content: which file was read, which was
edited and by how much, which command ran and what it exited with. The bodies —
file contents, diffs, stdout — are what made the old input large, and none of
them decide a predicate.

The provider comes from the session's own registry (``AtomAPI.get_provider``),
so this shares the host's retry policy, cancellation and token accounting
instead of re-opening a second path to the model.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from loguru import logger

from agentm.core.abi import (
    AgentMessage,
    AssistantMessage,
    MessageEnd,
    Model,
    StreamFn,
    text_message,
)

from .compile import generate_tagger_prompt, load_vocabulary
from .pg_query import PgQuerySource

_VOCABULARY_PATH = Path(__file__).parent / "vocabulary.yaml"

_VALID_PHASES = frozenset(
    {"exploring", "diagnosing", "implementing", "validating", "concluding"}
)

# Bodies are dropped, but a command line is itself the evidence for several
# predicates ("does every test command carry a narrowing selector"), so it is
# kept whole up to a generous bound.
_COMMAND_LIMIT = 400
_RESULT_LIMIT = 200
_TEXT_LIMIT = 600
_TASK_LIMIT = 2000


@dataclass(frozen=True, slots=True)
class TurnAnnotation:
    session_id: str
    turn_index: int
    phase: str
    tags: tuple[str, ...]


def _content_text(obj: object, limit: int = 3000) -> str:
    """Extract joined .text from content blocks on any message-like object."""
    content = getattr(obj, "content", None)  # code-health: ignore[AM021]
    if not isinstance(content, (list, tuple)):
        return ""
    parts: list[str] = []
    for block in content:
        text = getattr(block, "text", None)  # code-health: ignore[AM021]
        if isinstance(text, str):
            parts.append(text)
    return "\n".join(parts)[:limit]


@lru_cache(maxsize=1)
def _tagger_system_prompt(vocab_path: str, mtime: float) -> str:
    vocab = load_vocabulary(Path(vocab_path))
    if not vocab:
        return _fallback_tagger_prompt()
    return generate_tagger_prompt(vocab)


def _get_tagger_prompt() -> str:
    if not _VOCABULARY_PATH.is_file():
        return _fallback_tagger_prompt()
    mtime = _VOCABULARY_PATH.stat().st_mtime
    return _tagger_system_prompt(str(_VOCABULARY_PATH), mtime)


def _fallback_tagger_prompt() -> str:
    return (
        "A software-engineering agent solves coding tasks by calling tools "
        "(read, edit, write, bash). You see its work step by step.\n\n"
        "Output JSON with:\n"
        '- "phase": one of exploring/diagnosing/implementing/validating/concluding\n'
        '- "tags": empty array (no vocabulary loaded yet)\n'
    )


# -- Turn rendering ------------------------------------------------------------


def _edit_extent(arguments: Mapping[str, object]) -> str:
    """Rough size of an edit, without carrying the edited text itself."""
    added = arguments.get("new_string")
    removed = arguments.get("old_string")
    added_lines = len(added.splitlines()) if isinstance(added, str) else 0
    removed_lines = len(removed.splitlines()) if isinstance(removed, str) else 0
    if not added_lines and not removed_lines:
        return ""
    return f" (+{added_lines}/-{removed_lines})"


def _render_tool_call(call: Mapping[str, object]) -> str:
    name = str(call.get("name", "?"))
    arguments = call.get("arguments")
    arguments = arguments if isinstance(arguments, Mapping) else {}
    is_error = bool(call.get("is_error"))
    result_text = str(call.get("result_text", ""))

    if name == "bash":
        command = str(arguments.get("cmd", ""))[:_COMMAND_LIMIT]
        status = "FAILED" if is_error else "ok"
        tail = result_text.strip().splitlines()
        detail = tail[-1][:_RESULT_LIMIT] if is_error and tail else ""
        line = f"bash {command} -> {status}"
        return f"{line}\n    {detail}" if detail else line

    path = arguments.get("path") or arguments.get("file_path")
    path = str(path) if isinstance(path, str) else ""
    suffix = _edit_extent(arguments) if name in {"edit", "write"} else ""
    status = " FAILED" if is_error else ""
    purpose = arguments.get("purpose")
    note = f"  [{str(purpose)[:80]}]" if isinstance(purpose, str) and purpose else ""
    return f"{name} {path}{suffix}{status}{note}"


def render_turn(
    turn_index: int,
    assistant_text: str,
    tool_calls: Sequence[Mapping[str, object]],
    *,
    task_text: str = "",
) -> str:
    """One turn as an event line or two — no file bodies, no diffs, no stdout."""
    parts: list[str] = []
    if task_text:
        parts.append(f"Task:\n{task_text[:_TASK_LIMIT]}\n")
    parts.append(f"Turn {turn_index}:")
    if assistant_text:
        parts.append(f"  says: {assistant_text[:_TEXT_LIMIT]}")
    for call in tool_calls:
        parts.append(f"  {_render_tool_call(call)}")
    return "\n".join(parts)


# -- Conversation --------------------------------------------------------------


@dataclass(slots=True)
class TaggerConversation:
    """A model conversation that grows by one exchange per agent turn.

    Only the tail moves, so the cached prefix keeps growing with the session.
    """

    session_id: str
    stream_fn: StreamFn
    model: Model
    system: str = field(default_factory=_get_tagger_prompt)
    messages: list[AgentMessage] = field(default_factory=list)
    seen_tags: set[str] = field(default_factory=set)

    async def annotate(
        self,
        *,
        turn_index: int,
        assistant_text: str,
        tool_calls: Sequence[Mapping[str, object]],
        task_text: str = "",
    ) -> TurnAnnotation | None:
        """Append one turn, ask what became true, keep the reply in history."""
        rendered = render_turn(
            turn_index, assistant_text, tool_calls, task_text=task_text
        )
        self.messages.append(text_message(rendered, timestamp=time.time()))

        reply = await self._complete()
        if reply is None:
            # Drop the unanswered turn so the history stays a clean alternation.
            self.messages.pop()
            return None

        self.messages.append(reply)
        annotation = _parse_tagger_result(
            _content_text(reply, 2000),
            session_id=self.session_id,
            turn_index=turn_index,
        )
        if annotation is not None:
            self.seen_tags.update(annotation.tags)
        return annotation

    async def _complete(self) -> AssistantMessage | None:
        try:
            stream = self.stream_fn(
                messages=list(self.messages),
                model=self.model,
                tools=[],
                system=self.system,
            )
            async for event in stream:
                if isinstance(event, MessageEnd):  # code-health: ignore[AM025]
                    return event.message
        except Exception as exc:  # noqa: BLE001
            logger.warning("tagger: model call failed: {}", exc)
            return None
        logger.warning("tagger: stream ended without a message")
        return None


def _parse_tagger_result(
    text: str, *, session_id: str, turn_index: int
) -> TurnAnnotation | None:
    if not text:
        return None
    clean = text.strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        parsed = json.loads(clean)
    except json.JSONDecodeError:
        logger.warning("tagger: invalid JSON response")
        return None
    if not isinstance(parsed, Mapping):
        return None
    phase = parsed.get("phase", "exploring")
    if phase not in _VALID_PHASES:
        phase = "exploring"
    raw_tags = parsed.get("tags", [])
    tags = tuple(tag for tag in raw_tags if isinstance(tag, str))
    return TurnAnnotation(
        session_id=session_id, turn_index=turn_index, phase=phase, tags=tags
    )


def write_annotation(source: PgQuerySource, annotation: TurnAnnotation) -> bool:
    try:
        source.execute(
            "INSERT INTO policy.turn_annotations "
            "(session_id, turn_index, phase, tags, annotated_at) "
            "VALUES (%s, %s, %s, %s, %s) "
            "ON CONFLICT (session_id, turn_index) DO UPDATE SET "
            "phase = EXCLUDED.phase, tags = EXCLUDED.tags, "
            "annotated_at = EXCLUDED.annotated_at",
            (
                annotation.session_id,
                annotation.turn_index,
                annotation.phase,
                list(annotation.tags),
                time.time(),
            ),
        )
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("tagger: PG write failed: {}", exc)
        return False


__all__ = [
    "TaggerConversation",
    "TurnAnnotation",
    "render_turn",
    "write_annotation",
]
