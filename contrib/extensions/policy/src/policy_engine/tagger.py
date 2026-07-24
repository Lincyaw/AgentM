"""Per-turn trajectory tagger.

The tagger's system prompt is generated from the predicate vocabulary
(vocabulary.yaml). When compile produces new predicates or prunes old
ones, the tagger prompt updates automatically.

There is no separate task classifier — the tagger handles task-level
predicates on the first turn (which includes the task description as
context) and behavior-level predicates on every turn. All predicates
come from the same vocabulary.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from loguru import logger

from agentm.core.abi import AtomAPI

from .compile import generate_tagger_prompt, load_vocabulary
from .pg_query import PgQuerySource

_VOCABULARY_PATH = Path(__file__).parent / "vocabulary.yaml"

_VALID_PHASES = frozenset(
    {"exploring", "diagnosing", "implementing", "validating", "concluding"}
)


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


def _extract_result_text(result: object) -> str:
    text = getattr(result, "text", None)  # code-health: ignore[AM021]
    if isinstance(text, str):  # code-health: ignore[AM025]
        return text.strip()
    return str(result).strip() if result is not None else ""


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
        "(read, edit, write, bash). You see one step of its work.\n\n"
        "Output JSON with:\n"
        '- "phase": one of exploring/diagnosing/implementing/validating/concluding\n'
        '- "tags": empty array (no vocabulary loaded yet)\n'
    )


def _format_turn_content(
    turn_index: int,
    assistant_text: str,
    tool_calls: Sequence[Mapping[str, object]],
    *,
    task_text: str = "",
) -> str:
    parts = [f"Turn {turn_index}:"]
    if task_text:
        parts.append(f"\nTask description:\n{task_text[:2000]}")
    if assistant_text:
        parts.append(f"\nAssistant reasoning:\n{assistant_text[:2000]}")
    for tc in tool_calls:
        name = tc.get("name", "?")
        args = tc.get("arguments", {})
        result_text = tc.get("result_text", "")
        is_error = tc.get("is_error", False)
        args_summary = json.dumps(args, default=str)[:500]
        parts.append(f"\nTool call: {name}")
        parts.append(f"  Args: {args_summary}")
        if result_text:
            parts.append(
                f"  Result ({'ERROR' if is_error else 'ok'}): {str(result_text)[:800]}"
            )
    return "\n".join(parts)


async def annotate_turn(
    api: AtomAPI,
    *,
    session_id: str,
    turn_index: int,
    assistant_text: str,
    tool_calls: Sequence[Mapping[str, object]],
    task_text: str = "",
) -> TurnAnnotation | None:
    """Annotate one turn. Pass task_text on turn 0 for task-level predicates."""
    system = _get_tagger_prompt()
    prompt = _format_turn_content(
        turn_index, assistant_text, tool_calls, task_text=task_text
    )
    try:
        child = await api.spawn(
            purpose="policy-tagger", system=system, tools=[], max_turns=1
        )
        result = await child.prompt(prompt, origin="policy_engine")
    except Exception as exc:  # noqa: BLE001
        logger.warning("tagger: spawn/prompt failed: {}", exc)
        return None

    return _parse_tagger_result(
        _extract_result_text(result),
        session_id=session_id,
        turn_index=turn_index,
    )


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
