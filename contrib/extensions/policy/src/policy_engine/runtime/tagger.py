# code-health: ignore-file[AM025] -- tool-call payloads and model JSON are untyped
"""Side-car tagger: one growing conversation over the agent's whole trajectory.

One conversation that grows by one exchange per turn. The system prompt and
every prior turn stay byte-identical at the head, which is the shape prompt
caching rewards, and each turn only has to report what became true *this* step.

The conversation is what makes half the vocabulary answerable at all: predicates
quantified over the session ("*all* executed test commands narrow scope",
"*only* agent-authored tests", "the same failure in *at least two* runs")
cannot be decided from a single step.

Turns arrive already rendered (see ``record``): events, not content. Bodies are
what make the input large and none of them decide a predicate.

The system prompt is generated from the vocabulary, so a predicate cannot be
gated on in the checklist without the tagger being told what to look for. No
vocabulary means no tagger: there is no empty-tag-list fallback, which would
spend a model call per batch to learn nothing.

The provider comes from the session's own registry (``AtomAPI.get_provider``),
so this shares the host's retry policy, cancellation and token accounting
instead of re-opening a second path to the model.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from loguru import logger

from agentm.core.abi import (
    AgentMessage,
    AssistantMessage,
    FunctionTool,
    JsonValue,
    MessageEnd,
    Model,
    StreamFn,
    ToolCallBlock,
    ToolResult,
    text_message,
    tool_result,
)
from policy_engine.shared.pg_query import PgQuerySource
from policy_engine.shared.vocabulary import PredicateDef, load_vocabulary

_VOCABULARY_PATH = Path(__file__).parent / "vocabulary.yaml"

_VALID_PHASES = frozenset(
    {"exploring", "diagnosing", "implementing", "validating", "concluding"}
)

_RECORD_TOOL = "record"


@dataclass(frozen=True, slots=True)
class TurnAnnotation:
    session_id: str
    turn_index: int
    phase: str
    tags: tuple[str, ...]


# -- System prompt, generated from the vocabulary ------------------------------

_PREAMBLE = """\
A software-engineering agent solves coding tasks by calling tools. You watch it
work, one step at a time, and record what has become true of its trajectory.

You see the whole run so far: this conversation holds every step the agent has
taken and everything you said about them. Each new message carries the steps
taken since your last reply — usually a few, sometimes one.

Steps arrive as events, not contents. A step shows the files read, the files
edited with a line count, the commands run with their exit status, and whatever
the agent said. File bodies, diffs, and command output are not shown — they do
not decide any tag below. The first message also carries the task description.

Answer each message by calling the `record` tool, with two arguments.

**phase** (exactly one, for where the agent stands at the end of these steps):
- "exploring" — reading files, searching, running commands to
  understand the codebase or problem
- "diagnosing" — analyzing a specific issue, forming a hypothesis
  about root cause based on what it read
- "implementing" — editing or creating code files
- "validating" — running tests, builds, linters, or other checks
  to verify its changes work
- "concluding" — declaring done, summarizing what was changed

**tags** (may be empty):
Report only what became true *in these steps* and that you have not already
reported. A tag you emitted earlier stays in force — never repeat it. Many
messages warrant no tag at all; an empty array is a normal answer.

Several tags below quantify over the whole run — "all executed test commands",
"only agent-authored tests", "at least two runs", "differs from a previous run".
Judge those against every step you have seen, and emit the tag as soon as it
first holds. If the run so far leaves one of them uncertain, leave it out; you
will see more steps.

The tag names, with what each means:
"""

_FOOTER = """
Do not restate tags you have already emitted. Emit a tag only on clear
evidence: when the run so far does not settle it, omit it.
"""


def generate_tagger_prompt(vocab: Mapping[str, PredicateDef]) -> str:
    """The tagger system prompt, from the predicate vocabulary."""
    tag_lines = [
        f'- "{name}" — {pred.definition}' for name, pred in sorted(vocab.items())
    ]
    return _PREAMBLE + "\n".join(tag_lines) + "\n" + _FOOTER


@lru_cache(maxsize=1)
def _cached_prompt(vocab_path: str, mtime: float) -> str:
    vocab = load_vocabulary(Path(vocab_path))
    return generate_tagger_prompt(vocab) if vocab else ""


def tagger_system_prompt() -> str | None:
    """The prompt, or ``None`` when there is no vocabulary to tag against."""
    if not _VOCABULARY_PATH.is_file():
        return None
    mtime = _VOCABULARY_PATH.stat().st_mtime
    return _cached_prompt(str(_VOCABULARY_PATH), mtime) or None


def _vocab_names() -> tuple[str, ...]:
    """The tag names the schema will accept — the vocabulary, verbatim."""
    return tuple(sorted(load_vocabulary(_VOCABULARY_PATH)))


def _record_tool(vocab_names: Sequence[str]) -> FunctionTool:
    """The tagger answers by calling this, not by writing JSON.

    Free text cost 19% of annotations on one run: a model with something to say
    puts it before the object, and a strict parse then drops the whole reply —
    disproportionately on the steps worth reading, which are the ones it had
    something to say about. A tool call carries parsed arguments, so there is
    no prose to get past. It is never executed; only the requested arguments
    are read.
    """

    async def record(args: dict[str, JsonValue]) -> ToolResult:
        raise NotImplementedError("the tagger reads arguments, never executes")

    return FunctionTool(
        name=_RECORD_TOOL,
        description="Record what this batch of steps shows.",
        parameters={  # code-health: ignore[AM011]
            "type": "object",
            "properties": {
                "phase": {"type": "string", "enum": sorted(_VALID_PHASES)},
                "tags": {
                    "type": "array",
                    "items": {"type": "string", "enum": list(vocab_names)},
                },
            },
            "required": ["phase", "tags"],
        },
        fn=record,
    )


# -- Conversation --------------------------------------------------------------


@dataclass(slots=True)
class TaggerConversation:
    """A model conversation that grows by one exchange per batch of turns.

    Only the tail moves, so the cached prefix keeps growing with the session.

    Batching costs nothing that is used. The tags feed a session-cumulative
    set, and a trigger reads the set, never which turn a tag arrived on — so
    per-turn resolution buys resolution nobody reads, at one model call per
    turn. Several turns per call is the same signal an order of magnitude
    cheaper. The recorded turn_index is the last turn in the batch, making an
    annotation a statement of what holds *as of* that turn.
    """

    session_id: str
    stream_fn: StreamFn
    model: Model
    #: Required, not defaulted: whoever constructs this has already decided the
    #: vocabulary exists (see ``tagger_system_prompt``).
    system: str
    messages: list[AgentMessage] = field(default_factory=list)
    tool: FunctionTool = field(default_factory=lambda: _record_tool(_vocab_names()))

    async def annotate(
        self, rendered_turns: Sequence[str], *, turn_index: int
    ) -> TurnAnnotation | None:
        """Append a batch of turns, ask what became true, keep the reply."""
        if not rendered_turns:
            return None
        self.messages.append(
            text_message("\n\n".join(rendered_turns), timestamp=time.time())
        )

        reply = await self._complete()
        call = _record_call(reply) if reply is not None else None
        if reply is None or call is None:
            # Drop the unanswered batch so the history stays a clean alternation.
            self.messages.pop()
            if reply is not None:
                logger.warning("tagger: reply did not call {}", _RECORD_TOOL)
            return None

        # Keep the call and a result for it: an assistant tool call left
        # unanswered is not a shape every provider will accept on the next
        # request, and the whole point of this conversation is that it grows.
        self.messages.append(reply)
        self.messages.append(tool_result(call.id, "recorded", timestamp=time.time()))
        return _annotation_from(
            call.arguments, session_id=self.session_id, turn_index=turn_index
        )

    async def _complete(self) -> AssistantMessage | None:
        try:
            stream = self.stream_fn(
                messages=list(self.messages),
                model=self.model,
                tools=[self.tool],
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


def _record_call(message: AssistantMessage) -> ToolCallBlock | None:
    for block in message.content:
        if isinstance(block, ToolCallBlock) and block.name == _RECORD_TOOL:
            return block
    return None


def _annotation_from(
    arguments: Mapping[str, object], *, session_id: str, turn_index: int
) -> TurnAnnotation:
    phase = arguments.get("phase")
    if phase not in _VALID_PHASES:
        phase = "exploring"
    raw_tags = arguments.get("tags")
    tags = (
        tuple(t for t in raw_tags if isinstance(t, str))
        if isinstance(raw_tags, (list, tuple))
        else ()
    )
    return TurnAnnotation(
        session_id=session_id,
        turn_index=turn_index,
        phase=str(phase),
        tags=tags,
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
    "generate_tagger_prompt",
    "tagger_system_prompt",
    "write_annotation",
]
