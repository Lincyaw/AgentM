# code-health: ignore-file[AM025] -- ABI DTOs and codecs enforce runtime invariants at trust boundaries
"""Provider-agnostic termination hint sum-type.

The kernel must decide what a turn's end means without speaking any specific
vendor's stop-reason vocabulary. ``TerminationHint`` is the kernel-canonical
shape that LLM provider adapters translate their raw ``stop_reason`` /
``finish_reason`` strings into; the mapping table for each provider lives in
that adapter's module. The agent loop dispatches on this sum-type (see
``agentm.core.abi.loop.default_loop_action``); providers MUST set the
``AssistantMessage.termination`` field on the final ``MessageEnd`` event.

Variants are intentionally narrow:

* :class:`EndTurn` — the model finished its turn cleanly.
* :class:`ToolUseExpected` — the model wants to call tools (next turn).
* :class:`MaxTokens` — output truncated by the provider's token cap.
* :class:`PauseTurn` — provider paused mid-turn and expects the caller to
  resend the same input to continue. The kernel treats this as a
  continuation signal: append the partial assistant message to history
  and step into another turn so the model can finish.
* :class:`ProviderError` — provider reported a non-recoverable error
  (e.g. content filter); ``detail`` carries a short human-readable tag.
* :class:`Aborted` — the request was aborted by the caller (signal).
* :class:`VendorSpecific` — anything the provider couldn't classify; ``raw``
  preserves the original vendor string so downstream tooling can inspect it.
  The kernel treats this as ``EndTurn``-equivalent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal


@dataclass(frozen=True, slots=True)
class TerminationCause:
    """Typed, durable reason why a committed turn ended."""

    overridable: ClassVar[bool] = True
    session_terminal: ClassVar[bool] = False
    replayable: ClassVar[bool] = True


@dataclass(frozen=True, slots=True)
class ModelEndTurn(TerminationCause):
    """The model chose to finish this turn; the session remains active."""


@dataclass(frozen=True, slots=True)
class ToolTerminated(TerminationCause):
    """A tool explicitly terminated the session."""

    overridable: ClassVar[bool] = False
    session_terminal: ClassVar[bool] = True
    tool_name: str = ""
    reason: str = ""


@dataclass(frozen=True, slots=True)
class MaxTurnsExhausted(TerminationCause):
    overridable: ClassVar[bool] = False
    session_terminal: ClassVar[bool] = True


@dataclass(frozen=True, slots=True)
class SignalAborted(TerminationCause):
    """The current turn was interrupted; the session remains reusable."""

    overridable: ClassVar[bool] = False
    session_terminal: ClassVar[bool] = False
    reason: str = ""


@dataclass(frozen=True, slots=True)
class ProviderTruncated(TerminationCause):
    kind: Literal["max_tokens", "error"] = "max_tokens"


@dataclass(frozen=True, slots=True)
class ProviderRequestFailed(TerminationCause):
    """A provider request failed after provider-side retry policy completed."""

    overridable: ClassVar[bool] = False
    session_terminal: ClassVar[bool] = True
    replayable: ClassVar[bool] = False

    error_type: str
    detail: str
    partial_event_count: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.error_type, str) or not self.error_type:
            raise ValueError("provider failure error_type must be non-empty")
        if not isinstance(self.detail, str):
            raise TypeError("provider failure detail must be a string")
        if (
            not isinstance(self.partial_event_count, int)
            or isinstance(self.partial_event_count, bool)
            or self.partial_event_count < 0
        ):
            raise ValueError(
                "provider failure partial_event_count must be non-negative"
            )


@dataclass(frozen=True, slots=True)
class BudgetExhausted(TerminationCause):
    overridable: ClassVar[bool] = False
    session_terminal: ClassVar[bool] = True
    detail: str = ""


@dataclass(slots=True, frozen=True)
class EndTurn:
    """The model finished its turn (no further tool calls expected)."""


@dataclass(slots=True, frozen=True)
class ToolUseExpected:
    """The model emitted (or intends to emit) tool calls."""


@dataclass(slots=True, frozen=True)
class MaxTokens:
    """Output was truncated by the provider's max-output-tokens cap."""


@dataclass(slots=True, frozen=True)
class PauseTurn:
    """Provider paused mid-turn; resend the same input to continue.

    The model is signalling "I have more to say but stopped here" — the
    kernel responds by stepping into another turn with the partial
    assistant message in history so the next request resumes the
    response. Distinct from :class:`MaxTokens`, which is a hard
    truncation by the token budget. Concrete vendor strings that map to
    this variant are documented in each provider adapter, not here.
    """


@dataclass(slots=True, frozen=True)
class ProviderError:
    """Provider reported a non-recoverable error.

    ``detail`` is a short, vendor-neutral tag (e.g. ``"content_filter"``,
    ``"server_error"``) — never an opaque vendor message dump.
    """

    detail: str


@dataclass(slots=True, frozen=True)
class Aborted:
    """The streaming call was aborted by the caller (signal/cancellation)."""


@dataclass(slots=True, frozen=True)
class VendorSpecific:
    """A provider stop reason the adapter couldn't map to any known variant.

    ``raw`` is the original vendor string verbatim. The kernel falls back to
    ``EndTurn``-like behavior so unknown vendor reasons never block the loop.
    """

    raw: str


# Sum-type alias used by the kernel and provider adapters.
TerminationHint = (
    EndTurn
    | ToolUseExpected
    | MaxTokens
    | PauseTurn
    | ProviderError
    | Aborted
    | VendorSpecific
)


__all__ = [
    "Aborted",
    "BudgetExhausted",
    "EndTurn",
    "MaxTokens",
    "MaxTurnsExhausted",
    "ModelEndTurn",
    "PauseTurn",
    "ProviderError",
    "ProviderRequestFailed",
    "ProviderTruncated",
    "SignalAborted",
    "TerminationCause",
    "TerminationHint",
    "ToolTerminated",
    "ToolUseExpected",
    "VendorSpecific",
]
