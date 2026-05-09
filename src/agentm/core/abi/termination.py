"""Provider-agnostic termination hint sum-type.

The kernel must decide what a turn's end means without speaking any specific
vendor's stop-reason vocabulary. ``TerminationHint`` is the kernel-canonical
shape that LLM provider adapters (e.g. ``agentm.llm.anthropic``,
``agentm.llm.openai``) translate their raw ``stop_reason`` /
``finish_reason`` strings into. The agent loop dispatches on this sum-type
(see ``agentm.core.abi.loop._default_action``); providers MUST set the
``AssistantMessage.termination`` field on the final ``MessageEnd`` event.

Variants are intentionally narrow:

* :class:`EndTurn` — the model finished its turn cleanly.
* :class:`ToolUseExpected` — the model wants to call tools (next turn).
* :class:`MaxTokens` — output truncated by the provider's token cap.
* :class:`ProviderError` — provider reported a non-recoverable error
  (e.g. content filter); ``detail`` carries a short human-readable tag.
* :class:`Aborted` — the request was aborted by the caller (signal).
* :class:`VendorSpecific` — anything the provider couldn't classify; ``raw``
  preserves the original vendor string so downstream tooling can inspect it.
  The kernel treats this as ``EndTurn``-equivalent.
"""

from __future__ import annotations

from dataclasses import dataclass


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
    | ProviderError
    | Aborted
    | VendorSpecific
)


__all__ = [
    "Aborted",
    "EndTurn",
    "MaxTokens",
    "ProviderError",
    "TerminationHint",
    "ToolUseExpected",
    "VendorSpecific",
]
