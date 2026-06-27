"""Token counting helpers shared by atoms.

All SDK-side token estimates go through tiktoken. Character counts are not
used as token substitutes.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import tiktoken

_FALLBACK_ENCODING = "cl100k_base"


@lru_cache(maxsize=64)
def _encoding(model: str | None, fallback_encoding: str) -> tiktoken.Encoding:
    if model:
        try:
            return tiktoken.encoding_for_model(model)
        except KeyError:
            pass
    return tiktoken.get_encoding(fallback_encoding)


def _encode_text(
    text: str,
    *,
    model: str | None = None,
    fallback_encoding: str = _FALLBACK_ENCODING,
) -> list[int]:
    encoding = _encoding(model, fallback_encoding)
    return encoding.encode(text, disallowed_special=())


def count_text_tokens(
    text: str,
    *,
    model: str | None = None,
    fallback_encoding: str = _FALLBACK_ENCODING,
) -> int:
    """Return the tiktoken token count for arbitrary text."""

    if not text:
        return 0
    return len(
        _encode_text(text, model=model, fallback_encoding=fallback_encoding)
    )


@dataclass(frozen=True, slots=True)
class TokenTruncation:
    text: str
    original_tokens: int
    kept_tokens: int

    @property
    def truncated_tokens(self) -> int:
        return max(0, self.original_tokens - self.kept_tokens)

    @property
    def was_truncated(self) -> bool:
        return self.truncated_tokens > 0


def truncate_text_tokens(
    text: str,
    max_tokens: int,
    *,
    model: str | None = None,
    fallback_encoding: str = _FALLBACK_ENCODING,
) -> TokenTruncation:
    """Keep the first ``max_tokens`` tokens of ``text``."""

    if max_tokens < 0:
        raise ValueError("max_tokens must be non-negative")
    if not text or max_tokens == 0:
        return TokenTruncation(
            text="",
            original_tokens=count_text_tokens(
                text,
                model=model,
                fallback_encoding=fallback_encoding,
            ),
            kept_tokens=0,
        )
    encoding = _encoding(model, fallback_encoding)
    tokens = encoding.encode(text, disallowed_special=())
    if len(tokens) <= max_tokens:
        return TokenTruncation(
            text=text,
            original_tokens=len(tokens),
            kept_tokens=len(tokens),
        )
    kept = tokens[:max_tokens]
    return TokenTruncation(
        text=encoding.decode(kept),
        original_tokens=len(tokens),
        kept_tokens=len(kept),
    )


def truncate_text_tokens_middle(
    text: str,
    max_tokens: int,
    *,
    model: str | None = None,
    fallback_encoding: str = _FALLBACK_ENCODING,
) -> TokenTruncation:
    """Keep the first and last portions of ``max_tokens`` tokens."""

    if max_tokens < 0:
        raise ValueError("max_tokens must be non-negative")
    if not text or max_tokens == 0:
        return TokenTruncation(
            text="",
            original_tokens=count_text_tokens(
                text,
                model=model,
                fallback_encoding=fallback_encoding,
            ),
            kept_tokens=0,
        )
    encoding = _encoding(model, fallback_encoding)
    tokens = encoding.encode(text, disallowed_special=())
    if len(tokens) <= max_tokens:
        return TokenTruncation(
            text=text,
            original_tokens=len(tokens),
            kept_tokens=len(tokens),
        )
    head_count = max_tokens // 2
    tail_count = max_tokens - head_count
    kept = tokens[:head_count] + tokens[len(tokens) - tail_count :]
    head = encoding.decode(tokens[:head_count])
    tail = encoding.decode(tokens[len(tokens) - tail_count :])
    return TokenTruncation(
        text=head + tail,
        original_tokens=len(tokens),
        kept_tokens=len(kept),
    )
