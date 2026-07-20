"""Shared CLI display helpers and exit-code constants."""

from __future__ import annotations

import sys

from rich.console import Console

from agentm.core.abi import Turn

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_NOT_FOUND = 3
EXIT_AUTH = 4
EXIT_CANCELLED = 6
EXIT_MISSING_DEP = 7

stderr_console = Console(stderr=True)
stdout_console = Console()


def is_tty() -> bool:
    return sys.stdout.isatty()


class SessionStats:
    """Tracks cumulative session token statistics."""

    __slots__ = ("turns", "input_tokens", "output_tokens", "cache_read_tokens")

    def __init__(self) -> None:
        self.turns = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_read_tokens = 0

    def update_from_turn(self, turn: Turn) -> None:
        self.turns += 1
        meta = turn.meta
        self.input_tokens += meta.total_input_tokens
        self.output_tokens += meta.total_output_tokens
        self.cache_read_tokens += meta.cache_read_tokens

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def title_string(self) -> str:
        return (
            f"agentm | turn {self.turns} | "
            f"↑{self.input_tokens:,} ↓{self.output_tokens:,} "
            f"(cache:{self.cache_read_tokens:,})"
        )

    def status_line(self) -> str:
        return (
            f"turn {self.turns} | "
            f"in:{self.input_tokens:,} out:{self.output_tokens:,} "
            f"cache:{self.cache_read_tokens:,} total:{self.total_tokens:,}"
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "turns": self.turns,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "total_tokens": self.total_tokens,
        }
