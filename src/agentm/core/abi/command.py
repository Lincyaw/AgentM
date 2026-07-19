"""Slash-command types — registration and dispatch contract.

Atoms register slash commands via ``CommandSpec``; the ``slash_commands``
atom resolves typed dispatch through ``CommandDispatcher``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(slots=True)
class CommandSpec:
    """A registered slash command."""

    description: str = ""
    handler: Any = None
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DispatchResult:
    """Outcome of a command dispatch attempt."""

    handled: bool = False
    output: str = ""
    messages: tuple[Any, ...] = ()
    owner: str = ""


@runtime_checkable
class CommandDispatcher(Protocol):
    """Protocol for the slash-command dispatch service."""

    async def dispatch(self, command: str, args: str) -> DispatchResult: ...


__all__ = [
    "CommandDispatcher",
    "CommandSpec",
    "DispatchResult",
]
