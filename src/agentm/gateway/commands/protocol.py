"""Command protocol — one shape, four handler kinds.

A slash command is any inbound message whose content starts with
``/`` (and not ``//`` — the second character escapes the prefix so
filesystem paths typed mid-conversation don't get hijacked).

Two dispatch modes:

* ``control`` — handler mutates gateway/session state, optionally
  emits outbound messages, and the loop returns *without* calling
  :meth:`AgentSession.prompt`. The user-typed command text never
  reaches the LLM.
* ``prompt`` — handler returns ``expanded_prompt`` (the template
  applied to the user's args); the gateway rewrites the inbound
  message and falls through to the normal session-prompt path. The
  *expanded* text reaches the LLM; the original ``/foo …`` does not.

A slash command unknown to the gateway registry is *not* rejected by the
router — it returns ``None`` so the gateway can forward the raw ``/...``
text to the session, where the in-session ``slash_commands`` atom
dispatches session-registered commands (``/compact`` etc). Only a name
unknown to *both* layers gets a user-visible "no such command" reply
(emitted by the gateway, which holds the per-session known-command set).
Surfacing that diagnostic when neither layer owns the name is deliberate —
silently forwarding ``/foo`` to the model when the user clearly intended a
command is the kind of failure mode that exposes private prompts.

The :class:`CommandContext` facade is intentionally narrow. Handlers
should never receive a gateway object directly; they get the few
capabilities they need (end the session, read stats, list commands,
reach the live ExtensionAPI). Mirrors the ``ExtensionAPI`` pattern:
atoms reach gateway internals only through documented services.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from ..wire.types import OutboundBody


CommandKind = Literal["control", "prompt"]


async def _default_switch_model(_name: str) -> tuple[bool, str]:
    """Default ``switch_model`` capability — model switching is unsupported on
    this context (e.g. a unit-test stub). The gateway injects a real one."""
    return (False, "model switching is not supported in this context")


async def _default_switch_scenario(_name: str) -> tuple[bool, str]:
    """Default ``switch_scenario`` capability for contexts without a gateway."""
    return (False, "scenario switching is not supported in this context")


async def _default_resume(_sid: str) -> None:
    raise NotImplementedError("resume_session not wired")


async def _default_load_session_history(_sid: str) -> dict[str, Any] | None:
    return None


async def _default_fork(_up_to: int | None) -> str | None:
    raise NotImplementedError("fork_session not wired")


def _default_create_schedule(
    _cron: str,
    _prompt: str,
    *,
    recurring: bool = True,
) -> dict[str, Any]:
    del recurring
    return {"error": "gateway scheduler is not configured"}


async def _default_run_schedule(_job_id: str) -> tuple[bool, str]:
    return (False, "gateway scheduler is not configured")


@dataclass(frozen=True, slots=True)
class CommandInbound:
    """The slash-command-relevant slice of an inbound envelope.

    Built by the gateway from the inbound :class:`Envelope` +
    :class:`InboundBody`; handlers read these fields rather than the
    full wire envelope.
    """

    session_key: str
    channel: str
    chat_id: str
    sender_id: str
    content: str
    thread_id: str | None = None


@dataclass(frozen=True, slots=True)
class CommandInvocation:
    """Parsed user invocation. Created by :func:`parse_invocation`."""

    raw: str
    """The original message content (``/foo bar``)."""

    name: str
    """The command name, lowercased. ``""`` if the message was just ``/``."""

    namespace: str | None
    """Namespace from ``/<ns>:<name>`` syntax; ``None`` for builtins."""

    args: str
    """Everything after the first whitespace, preserved verbatim. Empty
    when the command has no args. *Not* shell-split — handlers parse
    their own args."""

    inbound: CommandInbound
    """The originating inbound — handlers read ``sender_id`` / ``chat_id``
    when needed."""


@dataclass(slots=True)
class CommandResult:
    """What a handler returns.

    For ``control`` handlers, set ``outbound`` (messages to publish)
    and optionally ``side_effect`` (an async callback the router runs
    after publishing). ``expanded_prompt`` must be ``None``.

    For ``prompt`` handlers, set ``expanded_prompt`` to the rewritten
    text the gateway should feed to ``session.prompt(...)``. Other
    fields are ignored.
    """

    outbound: list[OutboundBody] = field(default_factory=list)
    side_effect: Callable[["Any"], Awaitable[None]] | None = None
    expanded_prompt: str | None = None


@dataclass(frozen=True, slots=True)
class CommandContext:
    """Narrow facade passed to every handler — see module docstring."""

    session_key: str
    channel: str
    chat_id: str
    sender_id: str
    thread_id: str | None
    end_session: Callable[[], Awaitable[None]]
    """Tear down the active ``AgentSession`` for this chat."""

    forget_chat_mapping: Callable[[], Awaitable[None]]
    """Clear the persistent ``ChatSessionMap`` entry for this chat so the
    next message starts a brand-new session with no prior history."""

    get_route_stats: Callable[[], dict[str, Any]]
    """Returns a snapshot dict (``session_id``, ``turn_count``,
    ``pending_approvals``) for ``/status``-style introspection."""

    list_commands: Callable[[], list["CommandHandler"]]
    """Returns every handler the router currently knows about. Used by
    ``/help``. The list is a snapshot — handlers must not mutate it."""

    get_gateway_debug_state: Callable[[], dict[str, Any]] = lambda: {}
    """Returns a compact gateway runtime snapshot used by ``/gateway_debug``.

    The payload is intentionally opaque to callers: it may include
    internal fields useful for client-side diagnostics and should be rendered
    verbatim (``json.dumps(..., indent=2)``) rather than interpreted as a
    stable control API."""

    get_extension_api: Callable[[], Any | None] = lambda: None
    """Live ``ExtensionAPI`` for this chat's session, or ``None`` if the
    session has not been created yet. Used by ``/atom:*`` commands."""

    list_models: Callable[[], tuple[str, list[str]]] = lambda: ("", [])
    """Returns ``(active_model_name, available_profile_names)`` for ``/model``.
    Names are the ``[models.<name>]`` keys from ``config.toml``. The active
    value is scoped to this command context's chat/session key."""

    switch_model: Callable[[str], Awaitable[tuple[bool, str]]] = lambda _name: (
        _default_switch_model(_name)
    )
    """Switch this chat's active model profile and start a fresh session.
    Returns ``(ok, message)``."""

    list_scenarios: Callable[[], tuple[str, list[dict[str, str]]]] = lambda: ("", [])
    """Returns ``(active_scenario, scenarios)`` for ``/scenario``. The active
    value is scoped to this command context's chat/session key. Scenario dicts
    carry at least ``name``, ``source``, ``manifest_path`` and ``description``
    keys."""

    switch_scenario: Callable[[str], Awaitable[tuple[bool, str]]] = lambda _name: (
        _default_switch_scenario(_name)
    )
    """Switch this chat's active scenario and start a fresh session.
    Returns ``(ok, message)``."""

    create_schedule: Callable[..., dict[str, Any]] = _default_create_schedule
    """Create a durable gateway scheduled prompt for this chat/session."""

    list_schedules: Callable[[], list[dict[str, Any]]] = lambda: []
    """List durable gateway scheduled prompts targeting this chat/session."""

    delete_schedule: Callable[[str], bool] = lambda _job_id: False
    """Delete a durable gateway scheduled prompt by id."""

    run_schedule: Callable[[str], Awaitable[tuple[bool, str]]] = (
        _default_run_schedule
    )
    """Fire a durable gateway scheduled prompt immediately by id."""

    cwd: str = "."
    """Working directory for the gateway process."""

    resume_session: Callable[[str], Awaitable[None]] = lambda _sid: _default_resume(
        _sid
    )
    """Shut down the current session and set the ChatSessionMap entry to
    ``session_id`` so the next inbound message resumes from that
    session's transcript."""

    load_session_history: Callable[[str], Awaitable[dict[str, Any] | None]] = (
        lambda _sid: _default_load_session_history(_sid)
    )
    """Return a JSON-safe transcript payload for a previously persisted session,
    or ``None`` when the session cannot be loaded."""

    fork_session: Callable[[int | None], Awaitable[str | None]] = (
        lambda _up_to: _default_fork(_up_to)
    )
    """Fork the current session: shut it down and arrange for the next inbound
    message to start a *new* session seeded with the current transcript (up to
    ``up_to`` messages, or all when ``None``). Returns the source session id, or
    ``None`` when there is no active session to fork."""

    list_session_commands: Callable[[], list[str]] = lambda: []
    """Returns the bare names of commands registered *inside* this chat's
    session (dispatched by the in-session ``slash_commands`` atom, e.g.
    ``compact``). These never appear in the gateway registry, so ``/help``
    folds them in separately. Empty until the session's ``session_ready``
    frame has been seen, or for contexts with no session knowledge."""

    def reply(self, text: str, **meta: Any) -> OutboundBody:
        """Build a plain ``assistant_text`` outbound back to this chat."""
        metadata: dict[str, Any] = {"kind": "assistant_text"}
        metadata.update(meta)
        return OutboundBody(
            channel=self.channel,
            chat_id=self.chat_id,
            content=text,
            thread_id=self.thread_id,
            metadata=metadata,
        )

    def session_history(self, payload: dict[str, Any]) -> OutboundBody:
        """Build a ``session_history`` outbound carrying a replayable transcript."""
        metadata: dict[str, Any] = {"kind": "session_history", **payload}
        return OutboundBody(
            channel=self.channel,
            chat_id=self.chat_id,
            content="",
            thread_id=self.thread_id,
            metadata=metadata,
        )

    def notice(self, text: str, *, title: str | None = None, **meta: Any) -> OutboundBody:
        """Build a ``command_result`` outbound — the output of a *control*
        command (``/status``, ``/help``, ...).

        Distinct from :meth:`reply` (``assistant_text``) so a chat client can
        render control-command output as a system notice rather than as agent
        speech: no agent author, no working spinner, not part of the LLM
        transcript. ``title`` is an optional header the client may show above
        the body."""
        metadata: dict[str, Any] = {"kind": "command_result"}
        if title is not None:
            metadata["title"] = title
        metadata.update(meta)
        return OutboundBody(
            channel=self.channel,
            chat_id=self.chat_id,
            content=text,
            thread_id=self.thread_id,
            metadata=metadata,
        )


@runtime_checkable
class CommandHandler(Protocol):
    """The contract every command implements.

    ``name`` is lowercase, no leading slash. ``namespace`` is ``None``
    for builtin/control commands and the literal source name otherwise
    (``"skill"``, ``"atom"``, plugin identifier).
    """

    name: str
    namespace: str | None
    summary: str
    kind: CommandKind

    async def handle(
        self, inv: CommandInvocation, ctx: CommandContext
    ) -> CommandResult: ...


# --- parsing ----------------------------------------------------------


def parse_invocation(msg: CommandInbound) -> CommandInvocation | None:
    """Return a :class:`CommandInvocation` if ``msg`` looks like a
    command, otherwise ``None``.

    Rules:

    * Must start with exactly one ``/``. ``//foo`` (escaped filesystem
      path) returns ``None``.
    * Bare ``/`` (just the slash, no name) returns an invocation with
      empty ``name`` — the router uses that to surface a "type
      ``/help``" hint.
    * Names are lowercased; args are preserved verbatim.
    * ``/<ns>:<name>`` splits on the first ``:`` only — names with
      colons in them (rare, but possible) survive after the first.
    """
    content = msg.content
    if not content.startswith("/"):
        return None
    if content.startswith("//"):
        return None
    stripped = content[1:]
    head, _, args = stripped.partition(" ")
    args = args.lstrip()
    namespace: str | None
    name: str
    if ":" in head:
        ns_part, _, name_part = head.partition(":")
        namespace = ns_part.lower() or None
        name = name_part.lower()
    else:
        namespace = None
        name = head.lower()
    return CommandInvocation(
        raw=content,
        name=name,
        namespace=namespace,
        args=args,
        inbound=msg,
    )


__all__ = [
    "CommandContext",
    "CommandHandler",
    "CommandInbound",
    "CommandInvocation",
    "CommandKind",
    "CommandResult",
    "parse_invocation",
]
