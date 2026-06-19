"""Public session-bootstrap helpers.

Carved out of :mod:`agentm.cli` so embedders other than the Typer CLI
(channel gateways, RPC servers, notebook drivers) can build an
:class:`AgentSession` without reaching for underscore-prefixed CLI
internals. The CLI now delegates to this module and only adds Typer-
flavored error wrapping on top.

Two functions:

* :func:`make_default_session_store` — returns the canonical session
  store.  When ClickHouse is reachable the store reads session state
  from the collector (no local JSONL); otherwise falls back to
  ``JsonlSessionStore``.
* :func:`resolve_session_state` — picks the right :class:`SessionState`
  given a resume id / continue-recent flag, falling back to a fresh
  state when neither is requested. Raises :class:`FileNotFoundError`
  for unknown resume ids; CLIs are free to translate that into their
  own error type.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from loguru import logger

from agentm.core.abi.session_store import SessionState, SessionStore


class ClickHouseSessionStore:
    """Session store backed by ClickHouse (OTLP collector).

    ``create`` returns an in-memory (non-persisting) session manager —
    messages reach ClickHouse via the OTLP exporter, not via local
    files.  ``open`` / ``most_recent`` / ``fork`` reconstruct state by
    querying ClickHouse for the recorded header + message entries.
    """

    def __init__(self, url: str) -> None:
        self._url = url

    @staticmethod
    def _ch() -> Any:
        from agentm.core.observability import clickhouse
        return clickhouse

    def create(self, cwd: Path) -> SessionState:
        from agentm.core.runtime.session_manager import SessionManager
        return SessionManager(cwd=str(cwd), persist=False)

    def open(self, id: str) -> SessionState:
        from agentm.core.runtime.session_manager import (
            SessionManager,
            _header_from_record,
        )

        ch = self._ch()
        header_raw = ch.session_header(self._url, id)
        if header_raw is None:
            raise FileNotFoundError(id)
        entries = ch.session_entries(self._url, id)
        return SessionManager.from_records(
            _header_from_record(header_raw), entries,
        )

    def most_recent(self, cwd: Path) -> SessionState | None:
        ch = self._ch()
        sid = ch.most_recent_session_id(self._url, str(cwd))
        if sid is None:
            return None
        return self.open(sid)

    def fork(
        self,
        source_id: str,
        *,
        up_to: int | None = None,
        message_id: str | None = None,
        turn_id: int | None = None,
        turn_index: int | None = None,
    ) -> SessionState:
        from agentm.core.abi.session import AgentMessage
        from agentm.core.runtime.session_manager import (
            ENTRY_TYPE_MESSAGE,
            SessionManager,
            resolve_turn_leaf_id,
        )

        source: SessionManager = self.open(source_id)  # type: ignore[assignment]
        selector_count = sum(
            value is not None for value in (message_id, turn_id, turn_index)
        )
        if selector_count > 1:
            raise ValueError("message_id, turn_id, and turn_index are mutually exclusive")

        if message_id is not None:
            entry = source.get_entry(message_id)
            if entry is None:
                raise KeyError(f"unknown message id: {message_id}")
            if entry.type != ENTRY_TYPE_MESSAGE:
                raise ValueError(f"entry {message_id!r} is not a message entry")
            return source.fork_at_entry(message_id, persist=False)

        if turn_id is not None or turn_index is not None:
            turns = list(self._ch().turns(self._url, source_id))
            leaf_id = resolve_turn_leaf_id(
                source,
                turns,
                turn_id=turn_id,
                turn_index=turn_index,
            )
            return source.fork_at_entry(leaf_id, persist=False)

        branch = source.get_branch()
        messages = [
            e.payload
            for e in branch
            if e.type == ENTRY_TYPE_MESSAGE and isinstance(e.payload, AgentMessage)
        ]
        if up_to is not None:
            messages = messages[:up_to]

        forked = SessionManager(
            cwd=source._cwd,
            persist=False,
            parent_session=source.get_session_id(),
        )
        header = source.get_header()
        if header is not None and header.config:
            forked.set_session_config(copy.deepcopy(header.config))
        for msg in messages:
            forked.append_message(msg)
        return forked


def make_default_session_store(cwd: str) -> SessionStore:
    """Return the best available session store.

    Prefers ClickHouse when reachable (no local file I/O); falls back
    to the JSONL-backed store otherwise.
    """

    try:
        from agentm.core.observability import clickhouse
        url = clickhouse.get_url()
        if url is not None:
            return ClickHouseSessionStore(url)  # type: ignore[return-value]
    except Exception as exc:
        # ClickHouse store unavailable/misconfigured — fall back to the local
        # JSONL store. Log so an operator who expected CH knows why it fell back.
        logger.debug("session store: ClickHouse unavailable, using JSONL store: {}", exc)
    from agentm.core.runtime.session_manager import JsonlSessionStore
    return JsonlSessionStore(cwd=Path(cwd))


def resolve_session_state(
    *,
    cwd: str,
    resume: str | None,
    continue_recent: bool,
    session_store: SessionStore,
    fork: str | None = None,
    fork_up_to: int | None = None,
    fork_message_id: str | None = None,
    fork_turn_id: int | None = None,
    fork_turn_index: int | None = None,
) -> SessionState:
    """Pick the right :class:`SessionState` for a session bootstrap.

    Resolution order:

    1. ``fork`` is set → :meth:`SessionStore.fork` — new session seeded
       with messages from the source (raises :class:`FileNotFoundError`
       if the source id is unknown).
    2. ``resume`` is set → :meth:`SessionStore.open` (raises
       :class:`FileNotFoundError` if the id is unknown).
    3. ``continue_recent`` is set → :meth:`SessionStore.most_recent`,
       falling through to ``create`` if no prior session exists.
    4. Otherwise → :meth:`SessionStore.create` (fresh session).
    """

    if fork:
        return session_store.fork(
            fork,
            up_to=fork_up_to,
            message_id=fork_message_id,
            turn_id=fork_turn_id,
            turn_index=fork_turn_index,
        )
    if resume:
        return session_store.open(resume)
    if continue_recent:
        state = session_store.most_recent(Path(cwd))
        if state is not None:
            return state
    return session_store.create(Path(cwd))


__all__ = ["make_default_session_store", "resolve_session_state"]
