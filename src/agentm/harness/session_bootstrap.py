"""Public session-bootstrap helpers.

Carved out of :mod:`agentm.cli` so embedders other than the Typer CLI
(channel gateways, RPC servers, notebook drivers) can build an
:class:`AgentSession` without reaching for underscore-prefixed CLI
internals. The CLI now delegates to this module and only adds Typer-
flavored error wrapping on top.

Two functions:

* :func:`make_default_session_store` — returns the canonical
  ``JsonlSessionStore`` rooted at ``<cwd>/.agentm``. Most embedders want
  this; pass a custom :class:`SessionStore` only if you need a different
  on-disk shape.
* :func:`resolve_session_state` — picks the right :class:`SessionState`
  given a resume id / continue-recent flag, falling back to a fresh
  state when neither is requested. Raises :class:`FileNotFoundError`
  for unknown resume ids; CLIs are free to translate that into their
  own error type.
"""

from __future__ import annotations

from pathlib import Path

from agentm.core.abi.session_store import SessionState, SessionStore


def make_default_session_store(cwd: str) -> SessionStore:
    """Return the canonical JSONL-backed :class:`SessionStore` for ``cwd``."""

    from agentm.harness.session_manager import JsonlSessionStore

    return JsonlSessionStore(cwd=Path(cwd))


def resolve_session_state(
    *,
    cwd: str,
    resume: str | None,
    continue_recent: bool,
    session_store: SessionStore,
) -> SessionState:
    """Pick the right :class:`SessionState` for a session bootstrap.

    Resolution order:

    1. ``resume`` is set → :meth:`SessionStore.open` (raises
       :class:`FileNotFoundError` if the id is unknown).
    2. ``continue_recent`` is set → :meth:`SessionStore.most_recent`,
       falling through to ``create`` if no prior session exists.
    3. Otherwise → :meth:`SessionStore.create` (fresh session).
    """

    if resume:
        return session_store.open(resume)
    if continue_recent:
        state = session_store.most_recent(Path(cwd))
        if state is not None:
            return state
    return session_store.create(Path(cwd))


__all__ = ["make_default_session_store", "resolve_session_state"]
