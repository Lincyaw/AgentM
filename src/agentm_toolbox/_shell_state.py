"""Persistent-cwd shell session state for bash tool wrapping.

Each named shell session tracks its working directory across one-shot
``BashOperations.exec`` calls.  State is synced via files on the
execution site — the toolbox itself is I/O-free: it produces wrapped
commands and parses results, mirroring the ``FileToolbox`` pattern.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Final

_STATE_ROOT: Final[str] = "/tmp/.agentm_shell"

_CWD_SENTINEL: Final[str] = "__AGENTM_SHELL_CWD__"


@dataclass(slots=True)
class ShellSession:
    """Host-side state mirror for one named shell."""

    cwd: str
    _dir_created: bool = False


class ShellStateStore:
    """I/O-free state tracker for multiple named shell sessions.

    The store itself never touches the filesystem — callers (the atom)
    drive ``BashOperations.exec`` with the commands this class produces.
    """

    def __init__(self, *, default_cwd: str, state_root: str = _STATE_ROOT) -> None:
        self._default_cwd = default_cwd
        self._state_root = state_root
        self._sessions: dict[str, ShellSession] = {}

    def get_or_create(self, name: str) -> ShellSession:
        session = self._sessions.get(name)
        if session is None:
            session = ShellSession(cwd=self._default_cwd)
            self._sessions[name] = session
        return session

    def effective_cwd(self, name: str) -> str:
        session = self._sessions.get(name)
        if session is None:
            return self._default_cwd
        return session.cwd

    def _state_dir(self, name: str) -> str:
        return f"{self._state_root}/{shlex.quote(name)}"

    # -- command wrapping ----------------------------------------------------

    def wrap_with_inline_cwd(self, cmd: str, name: str) -> str:
        """Append a pwd-echo suffix to *cmd* so the post-command cwd
        can be recovered from stdout via ``strip_inline_cwd``.

        The caller is expected to pass ``cwd=self.effective_cwd(name)``
        to ``BashOperations.exec`` — this method does NOT prepend a
        ``cd`` preamble.
        """
        session = self.get_or_create(name)
        sd = self._state_dir(name)

        parts: list[str] = []
        if not session._dir_created:
            parts.append(f"mkdir -p {shlex.quote(sd)}")
            session._dir_created = True
        parts.append(f"{{ {cmd}; }}")
        parts.append("__agentm_ec=$?")
        parts.append(f"pwd > {shlex.quote(sd)}/cwd")
        parts.append(f"echo {shlex.quote(_CWD_SENTINEL)}")
        parts.append("pwd")
        parts.append("exit $__agentm_ec")
        return "; ".join(parts)

    def strip_inline_cwd(self, stdout: str, name: str) -> str:
        """Strip the sentinel + pwd suffix from stdout and update state.

        Returns the cleaned stdout (what the model should see).
        """
        idx = stdout.rfind(_CWD_SENTINEL)
        if idx < 0:
            return stdout
        before = stdout[:idx].rstrip("\n")
        after = stdout[idx + len(_CWD_SENTINEL) :]
        cwd = after.strip()
        if cwd:
            session = self.get_or_create(name)
            session.cwd = cwd
        return before
