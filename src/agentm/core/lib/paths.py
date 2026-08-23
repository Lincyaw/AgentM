"""Path expansion helpers shared by CLI and gateway peers."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import ParseResult


def expand_path(path: Path | str) -> Path:
    """Expand environment variables and a leading user home marker in ``path``."""

    return Path(os.path.expandvars(str(path))).expanduser()


AGENTM_HOME_ENV = "AGENTM_HOME"
"""The environment variable naming the runtime state directory."""


def agentm_home(env: Mapping[str, str] | None = None) -> Path:
    """Where this installation keeps its runtime state.

    ``$AGENTM_HOME`` when set, ``~/.agentm`` otherwise. One function because
    the rule was written out seven times -- config, scenarios, skills, the
    trajectory resolver and the observability paths each carried their own
    copy, and the seventh, the session control socket, was written without it
    and sent every socket to ``~/.agentm/inbox`` however the deployment was
    configured. A container that mounts ``$AGENTM_HOME`` and leaves ``$HOME``
    read-only could not open one at all.

    ``env`` is accepted because the trajectory resolver is handed an explicit
    environment rather than reading the process's.
    """

    source = os.environ if env is None else env
    configured = source.get(AGENTM_HOME_ENV)
    return expand_path(configured) if configured else Path.home() / ".agentm"


def expand_path_from_cwd(path: Path | str, cwd: Path | str) -> Path:
    """Expand ``path`` and interpret relative results under expanded ``cwd``."""

    expanded = expand_path(str(path).strip())
    if expanded.is_absolute():
        return expanded
    return (expand_path(cwd) / expanded).absolute()


def expand_path_text(path: Path | str) -> str:
    """Text form of :func:`expand_path` for APIs that accept plain strings."""

    return str(expand_path(path))


def expand_optional_path_text(path: Path | str | None) -> str | None:
    """Expand an optional path-like value, preserving ``None`` and empty input."""

    if path is None:
        return None
    raw_path = str(path)
    if not raw_path:
        return None
    return expand_path_text(raw_path)


def parsed_unix_socket_path(parsed: ParseResult) -> str:
    """Return the socket path encoded in a parsed ``unix://`` URL.

    ``urllib.parse.urlparse("unix://~/gw.sock")`` treats ``~`` as ``netloc``
    and ``/gw.sock`` as ``path``. For AgentM's unix-socket URLs, both pieces
    are path text and must be rejoined before normal path expansion.
    """

    if parsed.netloc:
        return f"{parsed.netloc}{parsed.path}"
    return parsed.path
