"""agentm.gateway — the single-process gateway (channels v2).

One ``agentm gateway`` process holds every chat session in memory and
serves all chat-client peers over the v2 wire protocol. The wire exists
only to keep chat-client vendor SDKs (lark_oapi, Textual) out of the SDK
process. See ``.claude/designs/single-process-gateway.md``.

Public surface (imported by the ``agentm gateway`` CLI subcommand and by
chat-client peers):

* :class:`WireServer` / :class:`WireClient` — wire transport endpoints.
* :class:`Router` / :class:`SessionManager` / :class:`ApprovalManager` —
  gateway components.
* ``DEFAULT_SOCKET_URL`` / ``resolve_token`` — connect-side helpers shared
  by every chat-client CLI.
"""

from __future__ import annotations

from pathlib import Path

from agentm.env import autoload_dotenv
from agentm.gateway_daemon import default_daemon_connect_url


__version__ = "0.2.0"


def default_socket_url() -> str:
    """Conventional ``unix://`` URL shared by gateway and clients.

    Uses the same per-user, per-``AGENTM_HOME`` runtime path as
    ``agentm daemon`` so direct gateway and peer invocations agree:
    ``$AGENTM_RUNTIME_DIR/gateway.sock`` when set, otherwise
    ``$TMPDIR/agentm-<uid>-<home-hash>/gateway.sock``.
    """
    return default_daemon_connect_url(create_runtime_dir=True)


DEFAULT_SOCKET_URL = default_socket_url()


def resolve_token(token: str | None, token_file: str | None) -> str | None:
    """Resolve the bearer token shared by every client CLI.

    Precedence: ``--token-file`` > ``--token`` > ``AGENTM_TOKEN`` (env
    fallback is wired by typer via ``envvar=`` on the ``--token`` flag).
    The two CLI flags are mutually exclusive — secrets should flow through
    a file (no /proc / shell-history leak).

    Returns ``None`` when neither source is set. Raises ``ValueError`` on
    mutual-exclusion violation; CLI layers translate that into typer's
    ``BadParameter`` so the user sees a clean exit-2 error.
    """
    if token_file and token:
        raise ValueError(
            "--token and --token-file are mutually exclusive; "
            "--token-file is preferred (CLI args leak into /proc and shell history)"
        )
    if token_file:
        try:
            content = Path(token_file).read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(f"--token-file {token_file!r}: {exc}") from exc
        stripped = content.strip()
        if not stripped:
            raise ValueError(f"--token-file {token_file!r}: file is empty")
        return stripped
    return token


__all__ = [
    "DEFAULT_SOCKET_URL",
    "autoload_dotenv",
    "default_socket_url",
    "resolve_token",
]
