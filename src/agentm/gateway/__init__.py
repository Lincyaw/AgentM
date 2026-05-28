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

import os
from pathlib import Path


__version__ = "0.2.0"

_dotenv_loaded: bool = False


def autoload_dotenv() -> None:
    """Idempotent ``.env`` autoload for gateway / chat-client CLIs.

    Must be called at module-import time in every gateway-related CLI so
    environment variables (``AGENTM_PROVIDER``, ``LARK_APP_ID``, …) are
    visible by the time typer evaluates option defaults (including
    ``envvar=...``). Subsequent calls are no-ops.

    Opt-out: set ``AGENTM_SKIP_DOTENV=1`` to disable autoloading. Tests
    rely on this to keep the repo's own ``.env`` out of fixtures.
    """
    global _dotenv_loaded
    if _dotenv_loaded:
        return
    _dotenv_loaded = True
    if os.environ.get("AGENTM_SKIP_DOTENV"):
        return
    _load_dotenv_files(Path.cwd())


def _load_dotenv_files(cwd: Path) -> None:
    """Best-effort ``.env`` autoload: ``<cwd>/.env`` then the workspace-root
    ``.env`` (next to the nearest ``[tool.uv.workspace]`` pyproject, walked
    up at most 8 levels). No-op when ``python-dotenv`` isn't installed."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    candidates: list[Path] = [cwd / ".env"]
    walker = cwd.resolve()
    for _ in range(8):
        manifest = walker / "pyproject.toml"
        if manifest.exists():
            try:
                if "[tool.uv.workspace]" in manifest.read_text(encoding="utf-8"):
                    candidates.append(walker / ".env")
                    break
            except OSError:
                pass
        if walker.parent == walker:
            break
        walker = walker.parent
    for path in candidates:
        if path.exists():
            load_dotenv(path, override=False)


def default_socket_url() -> str:
    """Conventional ``unix://`` URL shared by gateway and clients.

    Uses ``$XDG_RUNTIME_DIR/agentm-gw.sock`` when set (per-user runtime
    dir on Linux desktops; cleared on logout), else
    ``/tmp/agentm-gw-<uid>.sock`` (uid-suffixed to avoid clobber on
    shared hosts). Peer-cred auth still restricts who can connect.
    """
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    if runtime and Path(runtime).is_dir():
        return f"unix://{Path(runtime) / 'agentm-gw.sock'}"
    return f"unix:///tmp/agentm-gw-{os.geteuid()}.sock"


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
