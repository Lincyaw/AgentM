"""agentm-channels — multi-channel chat gateway for AgentM.

Architecture (two-queue message bus + channel registry):

    ┌──────────────┐  inbound   ┌──────────────────┐  prompt
    │   Channels   │ ─────────► │   MessageBus     │ ────────► AgentSession
    │  (Feishu,    │            │  (asyncio.Queue) │           per session_key
    │   Slack,     │ ◄─────────┐│                  │ ◄─────── EventBus
    │   …)         │  outbound  │                  │
    └──────────────┘            └──────────────────┘

Adding a new channel = drop a single file under
``agentm_channels.channels.<name>`` extending
:class:`agentm_channels.base.BaseChannel`. Auto-discovery picks it up
from config (``channels.<name>.enabled: true``).
"""

from __future__ import annotations

import os
from pathlib import Path


__version__ = "0.1.0"


def load_dotenv_files(cwd: Path) -> None:
    """Best-effort ``.env`` autoload for any channels CLI.

    Loads (in order, never overriding existing env): ``<cwd>/.env``, then
    the ``.env`` next to the nearest ``[tool.uv.workspace]`` pyproject
    walking up at most 8 levels — same convention the gateway has used
    since v0, now shared so terminal/feishu/worker clients also see
    `LARK_APP_ID`, `OPENAI_API_KEY`, etc. without manual exports.

    No-op when ``python-dotenv`` isn't installed; the import is local
    so this stays import-safe even in minimal client environments.
    """
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
