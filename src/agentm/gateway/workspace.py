"""Per-channel workspace resolution for the gateway.

When ``[gateway] workspace_root`` is set in config.toml, the gateway
resolves channel -> ``{workspace_root}/{channel}/``, auto-creating the
directory on first use.  Explicit ``[gateway.workspaces]`` entries
override the convention path.  When workspace_root is unset, all
channels use the gateway's ``--cwd`` (full backward compat).
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

from loguru import logger


def _agentm_home() -> Path:
    home = os.environ.get("AGENTM_HOME")
    return Path(home) if home else Path.home() / ".agentm"


class WorkspaceResolver:
    """Map channel names to per-channel working directories."""

    def __init__(
        self,
        default_cwd: str,
        workspace_root: str | None = None,
        overrides: dict[str, str] | None = None,
    ) -> None:
        self._default_cwd = default_cwd
        self._root = (
            Path(workspace_root).expanduser().resolve()
            if workspace_root
            else None
        )
        self._overrides = overrides or {}
        self._created: set[str] = set()

    @property
    def active(self) -> bool:
        """True when workspace routing is configured (root or overrides)."""
        return self._root is not None or bool(self._overrides)

    def resolve(self, channel: str) -> str:
        """Return the working directory for *channel*.

        Falls back to *default_cwd* when the channel is empty or no
        workspace routing is configured.
        """
        if not channel:
            return self._default_cwd

        # Explicit override wins
        if channel in self._overrides:
            ws_path = Path(self._overrides[channel]).expanduser().resolve()
        elif self._root is not None:
            # ``channel`` is attacker-controlled (a remote peer stamps it on
            # its inbound), so a ``../`` or absolute path would otherwise
            # escape ``_root`` and point the session's cwd — and its file /
            # bash tools — at arbitrary directories. Resolve and require the
            # result to stay under ``_root``; fall back to the default cwd on
            # any escape rather than honouring the traversal.
            candidate = (self._root / channel).resolve()
            if not candidate.is_relative_to(self._root):
                logger.warning(
                    f"workspace: rejecting out-of-root channel {channel!r} "
                    f"(resolved to {candidate}); using default cwd"
                )
                return self._default_cwd
            ws_path = candidate
        else:
            return self._default_cwd

        cwd = str(ws_path)

        if cwd not in self._created:
            self._created.add(cwd)
            if not ws_path.exists():
                ws_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"created workspace for channel {channel!r}: {ws_path}")

        return cwd


def load_gateway_config(default_cwd: str) -> WorkspaceResolver:
    """Build a :class:`WorkspaceResolver` from ``[gateway]`` in config.toml."""
    path = _agentm_home() / "config.toml"
    if not path.is_file():
        return WorkspaceResolver(default_cwd)

    try:
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
    except Exception:
        logger.opt(exception=True).warning(f"config.toml: failed to parse {path}")
        return WorkspaceResolver(default_cwd)

    gateway = data.get("gateway")
    if not isinstance(gateway, dict):
        return WorkspaceResolver(default_cwd)

    workspace_root = gateway.get("workspace_root")
    if not isinstance(workspace_root, str) or not workspace_root.strip():
        workspace_root = None

    raw_overrides = gateway.get("workspaces")
    overrides: dict[str, str] = {}
    if isinstance(raw_overrides, dict):
        for k, v in raw_overrides.items():
            if isinstance(v, str) and v.strip():
                overrides[k] = v

    return WorkspaceResolver(
        default_cwd=default_cwd,
        workspace_root=workspace_root,
        overrides=overrides,
    )
