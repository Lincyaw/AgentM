"""Workspace-root resolution shared by catalog tools."""

from __future__ import annotations

from pathlib import Path

from agentm.core.abi import ExtensionAPI
from agentm.core.lib import expand_path, expand_path_from_cwd


def resolve_root(api: ExtensionAPI, raw_root: str | None) -> Path:
    if raw_root is None:
        return expand_path(api.cwd).resolve()
    return expand_path_from_cwd(raw_root, api.cwd).resolve()


__all__ = ("resolve_root",)
