"""Environment loading helpers shared by AgentM CLIs."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

_PACKAGE_WALK_DEPTH = 8


def autoload_dotenv(cwd: Path | None = None) -> None:
    """Load cwd-local and workspace-root ``.env`` files without overriding env."""
    if os.environ.get("AGENTM_SKIP_DOTENV"):
        return

    base = cwd if cwd is not None else Path.cwd()
    try:
        base = base.expanduser().resolve()
    except OSError as exc:
        logger.debug("env: could not resolve dotenv base {}: {}", base, exc)
        base = base.expanduser()

    candidates: list[Path] = [base / ".env"]
    walker = base
    for _ in range(_PACKAGE_WALK_DEPTH):
        manifest = walker / "pyproject.toml"
        if manifest.exists():
            try:
                if "[tool.uv.workspace]" in manifest.read_text(encoding="utf-8"):
                    workspace_env = walker / ".env"
                    if workspace_env != candidates[0]:
                        candidates.append(workspace_env)
                    break
            except OSError as exc:
                logger.debug("env: could not read {} during .env discovery: {}", manifest, exc)
        if walker.parent == walker:
            break
        walker = walker.parent

    for path in candidates:
        if path.is_file():
            load_dotenv(path, override=False)
