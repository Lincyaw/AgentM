"""Environment loading helpers shared by AgentM CLIs."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values
from loguru import logger

from agentm.core.lib.user_config import agentm_home_dir

_PACKAGE_WALK_DEPTH = 8
_external_env_keys: set[str] | None = None
_loaded_dotenv_values: dict[str, str] = {}


def autoload_dotenv(cwd: Path | None = None) -> None:
    """Load AgentM ``.env`` files without overriding existing environment.

    Precedence follows candidate order within one call: cwd-local,
    workspace-root, then ``$AGENTM_HOME/.env`` as machine/user defaults. Across
    multiple calls in the same process, a later call may replace values loaded
    by an earlier call, but never values that came from the real process
    environment.
    """
    if os.environ.get("AGENTM_SKIP_DOTENV"):
        return
    global _external_env_keys
    if _external_env_keys is None:
        _external_env_keys = set(os.environ)

    base = cwd if cwd is not None else Path.cwd()
    try:
        base = base.expanduser().resolve()
    except OSError as exc:
        logger.debug("env: could not resolve dotenv base {}: {}", base, exc)
        base = base.expanduser()

    local_candidates: list[Path] = [base / ".env"]
    walker = base
    for _ in range(_PACKAGE_WALK_DEPTH):
        manifest = walker / "pyproject.toml"
        if manifest.exists():
            try:
                if "[tool.uv.workspace]" in manifest.read_text(encoding="utf-8"):
                    workspace_env = walker / ".env"
                    if workspace_env != local_candidates[0]:
                        local_candidates.append(workspace_env)
                    break
            except OSError as exc:
                logger.debug("env: could not read {} during .env discovery: {}", manifest, exc)
        if walker.parent == walker:
            break
        walker = walker.parent

    values = _dotenv_values(local_candidates)
    home_env = _effective_agentm_home(values) / ".env"
    candidates = list(local_candidates)
    if home_env not in candidates:
        candidates.append(home_env)
        values = _dotenv_values(candidates)

    external_keys = _external_env_keys or set()
    for key, value in values.items():
        if value is None:
            continue
        current = os.environ.get(key)
        loaded_value = _loaded_dotenv_values.get(key)
        if (
            current is not None
            and key in external_keys
            and loaded_value is None
        ):
            continue
        if current is not None and loaded_value is not None and current != loaded_value:
            continue
        os.environ[key] = value
        _loaded_dotenv_values[key] = value


def _dotenv_values(candidates: list[Path]) -> dict[str, str | None]:
    values: dict[str, str | None] = {}
    for path in candidates:
        if path.is_file():
            for key, value in dotenv_values(path).items():
                values.setdefault(key, value)
    return values


def _effective_agentm_home(dotenv_values_by_key: dict[str, str | None]) -> Path:
    external_keys = _external_env_keys or set()
    if "AGENTM_HOME" in external_keys and "AGENTM_HOME" not in _loaded_dotenv_values:
        return agentm_home_dir()
    home = dotenv_values_by_key.get("AGENTM_HOME")
    if home:
        return Path(home).expanduser()
    return agentm_home_dir()
