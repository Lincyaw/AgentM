"""Auto-discovery of channel classes.

Built-ins live under ``agentm_channels.channels.<name>``; external
plugins register via the ``agentm_channels.channels`` entry-point group
in their own ``pyproject.toml``::

    [project.entry-points."agentm_channels.channels"]
    slack = "my_pkg.slack:SlackChannel"

Built-ins win on name collisions — a third-party plugin can't shadow
``feishu``.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from typing import TYPE_CHECKING


logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from .base import BaseChannel


def _builtin_module_names() -> list[str]:
    import agentm_channels.channels as pkg

    return [name for _, name, ispkg in pkgutil.iter_modules(pkg.__path__) if not ispkg]


def load_builtin(module_name: str) -> type[BaseChannel]:
    from .base import BaseChannel as _Base

    mod = importlib.import_module(f"agentm_channels.channels.{module_name}")
    for attr in dir(mod):
        obj = getattr(mod, attr)
        if isinstance(obj, type) and issubclass(obj, _Base) and obj is not _Base:
            return obj  # type: ignore[return-value]
    raise ImportError(
        f"agentm_channels.channels.{module_name} has no BaseChannel subclass"
    )


def _entry_point_plugins() -> dict[str, type[BaseChannel]]:
    from importlib.metadata import entry_points

    out: dict[str, type[BaseChannel]] = {}
    try:
        eps = entry_points(group="agentm_channels.channels")
    except TypeError:  # Python <3.10 shim — not relevant for our 3.12+ floor.
        return out
    for ep in eps:
        try:
            cls = ep.load()
            out[ep.name] = cls
        except Exception:
            logger.exception("failed to load channel plugin %r", ep.name)
    return out


def discover_all() -> dict[str, type[BaseChannel]]:
    """Return ``{name: BaseChannel-subclass}`` for every available channel."""
    builtin: dict[str, type[BaseChannel]] = {}
    for modname in _builtin_module_names():
        try:
            builtin[modname] = load_builtin(modname)
        except ImportError:
            logger.exception("skip built-in channel %r", modname)
    plugins = _entry_point_plugins()
    shadowed = set(plugins) & set(builtin)
    if shadowed:
        logger.warning("plugins shadowed by built-ins (ignored): %s", shadowed)
    return {**plugins, **builtin}
