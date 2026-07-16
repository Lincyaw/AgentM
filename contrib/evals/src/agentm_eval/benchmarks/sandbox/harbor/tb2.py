"""Terminal-Bench 2.0 adapter (github.com/terminal-bench/terminal-bench).

Standard Harbor task layout. Each task.toml declares its own
``docker_image`` (e.g. ``alexgshaw/chess-best-move:20251031``), so
``--source-images`` semantics apply by default.  The Guangzhou registry
mirror pulls docker.io images transparently.

How to run::

    uv run agentm-eval sandbox batch --bench tb2 --model litellm-dsv4flash -j 5

Repo auto-clones to ``$AGENTM_HOME/bench-repos/tb2`` on first run.
Override with ``--repo <path>`` or ``$TB2_REPO``.
Images: ``pair-cn-guangzhou.cr.volces.com/{source_image}``.
"""

from __future__ import annotations

from .base import HarborAdapter


class Tb2Adapter(HarborAdapter):
    """Terminal-Bench 2.0 adapter."""

    DEFAULT_REGISTRY = "pair-cn-guangzhou.cr.volces.com"
    DEFAULT_SOURCE_IMAGES = True
    DEFAULT_REPO_URL = "https://github.com/terminal-bench/terminal-bench.git"
    DEFAULT_REPO_SUBDIR = ""
    DEFAULT_REPO_ENV = "TB2_REPO"
