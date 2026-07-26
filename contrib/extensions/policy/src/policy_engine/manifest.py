"""Agent manifests: the prompt and shape of every model this package runs.

One file per agent under ``agents/``, and one loader, so a manifest field is
either read or absent. Both manifests previously declared ``max_turns``,
``tools`` and ``result_schema`` while the code read only ``system`` and hard
coded the rest — a reader editing the YAML would have changed nothing.

Tools are named here, not defined here: a tool is a closure over runtime
objects — the sandbox's BashOperations, the sink a verdict lands in — which a
YAML file cannot hold. The manifest says which tools an agent gets; the caller
supplies them by name and finds out at load time if a name is unknown.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

_AGENTS_DIR = Path(__file__).parent / "agents"


@dataclass(frozen=True, slots=True)
class AgentManifest:
    name: str
    system: str
    #: ``None`` leaves the child unbounded. A reviewer that runs out of turns
    #: mid-investigation returns no verdict at all, which fails open — the
    #: worst of both, paying for the work and discarding the answer.
    max_turns: int | None
    tools: tuple[str, ...]


@lru_cache(maxsize=8)
def load_manifest(name: str) -> AgentManifest:
    """Read ``agents/<name>.yaml``. Raises if it is missing or has no prompt."""
    path = _AGENTS_DIR / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"agent manifest not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):  # code-health: ignore[AM025] -- YAML boundary
        raise ValueError(f"agent manifest is not a mapping: {path}")
    system = str(raw.get("system", "")).strip()
    if not system:
        raise ValueError(f"agent manifest has no system prompt: {path}")
    tools = raw.get("tools") or ()
    raw_turns = raw.get("max_turns")
    return AgentManifest(
        name=name,
        system=system,
        max_turns=None if raw_turns is None else int(raw_turns),
        tools=tuple(str(t) for t in tools),
    )


__all__ = ["AgentManifest", "load_manifest"]
