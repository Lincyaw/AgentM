"""Agent manifests: the prompt and shape of every model this package runs.

One file per agent under ``agents/``, and one loader, so a manifest field is
either read or absent. A field the code does not read must not appear, or
editing the YAML changes nothing.

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

_AGENTS_DIR = Path(__file__).parent.parent / "runtime" / "agents"


@dataclass(frozen=True, slots=True)
class AgentManifest:
    name: str
    system: str
    #: ``None`` leaves the child unbounded. A reviewer that runs out of turns
    #: mid-investigation returns no verdict at all, which fails open — the
    #: worst of both, paying for the work and discarding the answer.
    max_turns: int | None
    tools: tuple[str, ...]


@lru_cache(maxsize=16)
def load_manifest(name: str, agents_dir: str = "") -> AgentManifest:
    """Read ``<agents_dir>/<name>.yaml``. Raises if missing or with no prompt.

    ``agents_dir`` empty means this package's own ``agents/``. The loop's stages
    keep their manifests beside themselves and pass their directory in, so there
    is one loader rather than one per directory -- which is what stops a
    manifest field being read in one place and ignored in the other.
    """
    directory = Path(agents_dir) if agents_dir else _AGENTS_DIR
    path = directory / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"agent manifest not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):  # code-health: ignore[AM025] -- YAML boundary
        raise TypeError(f"agent manifest is not a mapping: {path}")
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
