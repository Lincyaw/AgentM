"""Shared prompt resolver for the auditor / extractor children.

Each domain ships one or more named prompt files under
``<domain>/prompts/<name>.md``. Callers refer to a prompt by:

* **name** — a bare identifier matching a file in the domain's
  prompts directory (e.g. ``"minimal"`` resolves to
  ``audit/auditor/prompts/auditor_minimal.md``).
* **path** — an absolute filesystem path to a ``.md`` file. Useful
  for research A/B without modifying the package.

A name is recognised by having no directory separator. Anything with
``/`` (or that exists on disk as-is) is treated as a path.

The resolver caches reads — prompt files are loaded at most once per
process. They are static research artefacts; reload-on-edit during a
running session is not a requirement.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path


def _domain_dir(domain: str) -> Path:
    """``<audit/>/<domain>/prompts``.

    This module now lives at ``audit/toolkit/`` so the domain dirs
    (``audit/auditor/`` / ``audit/extractor/``) are one level up from
    ``__file__``.
    """
    return Path(__file__).parent.parent / domain / "prompts"


def _resolve_path(domain: str, name_or_path: str, *, filename_prefix: str) -> Path:
    """Resolve a user-supplied name-or-path to an absolute ``.md`` path.

    ``filename_prefix`` is the domain-tag used to qualify named prompts
    on disk (e.g. ``"auditor"`` → ``auditor_minimal.md``).
    """
    candidate = name_or_path.strip()
    if not candidate:
        raise ValueError(f"empty prompt spec for {domain}")

    # Path form: anything with a separator, or that already points at
    # an existing file. Resolve verbatim.
    if "/" in candidate or "\\" in candidate:
        path = Path(candidate).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"prompt file not found: {path}")
        return path

    # Name form: look up under the domain's prompts dir. Try the
    # bare name first, then a ``<prefix>_<name>.md`` form.
    base = _domain_dir(domain)
    for fname in (f"{candidate}.md", f"{filename_prefix}_{candidate}.md"):
        path = base / fname
        if path.is_file():
            return path

    available = sorted(p.name for p in base.glob("*.md"))
    raise FileNotFoundError(f"unknown {domain} prompt {candidate!r}; available: {available}")


@lru_cache(maxsize=64)
def _read(path_str: str) -> str:
    return Path(path_str).read_text(encoding="utf-8")


def load_prompt(domain: str, name_or_path: str, *, filename_prefix: str) -> str:
    """Resolve and return prompt text.

    Cached: repeat calls for the same path skip the disk read.
    """
    path = _resolve_path(domain, name_or_path, filename_prefix=filename_prefix)
    return _read(str(path))


__all__ = ["load_prompt"]
