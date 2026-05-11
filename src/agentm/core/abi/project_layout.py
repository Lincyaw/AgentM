"""Project layout port — where on-disk artefacts of a session live.

Project layout is the "where" of session state: catalog roots, skill
directories, prompt directories, artefact directories, observability roots.
Atoms reach their on-disk neighbours via this Protocol so that policy
(today: ``<cwd>/.agentm/...``) is replaceable without changing the kernel.

Layer purity: this module is part of ``core.abi``. It defines a Protocol
only — it does not import any harness/extension code and does not touch
the filesystem at import time. The default implementation lives in
``agentm.core.runtime.catalog``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class ProjectLayout(Protocol):
    """Where the session looks for persistent on-disk state.

    All methods return :class:`pathlib.Path` objects rooted under the
    session's effective workspace. They MUST NOT create directories at
    import time; callers ``mkdir`` lazily as needed.
    """

    def catalog_root(self) -> Path:
        """Catalog root (atoms/<name>/<version>/{runs,metrics,...})."""
        ...

    def skills_dirs(self) -> list[Path]:
        """Project-scope skills directories searched by the loader."""
        ...

    def artifacts_root(self, session_id: str) -> Path:
        """Per-session artifacts root."""
        ...

    def prompts_dirs(self) -> list[Path]:
        """Project-scope prompt-template directories."""
        ...

    def observability_root(self) -> Path:
        """Where session traces (``<trace>.jsonl``) are written."""
        ...


__all__ = ["ProjectLayout"]
