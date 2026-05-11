"""Per-session atom-source override sandbox.

Pulled out of ``AgentSession.create`` so the constitution-layer session
bootstrap delegates override application to a single helper call. The
helper materializes ``atom_source_overrides`` (per-task-evolution loop
§6.3) into ``<cwd>/.agentm/eval-sandbox/<session_id>/`` and rewires each
loaded atom's ``file_path`` so ``ResourceWriter`` classifies the
redirected path as ``unmanaged`` (no git mutation, no working-tree
change). The caller is responsible for tearing the sandbox down on
session shutdown.

Layer rule: this module sits in the harness layer alongside the
reloader. It imports only stdlib + ``agentm.core.runtime.atom_reloader`` +
``agentm.core.abi.events`` so it can be invoked from session
bootstrap without crossing layer boundaries.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Mapping
from pathlib import Path

from agentm.core.abi import EventBus
from agentm.core.abi.events import DiagnosticEvent
from agentm.core.abi.resource import ResourceWriter
from agentm.core.runtime.atom_reloader import AtomReloader

logger = logging.getLogger(__name__)


async def apply_atom_source_overrides(
    *,
    reloader: AtomReloader,
    bus: EventBus,
    resource_writer: ResourceWriter,
    cwd: str,
    session_id: str,
    overrides: Mapping[str, str],
) -> Path | None:
    """Materialize per-session atom-source overrides into a sandbox dir.

    Each override redirects the loaded atom's ``file_path`` to a
    sandbox file under ``.agentm/eval-sandbox/<session_id>/`` and
    issues a ``reload_atom`` so the running module reflects the new
    source. Failures are reported via ``DiagnosticEvent`` and skipped;
    the caller still gets the sandbox dir back so it can be cleaned up
    on shutdown.

    Returns the sandbox directory if any overrides were requested, else
    ``None``.
    """

    if not overrides:
        return None

    sandbox = (Path(cwd) / ".agentm" / "eval-sandbox" / session_id).resolve()
    sandbox.mkdir(parents=True, exist_ok=True)

    for atom_name, new_source in overrides.items():
        atom = reloader.loaded_by_name.get(atom_name)
        if atom is None:
            await bus.emit(
                DiagnosticEvent.CHANNEL,
                DiagnosticEvent(
                    level="error",
                    source="atom_source_overrides",
                    message=(
                        f"override target {atom_name!r} is not loaded; "
                        f"skipping override"
                    ),
                ),
            )
            continue
        sandbox_path = sandbox / f"{atom_name}.py"
        # Seed the sandbox file through the writer so custom ResourceWriter
        # impls (e.g. a Docker sandbox or remote FS) can intercept this side
        # of the eval-sandbox path the same way they see normal atom writes.
        # Sandbox lives under .agentm/eval-sandbox/** which is unmanaged —
        # the writer no-ops the git side and just performs the file write.
        await resource_writer.write(
            str(sandbox_path),
            new_source.encode("utf-8"),
            rationale="atom_source_override seed",
            author="indexer",
        )
        # Redirect the atom's file_path so reload_atom routes its write
        # to the sandbox (path_class=unmanaged → no git).
        atom.file_path = sandbox_path.resolve()
        # Also update the loaded module's ``__file__`` so future reloads
        # pick up the right source.
        loaded_mod = sys.modules.get(atom.module_path)
        if loaded_mod is not None:
            try:
                loaded_mod.__file__ = str(sandbox_path)
            except (AttributeError, TypeError):
                pass
        result = reloader.reload_atom(
            atom_name,
            new_source,
            agent_initiated=False,
            rationale="atom_source_override (eval sandbox)",
        )
        if not result.ok:
            await bus.emit(
                DiagnosticEvent.CHANNEL,
                DiagnosticEvent(
                    level="error",
                    source="atom_source_overrides",
                    message=(
                        f"override of {atom_name!r} failed: {result.error}"
                    ),
                ),
            )

    return sandbox


__all__ = ["apply_atom_source_overrides"]
