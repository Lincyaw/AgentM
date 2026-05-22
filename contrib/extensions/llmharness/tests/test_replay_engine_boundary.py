"""Fail-stop: no atom imports from the ``llmharness.tools.*`` host-driver modules.

``tools/engine.py`` and ``tools/prefix_replay.py`` directly import
``agentm.core.runtime.*`` because they spawn standalone / branched
sessions for offline replay. Those imports are legal in host-side
driver modules shipped inside the package, but they MUST NOT leak
into any atom — if they did, every atom in this package would
transitively pull ``core.runtime`` and silently violate the §11 atom
contract.

Atoms are identified by exporting a top-level ``MANIFEST`` symbol of
type :class:`agentm.extensions.ExtensionManifest`. This test walks
every ``.py`` file under ``src/llmharness/`` looking for one of those,
and asserts none of them import from ``llmharness.tools.engine`` or
``llmharness.tools.prefix_replay`` (directly or via relative path).

Why fail-stop: if a future atom adds
``from ..tools.{engine,prefix_replay} import …`` and CI ever runs
``ruff`` clean, the boundary is gone. This test is the only structural
barrier.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_SRC_ROOT = Path(__file__).parent.parent / "src" / "llmharness"

_ENGINE_TARGETS = {
    "llmharness.tools.engine",
    "llmharness.tools.prefix_replay",
    "..tools.engine",
    "..tools.prefix_replay",
}


def _module_has_manifest(path: Path) -> bool:
    """True iff the file declares a top-level ``MANIFEST = ...`` assignment."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return False
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "MANIFEST":
                    return True
        elif isinstance(node, ast.AnnAssign) and (
            isinstance(node.target, ast.Name) and node.target.id == "MANIFEST"
        ):
            return True
    return False


def _imports_replay_engine(path: Path) -> list[str]:
    """Return the offending import lines (if any). Captures both
    absolute and relative imports of ``tools.engine`` /
    ``tools.prefix_replay`` at any depth — either module reaches
    ``agentm.core.runtime.*`` directly and would taint any atom that
    pulls it in transitively."""
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"^(?:from|import)\s+[^\n]*?\btools\.(?:engine|prefix_replay)\b",
        re.MULTILINE,
    )
    return pattern.findall(text)


@pytest.mark.parametrize(
    "atom_path",
    [p for p in _SRC_ROOT.rglob("*.py") if _module_has_manifest(p)],
    ids=lambda p: str(p.relative_to(_SRC_ROOT)),
)
def test_atom_does_not_import_replay_engine(atom_path: Path) -> None:
    offending = _imports_replay_engine(atom_path)
    assert not offending, (
        f"{atom_path.relative_to(_SRC_ROOT)} declares MANIFEST and "
        f"imports a tools/ host-driver module (engine / prefix_replay) — "
        f"this would pull agentm.core.runtime into an atom. "
        f"Offending lines: {offending}"
    )


def test_at_least_one_atom_discovered() -> None:
    """Sanity check: if the discovery walker breaks, all atom tests
    above silently parametrize to zero cases and pass. Guard against
    that."""
    atoms = [p for p in _SRC_ROOT.rglob("*.py") if _module_has_manifest(p)]
    assert len(atoms) >= 5, (
        f"expected at least 5 MANIFEST-bearing files; found {len(atoms)}. "
        "Has the atom-discovery walker drifted?"
    )
