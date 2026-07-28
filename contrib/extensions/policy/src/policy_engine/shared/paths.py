"""How a reference to a policy file resolves to a path."""

from __future__ import annotations

from pathlib import Path

#: Package-shipped policy files (``checklist.yaml``, ``vocabulary.yaml``) live
#: beside the runtime that reads them, not beside this resolver.
_PACKAGE_POLICY_DIR = Path(__file__).parent.parent / "runtime"


def resolve_policy_path(file_ref: str, *, cwd: Path) -> Path | None:
    """Resolve a policy file reference: ``package:`` paths ship with this
    package under ``runtime/``; bare paths resolve against ``cwd``, then
    ``~/.agentm/policies``."""

    package_prefix = "package:"
    if file_ref.startswith(package_prefix):
        relative = Path(file_ref.removeprefix(package_prefix))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"invalid package policy path: {file_ref}")
        package_candidate = _PACKAGE_POLICY_DIR / relative
        return package_candidate if package_candidate.exists() else None
    path = Path(file_ref)
    if path.is_absolute() and path.exists():
        return path
    candidate = cwd / file_ref
    if candidate.exists():
        return candidate
    home_candidate = Path.home() / ".agentm" / "policies" / file_ref
    return home_candidate if home_candidate.exists() else None


__all__ = ["resolve_policy_path"]
