"""Authoring artifact helpers for prompts, skills, and scenario snippets."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from agentm.presenter.frontmatter import FrontmatterDocument


@dataclass(frozen=True, slots=True)
class AuthoringArtifact:
    path: str
    body: str
    metadata: Mapping[str, str | int | float | bool | None] = field(default_factory=dict)

    def render(self) -> str:
        return FrontmatterDocument(metadata=self.metadata, body=self.body).render()


def write_authoring_artifact(root: str | Path, artifact: AuthoringArtifact) -> Path:
    path = _resolve_inside(Path(root), artifact.path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(artifact.render(), encoding="utf-8")
    return path


def _resolve_inside(root: Path, path: str) -> Path:
    candidate = (root / path).expanduser().resolve()
    real_root = root.expanduser().resolve()
    if candidate != real_root and real_root not in candidate.parents:
        raise ValueError(f"artifact path escapes root: {path}")
    return candidate


__all__ = ["AuthoringArtifact", "write_authoring_artifact"]
