"""Prompt template loading via Jinja2."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader


def load_prompt_template(
    path: Path | str, *, base_dir: Path | None = None, **context: Any
) -> str:
    """Load a Jinja2 prompt template and render it with the given context.

    Uses ``FileSystemLoader`` so templates can use ``{% include %}`` to
    compose shared fragments.  The loader root is the directory containing
    *path*; the template name is the file's basename.

    Args:
        path: Path to the .j2 template file.
        base_dir: If provided, validates that path resolves within this directory.
        **context: Template variables to render.

    Returns:
        Rendered prompt string.
    """
    resolved = Path(path).resolve()
    if base_dir is not None:
        try:
            resolved.relative_to(base_dir.resolve())
        except ValueError:
            raise ValueError(
                f"Prompt path {path} resolves outside base directory {base_dir}"
            ) from None
    env = Environment(
        loader=FileSystemLoader(str(resolved.parent)),
        keep_trailing_newline=True,
    )
    template = env.get_template(resolved.name)
    return template.render(**context)
