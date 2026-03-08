"""Prompt template loading via Jinja2."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Template


def load_prompt_template(path: Path | str, **context: Any) -> str:
    """Load a Jinja2 prompt template and render it with the given context.

    Args:
        path: Path to the .j2 template file.
        **context: Template variables to render.

    Returns:
        Rendered prompt string.
    """
    text = Path(path).read_text(encoding="utf-8")
    return Template(text).render(**context)
