"""System prompt loader for the v3.1 extractor child session.

The framing text lives in markdown files under
:mod:`audit.extractor.prompts`. Pick a variant by name (default
``"default"``) or by an absolute path. The adapter substitutes
``{TURN_WINDOW_JSON}`` at child-session-spawn time with the JSON list
of the new-turn window — that substitution is the same shape the
prompt file embeds.

Drop in a new variant by adding ``audit/extractor/prompts/extractor_<name>.md``
and pointing the adapter config at it.
"""

from __future__ import annotations

from .._prompt_loader import load_prompt

DEFAULT_PROMPT_NAME = "default"


def load_extractor_prompt(name_or_path: str = DEFAULT_PROMPT_NAME) -> str:
    """Load the extractor framing text for the given variant.

    Result is cached by :func:`_prompt_loader.load_prompt`, so repeated
    calls for the same name skip the disk read.
    """
    return load_prompt("extractor", name_or_path, filename_prefix="extractor")


__all__ = [
    "DEFAULT_PROMPT_NAME",
    "load_extractor_prompt",
]
