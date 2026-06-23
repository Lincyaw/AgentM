"""Context atom: injects the extraction system prompt + JSON schema + vocabulary."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Final

from agentm.core.abi import BeforeAgentStartEvent, ExtensionAPI
from agentm.extensions import ExtensionManifest
from pydantic import BaseModel

from .schema import ExtractionResult

_PROMPTS_DIR: Final = Path(__file__).parent / "prompts"


class ExtractorContextConfig(BaseModel):
    prompt_name: str = "default"


MANIFEST = ExtensionManifest(
    name="trajectory_extractor_context",
    description="Inject the symbol-extraction system prompt with JSON schema.",
    registers=(f"event:{BeforeAgentStartEvent.CHANNEL}",),
    config_schema=ExtractorContextConfig,
)


def _build_vocabulary_section() -> str:
    """Build the vocabulary section from the description dicts in index.py."""
    from trajectory_index.index import (
        REFERENCE_KIND_DESCRIPTIONS,
        RELATION_TYPE_DESCRIPTIONS,
        SYMBOL_KIND_DESCRIPTIONS,
    )

    lines: list[str] = []

    lines.append("## Symbol kinds\n")
    for key, desc in SYMBOL_KIND_DESCRIPTIONS.items():
        if key != "unknown":
            lines.append(f"- `{key}` — {desc}")

    lines.append("\n## Reference kinds\n")
    for key, desc in REFERENCE_KIND_DESCRIPTIONS.items():
        if key != "unknown":
            lines.append(f"- `{key}` — {desc}")

    lines.append("\n## Relation types\n")
    for key, desc in RELATION_TYPE_DESCRIPTIONS.items():
        lines.append(f"- `{key}` — {desc}")

    return "\n".join(lines)


def install(api: ExtensionAPI, config: ExtractorContextConfig) -> None:
    prompt_path = _PROMPTS_DIR / f"{config.prompt_name}.md"
    if not prompt_path.is_file():
        raise ValueError(f"prompt not found: {prompt_path}")
    base_prompt = prompt_path.read_text(encoding="utf-8")

    vocabulary = _build_vocabulary_section()
    schema = json.dumps(
        ExtractionResult.model_json_schema(),
        indent=2,
        ensure_ascii=False,
    )
    full_prompt = (
        f"{base_prompt}\n\n"
        f"{vocabulary}\n\n"
        f"## JSON Schema\n\nYour output must conform to this schema:\n\n```json\n{schema}\n```"
    )

    def _inject_system(event: BeforeAgentStartEvent) -> dict[str, str] | None:
        return {"system": full_prompt}

    api.on(BeforeAgentStartEvent.CHANNEL, _inject_system)
