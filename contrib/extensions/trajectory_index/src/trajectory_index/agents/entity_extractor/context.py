"""Context atom: injects the extraction system prompt + JSON schema + vocabulary."""
from __future__ import annotations

import json
import os
from importlib.resources import files
from pathlib import Path
from typing import Any, Final

import yaml
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


def _resolve_vocabulary_file() -> str:
    """Resolve vocabulary filename from TRAJ_INDEX_VOCABULARY env var."""
    name = os.environ.get("TRAJ_INDEX_VOCABULARY", "default")
    return "vocabulary.yaml" if name == "default" else f"vocabulary.{name}.yaml"


def _build_vocabulary_section() -> str:
    """Build the vocabulary section from the selected vocabulary yaml."""
    vocab_text = files("trajectory_index").joinpath(_resolve_vocabulary_file()).read_text(encoding="utf-8")
    vocab: dict[str, Any] = yaml.safe_load(vocab_text)

    sections: list[tuple[str, str, bool]] = [
        ("symbol_kinds", "Symbol kinds", True),
        ("reference_kinds", "Reference kinds", True),
        ("relation_types", "Relation types", False),
    ]
    lines: list[str] = []
    for key, heading, skip_unknown in sections:
        lines.append(f"## {heading}\n")
        for name, desc in vocab.get(key, {}).items():
            if skip_unknown and name == "unknown":
                continue
            lines.append(f"- `{name}` — {desc}")
        lines.append("")
    return "\n".join(lines).strip()


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
