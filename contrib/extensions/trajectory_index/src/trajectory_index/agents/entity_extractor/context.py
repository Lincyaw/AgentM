"""Context atom: injects the extraction system prompt + JSON schema + vocabulary."""
from __future__ import annotations

import json
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
    vocabulary: str = "default"


MANIFEST = ExtensionManifest(
    name="trajectory_extractor_context",
    description="Inject the symbol-extraction system prompt with JSON schema.",
    registers=(f"event:{BeforeAgentStartEvent.CHANNEL}",),
    config_schema=ExtractorContextConfig,
)


def _vocabulary_filename(name: str = "default") -> str:
    return "vocabulary.yaml" if name == "default" else f"vocabulary.{name}.yaml"


def _build_vocabulary_section(vocabulary: str = "default") -> str:
    """Build the vocabulary section from the selected vocabulary yaml.

    Only symbol_kinds are injected — the extractor outputs symbols, not
    references or relations (those are code-generated), so injecting
    reference_kinds/relation_types would add noise for the extraction model.
    """
    vocab_text = files("trajectory_index").joinpath(_vocabulary_filename(vocabulary)).read_text(encoding="utf-8")
    vocab: dict[str, Any] = yaml.safe_load(vocab_text)

    lines: list[str] = ["## Symbol kinds\n"]
    for name, desc in vocab.get("symbol_kinds", {}).items():
        if name == "unknown":
            continue
        lines.append(f"- `{name}` — {desc}")
    lines.append("")
    return "\n".join(lines).strip()


def install(api: ExtensionAPI, config: ExtractorContextConfig) -> None:
    prompt_path = _PROMPTS_DIR / f"{config.prompt_name}.md"
    if not prompt_path.is_file():
        raise ValueError(f"prompt not found: {prompt_path}")
    base_prompt = prompt_path.read_text(encoding="utf-8")

    vocabulary = _build_vocabulary_section(config.vocabulary)
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

    def _inject_system(event: BeforeAgentStartEvent) -> None:
        event.system = full_prompt

    api.on(BeforeAgentStartEvent.CHANNEL, _inject_system)
