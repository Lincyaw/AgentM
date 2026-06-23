"""Context atom: injects the extraction system prompt + JSON schema + vocabulary."""
from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Final

from agentm.core.abi import BeforeAgentStartEvent, ExtensionAPI
from agentm.extensions import ExtensionManifest
from pydantic import BaseModel

from .schema import ExtractionResult

_PROMPTS_DIR: Final = Path(__file__).parent / "prompts"
_VOCABULARY_PATH: Final = Path(__file__).resolve().parent.parent / "vocabulary.yaml"


class ExtractorContextConfig(BaseModel):
    prompt_name: str = "default"


MANIFEST = ExtensionManifest(
    name="trajectory_extractor_context",
    description="Inject the symbol-extraction system prompt with JSON schema.",
    registers=(f"event:{BeforeAgentStartEvent.CHANNEL}",),
    config_schema=ExtractorContextConfig,
)


def _load_and_validate_vocabulary() -> dict[str, dict[str, str]]:
    """Load vocabulary.yaml and validate keys match the enum definitions."""
    import yaml

    from trajectory_index.index import ReferenceKind, RelationType, SymbolKind

    data: dict[str, dict[str, str]] = yaml.safe_load(
        _VOCABULARY_PATH.read_text(encoding="utf-8"),
    )

    checks: list[tuple[str, type[StrEnum]]] = [
        ("symbol_kinds", SymbolKind),
        ("reference_kinds", ReferenceKind),
        ("relation_types", RelationType),
    ]
    for section, enum_cls in checks:
        enum_values = {e.value for e in enum_cls}
        yaml_keys = set(data.get(section, {}))
        missing = enum_values - yaml_keys
        extra = yaml_keys - enum_values
        if missing:
            raise ValueError(
                f"vocabulary.yaml [{section}] missing keys: {sorted(missing)}"
            )
        if extra:
            raise ValueError(
                f"vocabulary.yaml [{section}] extra keys: {sorted(extra)}"
            )

    return data


_VOCAB_SECTIONS: tuple[tuple[str, str, bool], ...] = (
    ("symbol_kinds", "Symbol kinds", True),
    ("reference_kinds", "Reference kinds", True),
    ("relation_types", "Relation types", False),
)


def _build_vocabulary_section() -> str:
    """Build the vocabulary section for the extraction prompt."""
    vocab = _load_and_validate_vocabulary()
    lines: list[str] = []
    for section_key, heading, skip_unknown in _VOCAB_SECTIONS:
        lines.append(f"\n## {heading}\n")
        for key, desc in vocab[section_key].items():
            if skip_unknown and key == "unknown":
                continue
            lines.append(f"- `{key}` — {desc}")
    return "\n".join(lines).lstrip("\n")


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
