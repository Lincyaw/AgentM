"""TEL context atom — builds system prompt and populates the span store."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

from agentm.core.abi import BeforeAgentStartEvent, ExtensionAPI
from agentm.extensions import ExtensionManifest
from pydantic import BaseModel

from .tools import SPAN_STORE_SERVICE_KEY, SpanStore

_PROMPTS_DIR: Final = Path(__file__).parent / "prompts"


def load_tel_prompt(name: str = "default") -> str:
    md = _PROMPTS_DIR / f"{name}.md"
    if md.is_file():
        return md.read_text(encoding="utf-8")
    path = Path(name).expanduser()
    if path.is_file():
        return path.read_text(encoding="utf-8")
    available = sorted(p.stem for p in _PROMPTS_DIR.glob("*.md"))
    raise ValueError(f"unknown TEL prompt {name!r}; available: {available}")


def build_tel_system_prompt(
    *,
    question: str,
    n_spans: int,
    base_prompt: str | None = None,
) -> str:
    framing = base_prompt if base_prompt is not None else load_tel_prompt("default")
    sections: list[str] = [framing.rstrip(), ""]

    sections.append("## QUESTION")
    sections.append("The agent was asked:")
    sections.append(question)
    sections.append("")

    sections.append("## TRAJECTORY")
    sections.append(
        f"The trajectory has {n_spans} spans. "
        "Use `list_spans` to see all spans, then `get_span` to read specific ones."
    )
    sections.append("")

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Atom
# ---------------------------------------------------------------------------


class TelContextConfig(BaseModel):
    question: str = ""
    spans: list[dict[str, Any]] = []
    stages: dict[str, str] = {}
    prompt_name: str = "default"
    config_file: str = ""


MANIFEST = ExtensionManifest(
    name="tel_context",
    description="Build the TEL system prompt and populate the span store.",
    registers=(f"service:{SPAN_STORE_SERVICE_KEY}",),
    config_schema=TelContextConfig,
)


def install(api: ExtensionAPI, config: TelContextConfig) -> None:
    import json as _json

    question = config.question
    spans = list(config.spans)
    stages = dict(config.stages)

    if config.config_file:
        data = _json.loads(Path(config.config_file).read_text(encoding="utf-8"))
        question = question or data.get("question", "")
        spans = spans or data.get("spans", [])
        stages = stages or data.get("stages", {})

    store = SpanStore(spans=spans, stages=stages)
    api.set_service(SPAN_STORE_SERVICE_KEY, store)

    base_prompt = load_tel_prompt(config.prompt_name)
    prompt_text = build_tel_system_prompt(
        question=question,
        n_spans=len(spans),
        base_prompt=base_prompt,
    )

    def _before_start(event: BeforeAgentStartEvent) -> dict[str, str]:
        return {"system": prompt_text}

    api.on(BeforeAgentStartEvent.CHANNEL, _before_start)
