"""Data utilities for trajectory-index SFT pipeline.

Export steps from AgentM traces, run teacher extraction, format SFT examples.

CLI usage::

    uv run python -m trajectory_index.data collect \\
        --model doubao \\
        --trace-dir datasets/ops-lite/cases/*/. \\
        --output sft_train.jsonl

    uv run python -m trajectory_index.data export-steps \\
        --trace-file path/to/session.jsonl
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import re
from pathlib import Path
from typing import Annotated, Final, TypedDict, cast

import typer
from loguru import logger

from .agents import extractor_scenario
from .agents.entity_extractor.schema import ReportEntitiesParams

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

type JsonValue = str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]
type ProviderSpec = tuple[str, dict[str, JsonValue]]


class ChatMessage(TypedDict):
    role: str
    content: str


class SftExample(TypedDict):
    messages: list[ChatMessage]

# ---------------------------------------------------------------------------
# Trace message cleaning — keep native format, strip metadata
# ---------------------------------------------------------------------------

_SKIP_ROLES: Final = frozenset({"system"})
_MAX_TEXT_CHARS: Final = 2000
_PAYLOAD_DROP_KEYS: Final = frozenset({
    "usage", "timestamp", "stop_reason", "termination",
})


def _truncate_block(block: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """Truncate long text content within a single content block."""
    btype = block.get("type", "")
    if btype == "text":
        text = block.get("text", "")
        if isinstance(text, str) and len(text) > _MAX_TEXT_CHARS:
            return {**block, "text": text[:_MAX_TEXT_CHARS] + "..."}
    elif btype == "tool_result":
        sub = block.get("content", [])
        if isinstance(sub, list):
            truncated: list[JsonValue] = [_truncate_block(s) for s in sub if isinstance(s, dict)]
            return {**block, "content": truncated}
    return block


def clean_trace_messages(entries: list[dict[str, JsonValue]]) -> list[dict[str, JsonValue]]:
    """Clean trace entries to extraction input: keep id, role, content blocks.

    Works with output from ``TraceReader.load_messages()`` and
    ``clickhouse.session_entries()`` — both produce the same
    ``{type, id, parent_id, timestamp, payload}`` shape.
    """
    out: list[dict[str, JsonValue]] = []
    for entry in entries:
        entry_id = entry.get("id", "")
        payload = entry.get("payload")
        if not isinstance(payload, dict):
            continue
        role = payload.get("role", "")
        if role in _SKIP_ROLES:
            continue

        content = payload.get("content", [])
        if not isinstance(content, list) or not content:
            continue

        blocks: list[JsonValue] = [_truncate_block(b) for b in content if isinstance(b, dict)]
        out.append({"id": entry_id, "role": role, "content": blocks})
    return out


def load_messages_from_trace_file(path: str | Path) -> list[dict[str, JsonValue]]:
    """Load and clean messages from a local session JSONL file."""
    from agentm.core.lib.trace_reader import TraceReader

    tr = TraceReader(str(path))
    return clean_trace_messages(tr.load_messages())


def load_messages_from_session(session_id: str) -> list[dict[str, JsonValue]]:
    """Load and clean messages by session ID from ClickHouse."""
    from agentm.core.observability.clickhouse import get_url, session_entries

    url = get_url()
    if url:
        entries = session_entries(url, session_id)
        if entries:
            return clean_trace_messages(entries)

    logger.debug(f"ClickHouse unavailable for session {session_id}")
    return []


# ---------------------------------------------------------------------------
# Teacher extraction
# ---------------------------------------------------------------------------

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*\n(.*?)\n\s*```", re.DOTALL)


def extract_json(text: str) -> dict[str, JsonValue] | None:
    """Extract a JSON object from model output text."""
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    m = _JSON_BLOCK_RE.search(text)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start : i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        pass
                    break
    return None


def resolve_provider(model_name: str) -> ProviderSpec:
    """Resolve a config.toml model profile to a provider tuple."""
    from agentm.ai import DEFAULT_PROVIDER_DESCRIPTORS
    from agentm.core.lib import resolve_model_profile

    profile = resolve_model_profile(model_name)
    if profile is None:
        raise RuntimeError(f"model profile {model_name!r} not found in config.toml")
    for desc in DEFAULT_PROVIDER_DESCRIPTORS:
        if desc.id == profile.provider and desc.extension_module:
            config = cast(dict[str, JsonValue], dict(profile.to_build_config()))
            return (desc.extension_module, config)
    raise RuntimeError(f"no extension module for provider {profile.provider!r}")


async def extract(
    steps: list[dict[str, JsonValue]],
    provider: ProviderSpec,
) -> ReportEntitiesParams | None:
    """Run the extraction agent with a teacher model and return the result."""
    from agentm.core.abi import AssistantMessage, LoopConfig, TextContent
    from agentm.core.abi.session_config import AgentSessionConfig
    from agentm.core.runtime.session import AgentSession

    scenario = extractor_scenario()
    config = AgentSessionConfig(
        cwd="/tmp",
        provider=provider,
        scenario=scenario,
        purpose="teacher_extraction",
        loop_config=LoopConfig(max_turns=1),
        log_trace_command=False,
    )
    prompt = json.dumps(steps, ensure_ascii=False, indent=2)

    session = await AgentSession.create(config)
    try:
        messages = await session.prompt(prompt)
    finally:
        with contextlib.suppress(Exception):
            await session.shutdown()

    for msg in reversed(messages):
        if not isinstance(msg, AssistantMessage):
            continue
        for block in reversed(msg.content):
            if isinstance(block, TextContent):
                obj = extract_json(block.text)
                if obj:
                    return ReportEntitiesParams.model_validate(obj)
    return None


# ---------------------------------------------------------------------------
# SFT formatting
# ---------------------------------------------------------------------------


SFT_SYSTEM_PROMPT: Final = (
    "Extract entities, mentions, and relations from the trajectory steps. Output JSON only."
)


def load_inference_prompt() -> str:
    """Load the full extraction prompt with JSON schema (for zero-shot inference)."""
    from .agents.entity_extractor.schema import ReportEntitiesParams as Schema

    prompts_dir = Path(__file__).parent / "agents" / "entity_extractor" / "prompts"
    base = (prompts_dir / "default.md").read_text(encoding="utf-8")
    schema = json.dumps(Schema.model_json_schema(), indent=2, ensure_ascii=False)
    return f"{base}\n\n## JSON Schema\n\nYour output must conform to this schema:\n\n```json\n{schema}\n```"


def build_sft_example(
    steps: list[dict[str, JsonValue]],
    teacher_output: ReportEntitiesParams,
    system_prompt: str | None = None,
) -> SftExample:
    """Build one SFT training example in chat-message format.

    Uses a minimal system prompt by default — the model learns the
    extraction schema from the training data, not from verbose instructions.
    """
    if system_prompt is None:
        system_prompt = SFT_SYSTEM_PROMPT

    return SftExample(messages=[
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=json.dumps(steps, ensure_ascii=False)),
        ChatMessage(role="assistant", content=teacher_output.model_dump_json(indent=None)),
    ])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

app = typer.Typer(pretty_exceptions_enable=False)


def _find_trace_files(paths: list[Path]) -> list[Path]:
    """Resolve paths to individual .jsonl trace files.

    Accepts files directly, or directories (searches for
    ``.agentm/observability/*.jsonl`` recursively).
    """
    files: list[Path] = []
    for p in paths:
        if p.is_file() and p.suffix == ".jsonl":
            files.append(p)
        elif p.is_dir():
            files.extend(sorted(p.rglob(".agentm/observability/*.jsonl")))
    return files


@app.command()
def export_messages(
    trace_file: Annotated[Path, typer.Argument(help="Session JSONL file")],
) -> None:
    """Export a single trace file as cleaned messages JSON."""
    msgs = load_messages_from_trace_file(trace_file)
    logger.info(f"{trace_file.name}: {len(msgs)} messages")
    typer.echo(json.dumps(msgs, ensure_ascii=False, indent=2))


@app.command()
def collect(
    paths: Annotated[list[Path] | None, typer.Argument(help="Trace files or directories (omit if using --session)")] = None,
    session: Annotated[list[str] | None, typer.Option(help="Session IDs to load from ClickHouse (repeatable)")] = None,
    model: Annotated[str, typer.Option(help="Model profile from config.toml")] = "doubao",
    output_dir: Annotated[Path, typer.Option("--output", help="HuggingFace dataset output directory")] = Path("data"),
    split: Annotated[str, typer.Option(help="Dataset split name")] = "train",
    concurrency: Annotated[int, typer.Option(help="Max concurrent extraction calls")] = 2,
) -> None:
    """Collect SFT training data: export traces, run extraction, write HF dataset.

    Sources (use one or both):
      - positional paths: local JSONL files or directories
      - --session: session IDs loaded from ClickHouse
    """
    asyncio.run(_collect_async(paths or [], session or [], model, output_dir, split, concurrency))


def _write_hf_dataset(
    examples: list[SftExample],
    output_dir: Path,
    split: str,
) -> None:
    """Write examples as a HuggingFace-compatible dataset directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    data_file = output_dir / f"{split}.jsonl"
    with data_file.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    info = {
        "dataset_info": {
            "description": "SFT training data for trajectory semantic index entity extraction.",
            "features": {
                "messages": {
                    "feature": {
                        "role": {"dtype": "string"},
                        "content": {"dtype": "string"},
                    },
                    "_type": "Sequence",
                },
            },
        },
        "splits": {
            split: {
                "num_examples": len(examples),
                "dataset_name": "trajectory_index_sft",
            },
        },
    }
    (output_dir / "dataset_info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


async def _collect_async(
    paths: list[Path],
    session_ids: list[str],
    model: str,
    output_dir: Path,
    split: str,
    concurrency: int,
) -> None:
    # Build work items: (label, steps_loader) pairs
    work: list[tuple[str, tuple[str, Path | str]]] = []

    for tf in _find_trace_files(paths):
        work.append((tf.name, ("file", tf)))

    for sid in session_ids:
        work.append((sid, ("session", sid)))

    if not work:
        logger.error("no trace files or session IDs provided")
        raise typer.Exit(1)

    provider = resolve_provider(model)
    sem = asyncio.Semaphore(concurrency)

    logger.info(f"model={model} sources={len(work)} output={output_dir} split={split}")

    async def _process_one(label: str, source: tuple[str, Path | str]) -> SftExample | None:
        async with sem:
            kind, ref = source
            if kind == "file":
                msgs = load_messages_from_trace_file(ref)
            else:
                msgs = load_messages_from_session(str(ref))
            if not msgs:
                logger.warning(f"{label}: 0 messages, skipping")
                return None
            try:
                result = await extract(msgs, provider)
            except Exception:
                logger.exception(f"{label}: extraction failed")
                return None
            if result is None:
                logger.warning(f"{label}: no parseable JSON in response")
                return None
            example = build_sft_example(msgs, result)
            logger.info(
                f"{label}: {len(msgs)} msgs -> "
                f"{len(result.entities)} ent, {len(result.mentions)} men, {len(result.relations)} rel"
            )
            return example

    tasks = [_process_one(label, source) for label, source in work]
    results = await asyncio.gather(*tasks)

    examples = [ex for ex in results if ex is not None]
    _write_hf_dataset(examples, output_dir, split)
    logger.info(f"done: {len(examples)}/{len(work)} examples -> {output_dir}/{split}.jsonl")


if __name__ == "__main__":
    app()
