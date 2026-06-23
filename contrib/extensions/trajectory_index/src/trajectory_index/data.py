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
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, Final, TypedDict, cast

import typer
from loguru import logger

from .agents import extractor_scenario
from .agents.entity_extractor.schema import ExtractionResult

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


_BLOCK_DROP_KEYS: Final = frozenset({
    "id", "tool_call_id", "signature",
})


def _truncate_block(block: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """Truncate long text content and strip IDs that confuse small models."""
    out = {k: v for k, v in block.items() if k not in _BLOCK_DROP_KEYS}
    btype = out.get("type", "")
    if btype == "text":
        text = out.get("text", "")
        if isinstance(text, str) and len(text) > _MAX_TEXT_CHARS:
            return {**out, "text": text[:_MAX_TEXT_CHARS] + "..."}
    elif btype == "tool_result":
        sub = out.get("content", [])
        if isinstance(sub, list):
            truncated: list[JsonValue] = [_truncate_block(s) for s in sub if isinstance(s, dict)]
            return {**out, "content": truncated}
    return out


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


_stream_debug: bool = False


def _install_stream_tap(session: Any) -> None:
    """Hook into a child session's bus to print streaming deltas in real-time."""
    import sys

    from agentm.core.abi.events import StreamDeltaEvent
    from agentm.core.abi.stream import TextDelta, ThinkingDelta, ToolCallStart

    prefix = {"thinking": False}

    def _on_delta(event: StreamDeltaEvent) -> None:
        d = event.delta
        if isinstance(d, ThinkingDelta):
            if not prefix["thinking"]:
                sys.stderr.write("\n[thinking] ")
                prefix["thinking"] = True
            sys.stderr.write(d.text)
            sys.stderr.flush()
        elif isinstance(d, TextDelta):
            if prefix["thinking"]:
                sys.stderr.write("\n[output] ")
                prefix["thinking"] = False
            sys.stderr.write(d.text)
            sys.stderr.flush()
        elif isinstance(d, ToolCallStart):
            sys.stderr.write(f"\n[tool_call: {d.name}] ")
            sys.stderr.flush()

    session.bus.on(StreamDeltaEvent.CHANNEL, _on_delta)


def _reindex_messages(
    msgs: list[dict[str, JsonValue]],
) -> tuple[list[dict[str, JsonValue]], dict[str, str]]:
    """Replace message IDs with sequential integers for the extraction prompt.

    Returns (reindexed_messages, id_map) where id_map maps "0","1",... back
    to the original IDs.
    """
    id_map: dict[str, str] = {}
    out: list[dict[str, JsonValue]] = []
    for i, msg in enumerate(msgs):
        orig = str(msg.get("id", f"s{i}"))
        id_map[str(i)] = orig
        out.append({**msg, "id": str(i)})
    return out, id_map


def _remap_turn_ids(result: ExtractionResult, id_map: dict[str, str]) -> None:
    """Map sequential turn_ids back to original message IDs."""
    for ref in result.references:
        ref.turn_id = id_map.get(ref.turn_id, ref.turn_id)
    for rel in result.relations:
        rel.turn_id = id_map.get(rel.turn_id, rel.turn_id)


def _try_parse_response(messages: list[Any]) -> tuple[ExtractionResult | None, str | None]:
    """Try to parse an ExtractionResult from assistant messages.

    Returns (result, error_text). If parsing succeeds error_text is None;
    if it fails, error_text describes the failure for retry.
    """
    from agentm.core.abi import AssistantMessage, TextContent

    for msg in reversed(messages):
        if not isinstance(msg, AssistantMessage):
            continue
        for block in reversed(msg.content):
            if not isinstance(block, TextContent):
                continue
            obj = extract_json(block.text)
            if obj:
                try:
                    return ExtractionResult.model_validate(obj), None
                except Exception as exc:
                    return None, f"JSON keys={list(obj.keys())}; validation error: {exc}"
            return None, f"Could not extract JSON from response ({len(block.text)} chars)"
    return None, "No assistant text in response"


async def extract(
    steps: list[dict[str, JsonValue]],
    provider: ProviderSpec,
    registry: list[dict[str, Any]] | None = None,
) -> ExtractionResult | None:
    """Run the extraction agent with a teacher model and return the result."""
    from agentm.core.abi import LoopConfig
    from agentm.core.abi.session_config import AgentSessionConfig
    from agentm.core.runtime.session import AgentSession

    reindexed, id_map = _reindex_messages(steps)

    scenario = extractor_scenario()
    config = AgentSessionConfig(
        cwd="/tmp",
        provider=provider,
        scenario=scenario,
        purpose="teacher_extraction",
        loop_config=LoopConfig(max_turns=1),
        log_trace_command=False,
    )
    if registry:
        prompt_data: dict[str, Any] = {"known_symbols": registry, "messages": reindexed}
        prompt = json.dumps(prompt_data, ensure_ascii=False, indent=2)
    else:
        prompt = json.dumps(reindexed, ensure_ascii=False, indent=2)

    session = await AgentSession.create(config)
    if _stream_debug:
        _install_stream_tap(session)
    try:
        messages = await session.prompt(prompt)
    finally:
        with contextlib.suppress(Exception):
            await session.shutdown()

    result, error = _try_parse_response(messages)
    if result:
        _remap_turn_ids(result, id_map)
        return result

    if not error:
        return None

    # Retry: send the error back to a fresh session
    logger.warning(f"extraction failed, retrying: {error}")
    retry_prompt = (
        f"Your previous output failed validation:\n{error}\n\n"
        f"Here is the input again:\n{prompt}\n\n"
        "Fix the errors and output valid JSON only."
    )
    retry_session = await AgentSession.create(config)
    if _stream_debug:
        _install_stream_tap(retry_session)
    try:
        retry_messages = await retry_session.prompt(retry_prompt)
    finally:
        with contextlib.suppress(Exception):
            await retry_session.shutdown()

    result, retry_error = _try_parse_response(retry_messages)
    if result:
        _remap_turn_ids(result, id_map)
        return result

    logger.warning(f"retry also failed: {retry_error}")
    return None


def _chunk_messages(
    messages: list[dict[str, JsonValue]],
    chunk_size: int,
) -> list[list[dict[str, JsonValue]]]:
    """Split messages into chunks, keeping tool_call/tool_result pairs intact."""
    if len(messages) <= chunk_size:
        return [messages]

    chunks: list[list[dict[str, JsonValue]]] = []
    start = 0

    while start < len(messages):
        end = start + chunk_size
        if end >= len(messages):
            chunks.append(messages[start:])
            break
        # Advance past tool_results so we don't split a pair
        while end < len(messages) and str(messages[end].get("role", "")) == "tool_result":
            end += 1
        chunks.append(messages[start:end])
        start = end

    return chunks


type OnChunkCallback = Callable[[dict[str, Any], ExtractionResult, list[dict[str, JsonValue]]], None]


async def extract_incremental(
    messages: list[dict[str, JsonValue]],
    provider: ProviderSpec,
    chunk_size: int = 20,
    on_chunk: OnChunkCallback | None = None,
) -> list[tuple[dict[str, Any], ExtractionResult]]:
    """Extract symbols incrementally in chunks with registry accumulation.

    Returns (prompt_input, result) per successful chunk.
    If ``on_chunk`` is provided, it is called after each successful chunk
    with (prompt_input, result) — useful for live index updates.
    """
    from .index import normalize_name

    chunks = _chunk_messages(messages, chunk_size)
    registry: list[dict[str, Any]] = []
    seen: set[str] = set()
    results: list[tuple[dict[str, Any], ExtractionResult]] = []

    for i, chunk in enumerate(chunks):
        chunk_registry = list(registry) if registry else None

        try:
            result = await extract(chunk, provider, registry=chunk_registry)
        except Exception:
            logger.exception(f"chunk {i+1}/{len(chunks)} ({len(chunk)} msgs) extraction failed")
            continue
        if result is None:
            logger.warning(f"chunk {i+1}/{len(chunks)} ({len(chunk)} msgs) no parseable JSON")
            continue

        reindexed, _ = _reindex_messages(chunk)
        if chunk_registry:
            prompt_input: dict[str, Any] = {"known_symbols": chunk_registry, "messages": reindexed}
        else:
            prompt_input = {"messages": reindexed}
        results.append((prompt_input, result))

        if on_chunk:
            on_chunk(prompt_input, result, chunk)

        for sym in result.symbols:
            norm = normalize_name(sym.name)
            if norm not in seen:
                seen.add(norm)
                entry: dict[str, Any] = {"name": sym.name, "kind": sym.kind}
                if sym.aliases:
                    entry["aliases"] = sym.aliases
                registry.append(entry)

    return results


# ---------------------------------------------------------------------------
# SFT formatting
# ---------------------------------------------------------------------------


SFT_SYSTEM_PROMPT: Final = (
    "Extract symbols, references, and relations from the trajectory steps. Output JSON only."
)


def load_inference_prompt() -> str:
    """Load the full extraction prompt with JSON schema (for zero-shot inference)."""
    from .agents.entity_extractor.schema import ExtractionResult as Schema

    prompts_dir = Path(__file__).parent / "agents" / "entity_extractor" / "prompts"
    base = (prompts_dir / "default.md").read_text(encoding="utf-8")
    schema = json.dumps(Schema.model_json_schema(), indent=2, ensure_ascii=False)
    return f"{base}\n\n## JSON Schema\n\nYour output must conform to this schema:\n\n```json\n{schema}\n```"


def build_sft_example(
    input_data: list[dict[str, JsonValue]] | dict[str, Any],
    teacher_output: ExtractionResult,
    system_prompt: str | None = None,
) -> SftExample:
    """Build one SFT training example in chat-message format.

    ``input_data`` can be a plain message list (full mode) or a dict with
    ``known_symbols`` and ``messages`` keys (incremental mode).
    """
    if system_prompt is None:
        system_prompt = SFT_SYSTEM_PROMPT

    return SftExample(messages=[
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=json.dumps(input_data, ensure_ascii=False)),
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
    index_output: Annotated[Path | None, typer.Option("--index-output", help="Write final index JSON to this path")] = None,
    split: Annotated[str, typer.Option(help="Dataset split name")] = "train",
    chunk_size: Annotated[int, typer.Option(help="Messages per extraction chunk")] = 20,
    concurrency: Annotated[int, typer.Option(help="Max concurrent traces")] = 2,
    debug: Annotated[bool, typer.Option("--debug", help="Enable debug logging")] = False,
) -> None:
    """Collect SFT training data: export traces, run incremental extraction, write HF dataset.

    Sources (use one or both):
      - positional paths: local JSONL files or directories
      - --session: session IDs loaded from ClickHouse
    """
    if debug:
        global _stream_debug
        _stream_debug = True
        import sys
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    asyncio.run(_collect_async(
        paths or [], session or [], model, output_dir, index_output,
        split, chunk_size, concurrency,
    ))


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
            "description": "SFT training data for trajectory semantic index symbol extraction.",
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


type ChunkResults = list[tuple[dict[str, Any], ExtractionResult]]


def _build_index_from_chunks_into(
    index: Any,
    prompt_input: dict[str, Any],
    result: ExtractionResult,
) -> None:
    """Populate a TrajectoryIndex with one chunk's extraction result."""
    from .index import (
        ReferenceKind,
        RelationType,
        Step,
        StepRole,
        Symbol,
        SymbolKind,
    )

    role_map = {"user": StepRole.USER, "assistant": StepRole.ASSISTANT, "tool_result": StepRole.TOOL_RESULT}
    msgs = prompt_input.get("messages", prompt_input if isinstance(prompt_input, list) else [])
    base_idx = len(index.steps)

    steps_by_id: dict[str, Step] = {}
    for i, msg in enumerate(msgs):
        mid = str(msg.get("id", f"s{base_idx + i}"))
        role = role_map.get(str(msg.get("role", "")), StepRole.USER)
        step = Step(run_id="", step_id=mid, index=base_idx + i, role=role, content="")
        index.add_step(step)
        steps_by_id[mid] = step

    def _resolve(name: str, local: dict[str, Symbol]) -> Symbol | None:
        return local.get(name) or index.resolve_symbol_by_name(name)

    symbol_map: dict[str, Symbol] = {}
    for ext_sym in result.symbols:
        try:
            kind = SymbolKind(ext_sym.kind.lower())
        except ValueError:
            kind = SymbolKind.UNKNOWN
        symbol = index.upsert_symbol(name=ext_sym.name, kind=kind, summary=ext_sym.summary, aliases=ext_sym.aliases)
        symbol_map[ext_sym.name] = symbol

    for ref in result.references:
        resolved = _resolve(ref.symbol_name, symbol_map)
        ref_step = steps_by_id.get(ref.turn_id)
        if not resolved or not ref_step:
            continue
        try:
            rk = ReferenceKind(ref.kind.lower())
        except ValueError:
            rk = ReferenceKind.UNKNOWN
        index.add_reference(symbol=resolved, step=ref_step, text=ref.text, kind=rk)

    for rel in result.relations:
        from_s = _resolve(rel.from_symbol, symbol_map)
        to_s = _resolve(rel.to_symbol, symbol_map)
        rel_step = steps_by_id.get(rel.turn_id)
        if not from_s or not to_s or not rel_step:
            continue
        try:
            rt = RelationType(rel.relation_type.lower())
        except ValueError:
            rt = RelationType.CO_MENTIONED
        index.add_relation(from_symbol=from_s, to_symbol=to_s, rel_type=rt, step=rel_step)


def _build_index_from_chunks(all_chunks: list[ChunkResults]) -> Any:
    """Rebuild a TrajectoryIndex from extraction chunk results."""
    from .index import TrajectoryIndex

    index = TrajectoryIndex()
    for chunk_list in all_chunks:
        for prompt_input, result in chunk_list:
            _build_index_from_chunks_into(index, prompt_input, result)
    return index


async def _collect_async(
    paths: list[Path],
    session_ids: list[str],
    model: str,
    output_dir: Path,
    index_output: Path | None,
    split: str,
    chunk_size: int,
    concurrency: int,
) -> None:
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

    logger.info(
        f"model={model} sources={len(work)} chunk_size={chunk_size} "
        f"output={output_dir} split={split}"
    )

    # Live index: built incrementally, dumped after each chunk
    live_index = _build_index_from_chunks([]) if index_output else None

    def _on_chunk(
        prompt_input: dict[str, Any],
        result: ExtractionResult,
        original_msgs: list[dict[str, JsonValue]],
    ) -> None:
        if live_index is None:
            return
        original_input: dict[str, Any] = {"messages": original_msgs}
        _build_index_from_chunks_into(live_index, original_input, result)
        live_index.dump(index_output)

    async def _process_one(label: str, source: tuple[str, Path | str]) -> ChunkResults:
        async with sem:
            kind, ref = source
            if kind == "file":
                msgs = load_messages_from_trace_file(ref)
            else:
                msgs = load_messages_from_session(str(ref))
            if not msgs:
                logger.warning(f"{label}: 0 messages, skipping")
                return []
            try:
                chunks = await extract_incremental(msgs, provider, chunk_size, on_chunk=_on_chunk)
            except Exception:
                logger.exception(f"{label}: extraction failed")
                return []
            if not chunks:
                logger.warning(f"{label}: no successful chunks")
                return []

            total_sym = sum(len(r.symbols) for _, r in chunks)
            total_ref = sum(len(r.references) for _, r in chunks)
            total_rel = sum(len(r.relations) for _, r in chunks)
            logger.info(
                f"{label}: {len(msgs)} msgs -> {len(chunks)} chunks, "
                f"{total_sym} sym, {total_ref} ref, {total_rel} rel"
            )
            return chunks

    tasks = [_process_one(label, source) for label, source in work]
    results = await asyncio.gather(*tasks)

    all_chunks = [cr for cr in results if cr]
    examples = [build_sft_example(pi, res) for cr in all_chunks for pi, res in cr]
    _write_hf_dataset(examples, output_dir, split)
    logger.info(f"done: {len(examples)} examples from {len(work)} sources -> {output_dir}/{split}.jsonl")

    if live_index:
        stats = live_index.stats()
        logger.info(
            f"index: {stats.symbol_count} symbols, {stats.reference_count} references, "
            f"{stats.relation_count} relations -> {index_output}"
        )


if __name__ == "__main__":
    app()
