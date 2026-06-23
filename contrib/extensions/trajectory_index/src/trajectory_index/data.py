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
import os
import re
from collections.abc import Callable
from importlib.resources import files
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


def load_messages_batch(session_ids: list[str]) -> dict[str, list[dict[str, JsonValue]]]:
    """Bulk-load messages for many sessions in one ClickHouse query.

    Returns {session_id: cleaned_messages} for sessions that had data.
    """
    from agentm.core.observability.clickhouse import _parse_body, _query, get_url

    if not session_ids:
        return {}
    url = get_url()
    if not url:
        logger.warning("ClickHouse unavailable, cannot batch-load sessions")
        return {}

    BATCH = 200
    result: dict[str, list[dict[str, JsonValue]]] = {}
    for i in range(0, len(session_ids), BATCH):
        batch = session_ids[i : i + BATCH]
        placeholders = ", ".join(f"'{sid}'" for sid in batch)
        rows = _query(
            url,
            "SELECT LogAttributes['agentm.session.id'] AS sid, Body "
            "FROM otel_logs "
            "WHERE EventName = 'agentm.message.appended' "
            f"  AND LogAttributes['agentm.session.id'] IN ({placeholders}) "
            "ORDER BY LogAttributes['agentm.session.id'], Timestamp",
            timeout=120,
        )
        by_sid: dict[str, list[dict[str, JsonValue]]] = {}
        for row in rows:
            sid = row.get("sid", "")
            body = _parse_body(row.get("Body"))
            if isinstance(body, dict) and sid:
                by_sid.setdefault(sid, []).append(body)
        for sid, entries in by_sid.items():
            msgs = clean_trace_messages(entries)
            if msgs:
                result[sid] = msgs
        logger.info(f"batch {i // BATCH + 1}: loaded {len(by_sid)}/{len(batch)} sessions")

    return result


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
_vocabulary: str = "default"


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
) -> list[dict[str, JsonValue]]:
    """Replace message IDs with sequential integers for the extraction prompt."""
    return [{**msg, "id": str(i)} for i, msg in enumerate(msgs)]



def _load_vocabulary_values(vocab_name: str = "default") -> tuple[set[str], set[str], set[str]]:
    """Load valid vocabulary values from the selected yaml file."""
    import yaml

    fname = "vocabulary.yaml" if vocab_name == "default" else f"vocabulary.{vocab_name}.yaml"
    text = files("trajectory_index").joinpath(fname).read_text(encoding="utf-8")
    vocab = yaml.safe_load(text)
    return (
        set(vocab.get("symbol_kinds", {})),
        set(vocab.get("reference_kinds", {})),
        set(vocab.get("relation_types", {})),
    )


def _validate_vocabulary(result: ExtractionResult) -> str | None:
    """Check extraction result against the selected vocabulary yaml."""
    symbol_values, reference_values, relation_values = _load_vocabulary_values(_vocabulary)

    errors: list[str] = []
    for sym in result.symbols:
        if sym.kind not in symbol_values:
            errors.append(f"symbol '{sym.name}' has invalid kind '{sym.kind}'")
    for ref in result.references:
        if ref.kind not in reference_values:
            errors.append(f"reference '{ref.symbol_name}' has invalid kind '{ref.kind}'")
    for rel in result.relations:
        if rel.relation_type not in relation_values:
            errors.append(f"relation '{rel.from_symbol}'->{rel.to_symbol}' has invalid type '{rel.relation_type}'")
    if errors:
        valid_kinds = ", ".join(v for v in symbol_values if v != "unknown")
        valid_refs = ", ".join(v for v in reference_values if v != "unknown")
        valid_rels = ", ".join(relation_values)
        return (
            f"Vocabulary errors: {'; '.join(errors)}. "
            f"Valid symbol kinds: {valid_kinds}. "
            f"Valid reference kinds: {valid_refs}. "
            f"Valid relation types: {valid_rels}."
        )
    return None


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
                    result = ExtractionResult.model_validate(obj)
                except Exception as exc:
                    return None, f"JSON keys={list(obj.keys())}; validation error: {exc}"
                vocab_error = _validate_vocabulary(result)
                if vocab_error:
                    return None, vocab_error
                return result, None
            return None, f"Could not extract JSON from response ({len(block.text)} chars)"
    return None, "No assistant text in response"


async def extract(
    steps: list[dict[str, JsonValue]],
    provider: ProviderSpec,
    registry: list[dict[str, Any]] | None = None,
) -> ExtractionResult | None:
    """Run the extraction agent with a teacher model and return the result.

    Turn IDs in the result are sequential (0,1,2…) matching the reindexed input.
    """
    from agentm.core.abi import LoopConfig
    from agentm.core.abi.session_config import AgentSessionConfig
    from agentm.core.runtime.session import AgentSession

    reindexed = _reindex_messages(steps)

    scenario = extractor_scenario()
    config = AgentSessionConfig(
        cwd=str(Path.cwd()),
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
        return result

    logger.warning(f"retry also failed: {retry_error}")
    return None


def _parse_chunk_size(value: str) -> tuple[int, int]:
    """Parse chunk size spec: '20' (fixed) or '5-10' (random range)."""
    if "-" in value:
        lo, hi = value.split("-", 1)
        return int(lo), int(hi)
    n = int(value)
    return n, n


def _chunk_messages(
    messages: list[dict[str, JsonValue]],
    size_range: tuple[int, int],
) -> list[list[dict[str, JsonValue]]]:
    """Split messages into chunks, keeping tool_call/tool_result pairs intact.

    ``size_range`` is (min, max); each chunk picks a random size in the range.
    """
    import random

    lo, hi = size_range
    if len(messages) <= lo:
        return [messages]

    chunks: list[list[dict[str, JsonValue]]] = []
    start = 0

    while start < len(messages):
        chunk_size = random.randint(lo, hi)
        end = start + chunk_size
        if end >= len(messages):
            chunks.append(messages[start:])
            break
        while end < len(messages) and str(messages[end].get("role", "")) == "tool_result":
            end += 1
        chunks.append(messages[start:end])
        start = end

    return chunks


type OnChunkCallback = Callable[[dict[str, Any], ExtractionResult, list[dict[str, JsonValue]]], None]


async def extract_incremental(
    messages: list[dict[str, JsonValue]],
    provider: ProviderSpec,
    chunk_size: tuple[int, int] = (20, 20),
    on_chunk: OnChunkCallback | None = None,
) -> list[tuple[dict[str, Any], ExtractionResult]]:
    """Extract symbols incrementally in chunks with registry accumulation.

    Returns (prompt_input, result) per successful chunk.
    Turn IDs in results are sequential (0,1,2…) matching the prompt input.
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

        reindexed = _reindex_messages(chunk)
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
                if sym.summary:
                    entry["summary"] = sym.summary
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
    from .agents.entity_extractor.context import _build_vocabulary_section
    from .agents.entity_extractor.schema import ExtractionResult as Schema

    prompts_dir = Path(__file__).parent / "agents" / "entity_extractor" / "prompts"
    base = (prompts_dir / "default.md").read_text(encoding="utf-8")
    vocabulary = _build_vocabulary_section()
    schema = json.dumps(Schema.model_json_schema(), indent=2, ensure_ascii=False)
    return f"{base}\n\n{vocabulary}\n\n## JSON Schema\n\nYour output must conform to this schema:\n\n```json\n{schema}\n```"


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


# ---------------------------------------------------------------------------
# Pluggable trajectory loaders
# ---------------------------------------------------------------------------


class TrajectorySource(TypedDict):
    label: str
    messages: list[dict[str, JsonValue]]


type LoaderFn = Callable[[list[Path]], list[TrajectorySource]]


def _load_agentm_paths(paths: list[Path]) -> list[TrajectorySource]:
    """Load trajectories from AgentM trace files/directories."""
    sources: list[TrajectorySource] = []
    for tf in _find_trace_files(paths):
        msgs = load_messages_from_trace_file(tf)
        if msgs:
            sources.append({"label": tf.name, "messages": msgs})
    return sources


def _telbench_to_messages(entry: dict[str, Any]) -> list[dict[str, JsonValue]]:
    """Convert a TELBench JSONL entry to extraction-ready messages."""
    msgs: list[dict[str, JsonValue]] = []
    question = entry.get("question", "")
    if question:
        msgs.append({
            "id": "q",
            "role": "user",
            "content": [{"type": "text", "text": question}],
        })
    for span in entry.get("spans", []):
        msgs.append({
            "id": span["id"],
            "role": "assistant",
            "content": [{"type": "text", "text": span["raw"]}],
        })
    return msgs


def _load_telbench(paths: list[Path]) -> list[TrajectorySource]:
    """Load trajectories from TELBench JSONL (one line = one trajectory)."""
    sources: list[TrajectorySource] = []
    for p in paths:
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                msgs = _telbench_to_messages(entry)
                if msgs:
                    sources.append({"label": str(entry.get("id", p.stem)), "messages": msgs})
    return sources


LOADERS: dict[str, LoaderFn] = {
    "agentm": _load_agentm_paths,
    "telbench": _load_telbench,
}


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
    chunk_size: Annotated[str, typer.Option(help="Messages per chunk: '20' (fixed) or '2-5' (random range)")] = "2-5",
    concurrency: Annotated[int, typer.Option(help="Max concurrent traces")] = 2,
    debug: Annotated[bool, typer.Option("--debug", help="Enable debug logging")] = False,
    fmt: Annotated[str, typer.Option("--format", help=f"Input format: {', '.join(LOADERS)}")] = "agentm",
    vocabulary: Annotated[str, typer.Option("--vocabulary", help="Vocabulary name (default, research, …)")] = "default",
    min_messages: Annotated[int, typer.Option("--min-messages", help="Skip trajectories with fewer messages")] = 10,
) -> None:
    """Collect SFT training data: export traces, run incremental extraction, write HF dataset.

    Sources (use one or both):
      - positional paths: local JSONL files or directories
      - --session: session IDs loaded from ClickHouse
    """
    global _stream_debug, _vocabulary
    _vocabulary = vocabulary
    os.environ["TRAJ_INDEX_VOCABULARY"] = vocabulary
    if debug:
        _stream_debug = True
        import sys
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    asyncio.run(_collect_async(
        paths or [], session or [], model, output_dir, index_output,
        split, _parse_chunk_size(chunk_size), concurrency, fmt, min_messages,
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
    from .index import Step, StepRole, Symbol

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
        symbol = index.upsert_symbol(name=ext_sym.name, kind=ext_sym.kind.lower(), summary=ext_sym.summary, aliases=ext_sym.aliases)
        symbol_map[ext_sym.name] = symbol

    for ref in result.references:
        resolved = _resolve(ref.symbol_name, symbol_map)
        ref_step = steps_by_id.get(ref.turn_id)
        if not resolved or not ref_step:
            continue
        index.add_reference(symbol=resolved, step=ref_step, text=ref.text, kind=ref.kind.lower())

    for rel in result.relations:
        from_s = _resolve(rel.from_symbol, symbol_map)
        to_s = _resolve(rel.to_symbol, symbol_map)
        rel_step = steps_by_id.get(rel.turn_id)
        if not from_s or not to_s or not rel_step:
            continue
        index.add_relation(from_symbol=from_s, to_symbol=to_s, rel_type=rel.relation_type.lower(), step=rel_step)


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
    chunk_size: tuple[int, int],
    concurrency: int,
    fmt: str = "agentm",
    min_messages: int = 10,
) -> None:
    loader = LOADERS.get(fmt)
    if not loader:
        logger.error(f"unknown format: {fmt!r}, available: {', '.join(LOADERS)}")
        raise typer.Exit(1)

    sources: list[TrajectorySource] = []
    if paths:
        sources.extend(loader(paths))
    if session_ids:
        batch = load_messages_batch(session_ids)
        for sid in session_ids:
            msgs = batch.get(sid)
            if msgs:
                sources.append({"label": sid, "messages": msgs})

    if not sources:
        logger.error("no trajectories found")
        raise typer.Exit(1)

    provider = resolve_provider(model)
    sem = asyncio.Semaphore(concurrency)

    logger.info(
        f"model={model} sources={len(sources)} chunk_size={chunk_size} "
        f"output={output_dir} split={split}"
    )

    live_index = _build_index_from_chunks([]) if index_output else None

    def _on_chunk(
        prompt_input: dict[str, Any],
        result: ExtractionResult,
        _original_msgs: list[dict[str, JsonValue]],
    ) -> None:
        if live_index is None:
            return
        _build_index_from_chunks_into(live_index, prompt_input, result)
        live_index.dump(index_output)

    async def _process_one(src: TrajectorySource) -> ChunkResults:
        async with sem:
            label, msgs = src["label"], src["messages"]
            if len(msgs) < min_messages:
                logger.debug(f"{label}: {len(msgs)} msgs < {min_messages}, skipped")
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

    tasks = [_process_one(src) for src in sources]
    results = await asyncio.gather(*tasks)

    all_chunks = [cr for cr in results if cr]
    examples = [build_sft_example(pi, res) for cr in all_chunks for pi, res in cr]
    _write_hf_dataset(examples, output_dir, split)
    logger.info(f"done: {len(examples)} examples from {len(sources)} sources -> {output_dir}/{split}.jsonl")

    if live_index:
        stats = live_index.stats()
        logger.info(
            f"index: {stats.symbol_count} symbols, {stats.reference_count} references, "
            f"{stats.relation_count} relations -> {index_output}"
        )


if __name__ == "__main__":
    app()
