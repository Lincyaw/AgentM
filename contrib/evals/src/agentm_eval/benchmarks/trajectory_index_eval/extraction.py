"""Trajectory-index extraction and SFT data pipeline.

Teacher extraction, incremental chunking, index rebuilding from chunks,
and SFT example formatting — used by eval benchmarks (aftraj, telbench)
and the SFT data collection CLI.

Moved from ``trajectory_index.data`` to separate eval code from the
runtime atom package.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Final, TypedDict

import typer
from loguru import logger

from trajectory_index.agents import extractor_scenario
from trajectory_index.agents.entity_extractor.schema import ExtractionResult
from trajectory_index.data import (
    JsonValue,
    ProviderSpec,
    _build_references,
    _reindex_messages,
    _symbol_aliases,
    _try_parse_response,
    clean_trace_messages,
)

DEFAULT_CHUNK_SIZE_SPEC: Final[str] = "2-5"
DEFAULT_CHUNK_SIZE: Final[tuple[int, int]] = (2, 5)


class ChatMessage(TypedDict):
    role: str
    content: str


class SftExample(TypedDict):
    messages: list[ChatMessage]


# ---------------------------------------------------------------------------
# Stream debugging
# ---------------------------------------------------------------------------

_stream_debug: bool = False


def _install_stream_tap(session: Any) -> None:
    import sys

    from agentm.core.abi import StreamDeltaEvent, TextDelta, ThinkingDelta, ToolCallStart

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


# ---------------------------------------------------------------------------
# Teacher extraction
# ---------------------------------------------------------------------------


async def extract(
    steps: list[dict[str, JsonValue]],
    provider: ProviderSpec | None = None,
    registry: list[dict[str, Any]] | None = None,
    message_id_start: int = 0,
    vocabulary: str = "default",
    model: str | None = None,
) -> ExtractionResult | None:
    """Run the extraction agent with a teacher model and return the result."""
    from agentm.core.abi import AgentSessionConfig, LoopConfig
    from agentm.core.runtime.session import AgentSession

    reindexed = _reindex_messages(steps, start=message_id_start)

    scenario = extractor_scenario()
    config = AgentSessionConfig(
        cwd=str(Path.cwd()),
        model=model,
        provider=provider,
        scenario=scenario,
        purpose="teacher_extraction",
        loop_config=LoopConfig(max_turns=1),
        log_trace_command=False,
        atom_config_overrides={
            "trajectory_extractor_context": {"vocabulary": vocabulary},
        },
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

    result, error = _try_parse_response(messages, vocabulary)
    if result:
        return result

    if not error:
        return None

    for attempt in range(3):
        logger.warning(f"extraction failed (retry {attempt + 1}/3): {error}")
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

        result, error = _try_parse_response(retry_messages, vocabulary)
        if result:
            return result
        if not error:
            break

    logger.warning(f"extraction failed after 3 retries: {error}")
    return None


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def _parse_chunk_size(value: str) -> tuple[int, int]:
    if "-" in value:
        lo, hi = value.split("-", 1)
        return int(lo), int(hi)
    n = int(value)
    return n, n


@dataclass(frozen=True, slots=True)
class MessageChunk:
    start: int
    messages: list[dict[str, JsonValue]]


def _chunk_messages(
    messages: list[dict[str, JsonValue]],
    size_range: tuple[int, int],
) -> list[MessageChunk]:
    import random

    lo, hi = size_range
    if len(messages) <= lo:
        return [MessageChunk(start=0, messages=messages)]

    chunks: list[MessageChunk] = []
    start = 0

    while start < len(messages):
        chunk_size = random.randint(lo, hi)
        end = start + chunk_size
        if end >= len(messages):
            chunks.append(MessageChunk(start=start, messages=messages[start:]))
            break
        while end < len(messages) and str(messages[end].get("role", "")) == "tool_result":
            end += 1
        chunks.append(MessageChunk(start=start, messages=messages[start:end]))
        start = end

    return chunks


# ---------------------------------------------------------------------------
# Incremental extraction
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExtractedChunk:
    run_id: str
    prompt_input: dict[str, Any]
    result: ExtractionResult


type OnChunkCallback = Callable[[ExtractedChunk, list[dict[str, JsonValue]]], None]


async def extract_incremental(
    messages: list[dict[str, JsonValue]],
    provider: ProviderSpec | None = None,
    chunk_size: tuple[int, int] = DEFAULT_CHUNK_SIZE,
    on_chunk: OnChunkCallback | None = None,
    run_id: str = "",
    vocabulary: str = "default",
    model: str | None = None,
) -> list[ExtractedChunk]:
    """Extract symbols incrementally in chunks with registry accumulation."""
    from trajectory_index.index import normalize_name

    chunks = _chunk_messages(messages, chunk_size)
    registry: list[dict[str, Any]] = []
    seen: set[str] = set()
    results: list[ExtractedChunk] = []

    for i, chunk in enumerate(chunks):
        chunk_registry = list(registry) if registry else None

        try:
            result = await extract(
                chunk.messages,
                provider,
                registry=chunk_registry,
                message_id_start=chunk.start,
                vocabulary=vocabulary,
                model=model,
            )
        except Exception:
            logger.exception(
                f"chunk {i+1}/{len(chunks)} ({len(chunk.messages)} msgs) extraction failed"
            )
            continue
        if result is None:
            logger.warning(
                f"chunk {i+1}/{len(chunks)} ({len(chunk.messages)} msgs) no parseable JSON"
            )
            continue

        reindexed = _reindex_messages(chunk.messages, start=chunk.start)
        if chunk_registry:
            prompt_input: dict[str, Any] = {"known_symbols": chunk_registry, "messages": reindexed}
        else:
            prompt_input = {"messages": reindexed}
        extracted = ExtractedChunk(run_id=run_id, prompt_input=prompt_input, result=result)
        results.append(extracted)

        if on_chunk:
            on_chunk(extracted, chunk.messages)

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
# Index building from chunks
# ---------------------------------------------------------------------------

_SPAN_SYMBOL_RE: Final = re.compile(r"^(?:span\s+)?s\d+$", re.IGNORECASE)


def _symbol_namespace(run_id: str, sym: dict[str, Any]) -> str:
    if not run_id or str(sym.get("kind", "")).lower() != "file":
        return ""
    names = [str(sym.get("name", "")), *_symbol_aliases(sym)]
    return run_id if any(_SPAN_SYMBOL_RE.match(str(name).strip()) for name in names) else ""


def _message_step_content(msg: dict[str, JsonValue]) -> tuple[str, str | None]:
    parts: list[str] = []
    tool_name: str | None = None
    blocks = msg.get("content", [])
    if not isinstance(blocks, list):
        return "", None
    for block in blocks:
        if not isinstance(block, dict):
            continue
        btype = block.get("type", "")
        if btype == "text":
            parts.append(str(block.get("text", "")))
        elif btype == "tool_call":
            name = block.get("name")
            tool_name = str(name) if name is not None else None
            args = block.get("arguments", block.get("input", {}))
            arg_text = json.dumps(args, ensure_ascii=False) if isinstance(args, dict) else str(args)
            parts.append(f"[tool_call: {tool_name}]\n{arg_text}")
        elif btype == "tool_result":
            sub = block.get("content", [])
            if isinstance(sub, list):
                parts.extend(str(s.get("text", "")) for s in sub if isinstance(s, dict))
            else:
                parts.append(str(sub))
    return "\n".join(part for part in parts if part), tool_name


def _step_index_from_id(step_id: str, fallback: int) -> int:
    try:
        return int(step_id)
    except ValueError:
        return fallback


def _build_index_from_chunks_into(
    index: Any,
    extracted: ExtractedChunk,
) -> None:
    """Populate a TrajectoryIndex: LLM symbols + programmatic references."""
    from trajectory_index.index import Step, StepRole

    role_map = {"user": StepRole.USER, "assistant": StepRole.ASSISTANT, "tool_result": StepRole.TOOL_RESULT}
    prompt_input = extracted.prompt_input
    result = extracted.result
    run_id = extracted.run_id
    msgs = prompt_input.get("messages", prompt_input if isinstance(prompt_input, list) else [])
    base_idx = len(index.steps)

    steps_by_id: dict[str, Step] = {}
    for i, msg in enumerate(msgs):
        mid = str(msg.get("id", f"s{base_idx + i}"))
        role = role_map.get(str(msg.get("role", "")), StepRole.USER)
        content, tool_name = _message_step_content(msg)
        step = Step(
            run_id=run_id,
            step_id=mid,
            index=_step_index_from_id(mid, base_idx + i),
            role=role,
            content=content,
            tool_name=tool_name,
        )
        index.add_step(step)
        steps_by_id[mid] = step

    known = prompt_input.get("known_symbols", [])
    all_syms = list(known) + [{"name": s.name, "kind": s.kind, "summary": s.summary, "aliases": s.aliases} for s in result.symbols]
    namespaces = {str(sym["name"]): _symbol_namespace(run_id, sym) for sym in all_syms}

    for ext_sym in result.symbols:
        sym_data = {"name": ext_sym.name, "kind": ext_sym.kind, "aliases": ext_sym.aliases}
        index.upsert_symbol(
            name=ext_sym.name,
            kind=ext_sym.kind.lower(),
            summary=ext_sym.summary,
            aliases=ext_sym.aliases,
            namespace=_symbol_namespace(run_id, sym_data),
            entity_class=getattr(ext_sym, "entity_class", "identifier"),
        )

    refs, _rels = _build_references(all_syms, msgs)
    for ref in refs:
        resolved = index.resolve_symbol_by_name(
            ref.symbol_name,
            namespace=namespaces.get(ref.symbol_name, ""),
        )
        ref_step = steps_by_id.get(ref.turn_id)
        if resolved and ref_step:
            index.add_reference(symbol=resolved, step=ref_step, text=ref.text, kind=ref.kind, start=ref.start)


type ChunkResults = list[ExtractedChunk]


def build_index_from_chunks(all_chunks: list[ChunkResults]) -> Any:
    """Rebuild a TrajectoryIndex from extraction chunk results."""
    from trajectory_index.index import TrajectoryIndex

    index = TrajectoryIndex()
    for chunk_list in all_chunks:
        for extracted in chunk_list:
            _build_index_from_chunks_into(index, extracted)
    index.build_dependencies()
    return index


# ---------------------------------------------------------------------------
# SFT formatting
# ---------------------------------------------------------------------------


def load_inference_prompt(vocabulary: str = "default") -> str:
    from trajectory_index.agents.entity_extractor.context import _build_vocabulary_section
    from trajectory_index.agents.entity_extractor.schema import ExtractionResult as Schema

    prompts_dir = Path(__file__).parents[3] / "trajectory_index" / "agents" / "entity_extractor" / "prompts"
    if not prompts_dir.is_dir():
        import trajectory_index.agents.entity_extractor as _ext
        prompts_dir = Path(_ext.__file__).parent / "prompts"
    base = (prompts_dir / "default.md").read_text(encoding="utf-8")
    vocab_section = _build_vocabulary_section(vocabulary)
    schema = json.dumps(Schema.model_json_schema(), indent=2, ensure_ascii=False)
    return f"{base}\n\n{vocab_section}\n\n## JSON Schema\n\nYour output must conform to this schema:\n\n```json\n{schema}\n```"


def build_sft_example(
    input_data: list[dict[str, JsonValue]] | dict[str, Any],
    teacher_output: ExtractionResult,
    system_prompt: str | None = None,
) -> SftExample:
    if system_prompt is None:
        system_prompt = load_inference_prompt()

    return SftExample(messages=[
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=json.dumps(input_data, ensure_ascii=False)),
        ChatMessage(role="assistant", content=teacher_output.model_dump_json(indent=None)),
    ])


# ---------------------------------------------------------------------------
# Trace loading
# ---------------------------------------------------------------------------


def load_messages_from_trace_file(path: str | Path) -> list[dict[str, JsonValue]]:
    from agentm.core.lib.trace_reader import TraceReader

    tr = TraceReader(str(path))
    return clean_trace_messages(tr.load_messages())


def _batch_load_sessions(
    session_ids: list[str],
) -> dict[str, list[dict[str, JsonValue]]]:
    from agentm.core.observability.clickhouse import bulk_session_entries, get_url

    if not session_ids:
        return {}
    url = get_url()
    if not url:
        logger.warning("ClickHouse unavailable, cannot batch-load sessions")
        return {}

    _BATCH = 200
    result: dict[str, list[dict[str, JsonValue]]] = {}
    for i in range(0, len(session_ids), _BATCH):
        batch = session_ids[i : i + _BATCH]
        loaded = 0
        for sid, entries in bulk_session_entries(url, batch).items():
            msgs = clean_trace_messages(entries)
            if msgs:
                result[sid] = msgs
                loaded += 1
        logger.info(f"batch {i // _BATCH + 1}: loaded {loaded}/{len(batch)} sessions")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

app = typer.Typer(pretty_exceptions_enable=False)

type TrajectorySource = dict[str, Any]
type LoaderFn = Callable[[list[Path]], list[TrajectorySource]]


def _find_trace_files(paths: list[Path]) -> list[Path]:
    found: list[Path] = []
    seen: set[Path] = set()

    def add(path: Path) -> None:
        if path.suffix != ".jsonl":
            return
        key = path.absolute()
        if key in seen:
            return
        seen.add(key)
        found.append(path)

    for p in paths:
        if p.is_file():
            add(p)
        elif p.is_dir():
            for tf in sorted(p.glob("*.jsonl")):
                add(tf)
            for tf in sorted(p.rglob(".agentm/observability/*.jsonl")):
                add(tf)
    return found


def _load_agentm_paths(paths: list[Path]) -> list[TrajectorySource]:
    sources: list[TrajectorySource] = []
    for tf in _find_trace_files(paths):
        msgs = load_messages_from_trace_file(tf)
        if msgs:
            sources.append({"label": tf.name, "messages": msgs})
    return sources


LOADERS: dict[str, LoaderFn] = {
    "agentm": _load_agentm_paths,
}


@app.command()
def collect(
    paths: Annotated[list[Path] | None, typer.Argument(help="Trace files or dirs")] = None,
    session: Annotated[list[str] | None, typer.Option(help="Session IDs from ClickHouse")] = None,
    session_file: Annotated[Path | None, typer.Option("--session-file")] = None,
    model: Annotated[str, typer.Option()] = "doubao",
    output_dir: Annotated[Path, typer.Option("--output")] = Path("data"),
    index_output: Annotated[Path | None, typer.Option("--index-output")] = None,
    split: Annotated[str, typer.Option()] = "train",
    chunk_size: Annotated[str, typer.Option()] = DEFAULT_CHUNK_SIZE_SPEC,
    concurrency: Annotated[int, typer.Option()] = 2,
    debug: Annotated[bool, typer.Option("--debug")] = False,
    fmt: Annotated[str, typer.Option("--format")] = "agentm",
    vocabulary: Annotated[str, typer.Option("--vocabulary")] = "default",
    min_messages: Annotated[int, typer.Option("--min-messages")] = 10,
) -> None:
    """Collect SFT training data from trajectories."""
    global _stream_debug
    if debug:
        _stream_debug = True
    all_sessions = list(session or [])
    if session_file:
        all_sessions.extend(
            line.strip() for line in session_file.read_text().splitlines() if line.strip()
        )
    asyncio.run(_collect_async(
        paths or [], all_sessions, model, output_dir, index_output,
        split, _parse_chunk_size(chunk_size), concurrency, fmt, min_messages,
        vocabulary,
    ))


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
    vocabulary: str = "default",
) -> None:
    loader = LOADERS.get(fmt)
    if not loader:
        logger.error(f"unknown format: {fmt!r}")
        raise typer.Exit(1)

    sources: list[TrajectorySource] = []
    if paths:
        sources.extend(loader(paths))
    if session_ids:
        loaded = _batch_load_sessions(session_ids)
        for sid in session_ids:
            msgs = loaded.get(sid, [])
            if msgs:
                sources.append({"label": sid, "messages": msgs})

    if not sources:
        logger.error("no trajectories found")
        raise typer.Exit(1)

    sem = asyncio.Semaphore(concurrency)
    logger.info(f"model={model} sources={len(sources)} chunk_size={chunk_size}")

    live_index = build_index_from_chunks([]) if index_output else None
    output_dir.mkdir(parents=True, exist_ok=True)
    data_file = output_dir / f"{split}.jsonl"
    example_count = 0
    write_lock = asyncio.Lock()

    def _on_chunk(extracted: ExtractedChunk, _msgs: list[dict[str, JsonValue]]) -> None:
        if live_index is None:
            return
        _build_index_from_chunks_into(live_index, extracted)

    async def _process_one(src: TrajectorySource) -> None:
        nonlocal example_count
        async with sem:
            label = src["label"]
            msgs = src["messages"]
            if len(msgs) < min_messages:
                return
            try:
                chunks = await extract_incremental(
                    msgs, chunk_size=chunk_size, model=model,
                    on_chunk=_on_chunk, run_id=label, vocabulary=vocabulary,
                )
            except Exception:
                logger.exception(f"{label}: extraction failed")
                return
            if not chunks:
                return

            examples = [build_sft_example(c.prompt_input, c.result) for c in chunks]
            async with write_lock:
                with data_file.open("a", encoding="utf-8") as f:
                    for ex in examples:
                        f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                example_count += len(examples)
            total_sym = sum(len(c.result.symbols) for c in chunks)
            logger.info(f"{label}: {len(chunks)} chunks, {total_sym} sym, {len(examples)} examples")

    await asyncio.gather(*[_process_one(s) for s in sources])
    logger.info(f"done: {example_count} examples -> {data_file}")

    if live_index and index_output:
        live_index.build_dependencies()
        live_index.dump(index_output)
