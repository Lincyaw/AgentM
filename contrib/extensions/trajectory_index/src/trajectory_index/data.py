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
from dataclasses import dataclass
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

DEFAULT_CHUNK_SIZE_SPEC: Final[str] = "2-5"
DEFAULT_CHUNK_SIZE: Final[tuple[int, int]] = (2, 5)


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
    except json.JSONDecodeError as exc:
        logger.debug("extract_json: full-text parse failed: {}", exc)

    m = _JSON_BLOCK_RE.search(text)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError as exc:
            logger.debug("extract_json: code-block parse failed: {}", exc)

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
                    except json.JSONDecodeError as exc:
                        logger.debug("extract_json: brace-scan parse failed: {}", exc)
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
            if desc.extension_module == "agentm.extensions.builtin.llm_openai":
                _inject_response_format(config)
                config.setdefault("max_output_tokens", 32768)
            return (desc.extension_module, config)
    raise RuntimeError(f"no extension module for provider {profile.provider!r}")


def _extraction_response_format() -> dict[str, JsonValue]:
    """Build an OpenAI Chat Completions json_schema response_format."""
    from agentm.core.lib.tool_schema import _force_strict, pydantic_to_tool_schema

    schema = cast(dict[str, JsonValue], _force_strict(pydantic_to_tool_schema(ExtractionResult)))
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "trajectory_extraction_result",
            "strict": True,
            "schema": schema,
        },
    }


def _inject_response_format(config: dict[str, JsonValue]) -> None:
    """Ask OpenAI-compatible providers to enforce extractor JSON output."""
    raw_extra = config.get("extra_body")
    extra_body: dict[str, JsonValue] = dict(raw_extra) if isinstance(raw_extra, dict) else {}
    extra_body.setdefault("response_format", _extraction_response_format())
    config["extra_body"] = extra_body


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
    start: int = 0,
) -> list[dict[str, JsonValue]]:
    """Replace message IDs with absolute sequential integers for extraction."""
    return [{**msg, "id": str(i)} for i, msg in enumerate(msgs, start=start)]



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
    """Check extraction result symbol kinds against the selected vocabulary yaml."""
    symbol_values, _reference_values, _relation_values = _load_vocabulary_values(_vocabulary)

    errors: list[str] = []
    for sym in result.symbols:
        if sym.kind not in symbol_values:
            errors.append(f"symbol '{sym.name}' has invalid kind '{sym.kind}'")
    if errors:
        valid_kinds = ", ".join(v for v in symbol_values if v != "unknown")
        return f"Vocabulary errors: {'; '.join(errors)}. Valid symbol kinds: {valid_kinds}."
    return None


# ---------------------------------------------------------------------------
# Programmatic reference generation
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class GeneratedReference:
    symbol_name: str
    turn_id: str
    text: str
    kind: str  # tool_input | tool_output | mention
    start: int


@dataclass(slots=True)
class GeneratedRelation:
    from_symbol: str
    to_symbol: str
    relation_type: str
    turn_id: str


def _usable_reference_term(term: str) -> bool:
    """Drop tiny aliases that create noisy substring matches."""
    norm = re.sub(r"\W+", "", term, flags=re.UNICODE)
    return len(norm) >= 2


def _symbol_aliases(sym: dict[str, Any]) -> list[str]:
    aliases = sym.get("aliases", [])
    if not isinstance(aliases, list):
        return []
    return [alias for alias in aliases if isinstance(alias, str)]


def _build_references(
    symbols: list[dict[str, Any]],
    messages: list[dict[str, JsonValue]],
) -> tuple[list[GeneratedReference], list[GeneratedRelation]]:
    """Grep symbol names + aliases in messages to produce references.

    Classifies by message structure:
    - tool_call content → tool_input
    - tool_result content → tool_output
    - text content → mention
    """
    names: list[tuple[str, str]] = []  # (search_term_lower, canonical_name)
    seen_terms: set[tuple[str, str]] = set()
    for sym in symbols:
        canonical = str(sym["name"])
        candidates = [canonical, *_symbol_aliases(sym)]
        for candidate in candidates:
            if not isinstance(candidate, str) or not _usable_reference_term(candidate):
                continue
            key = (candidate.lower(), canonical)
            if key in seen_terms:
                continue
            seen_terms.add(key)
            names.append(key)
    # longest-first to avoid partial matches shadowing longer names
    names.sort(key=lambda x: -len(x[0]))

    refs: list[GeneratedReference] = []
    seen: set[tuple[str, str, str]] = set()  # (canonical, turn_id, kind) dedup

    for msg in messages:
        mid = str(msg.get("id", ""))
        blocks = msg.get("content", [])
        if not isinstance(blocks, list):
            continue

        for block in blocks:
            if not isinstance(block, dict):
                continue
            btype = block.get("type", "")

            # Determine reference kind and extract searchable text
            if btype == "tool_call":
                kind = "tool_input"
                args = block.get("arguments", block.get("input", {}))
                text = json.dumps(args, ensure_ascii=False) if isinstance(args, dict) else str(args)
            elif btype == "tool_result":
                kind = "tool_output"
                sub = block.get("content", [])
                text = " ".join(
                    str(s.get("text", "")) for s in sub if isinstance(s, dict)
                ) if isinstance(sub, list) else str(sub)
            elif btype == "text":
                kind = "mention"
                text = str(block.get("text", ""))
            else:
                continue

            if not text:
                continue
            text_lower = text.lower()

            for search_term, canonical in names:
                pos = text_lower.find(search_term)
                if pos < 0:
                    continue
                dedup_key = (canonical, mid, kind)
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)

                # Extract verbatim snippet around the match
                snippet_start = max(0, pos)
                snippet_end = min(len(text), pos + len(search_term))
                snippet = text[snippet_start:snippet_end]
                if len(snippet) > 50:
                    snippet = snippet[:50]

                refs.append(GeneratedReference(
                    symbol_name=canonical,
                    turn_id=mid,
                    text=snippet,
                    kind=kind,
                    start=pos,
                ))

    return refs, []


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
                    logger.debug("data: caught exception: {}", exc)
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
    message_id_start: int = 0,
) -> ExtractionResult | None:
    """Run the extraction agent with a teacher model and return the result.

    Turn IDs in the result match the absolute message IDs sent in the prompt.
    """
    from agentm.core.abi import LoopConfig
    from agentm.core.abi.session_config import AgentSessionConfig
    from agentm.core.runtime.session import AgentSession

    reindexed = _reindex_messages(steps, start=message_id_start)

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


@dataclass(frozen=True, slots=True)
class MessageChunk:
    start: int
    messages: list[dict[str, JsonValue]]


def _chunk_messages(
    messages: list[dict[str, JsonValue]],
    size_range: tuple[int, int],
) -> list[MessageChunk]:
    """Split messages into chunks, keeping tool_call/tool_result pairs intact.

    ``size_range`` is (min, max); each chunk picks a random size in the range.
    """
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


@dataclass(frozen=True, slots=True)
class ExtractedChunk:
    run_id: str
    prompt_input: dict[str, Any]
    result: ExtractionResult


type OnChunkCallback = Callable[[ExtractedChunk, list[dict[str, JsonValue]]], None]


async def extract_incremental(
    messages: list[dict[str, JsonValue]],
    provider: ProviderSpec,
    chunk_size: tuple[int, int] = DEFAULT_CHUNK_SIZE,
    on_chunk: OnChunkCallback | None = None,
    run_id: str = "",
) -> list[ExtractedChunk]:
    """Extract symbols incrementally in chunks with registry accumulation.

    Returns extracted chunks with absolute message IDs preserved.
    """
    from .index import normalize_name

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
# SFT formatting
# ---------------------------------------------------------------------------


SFT_SYSTEM_PROMPT: Final = (
    "Extract a symbol table from the trajectory messages. Output JSON only."
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

    Accepts direct files, observability directories such as
    ``$AGENTM_HOME/observability``, and legacy case directories containing
    nested ``.agentm/observability/*.jsonl`` traces.
    """
    files: list[Path] = []
    seen: set[Path] = set()

    def add_trace_file(path: Path) -> None:
        if path.suffix != ".jsonl":
            return
        key = path.absolute()
        if key in seen:
            return
        seen.add(key)
        files.append(path)

    for p in paths:
        if p.is_file():
            add_trace_file(p)
        elif p.is_dir():
            for trace_file in sorted(p.glob("*.jsonl")):
                add_trace_file(trace_file)
            for trace_file in sorted(p.rglob(".agentm/observability/*.jsonl")):
                add_trace_file(trace_file)
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
    session_file: Annotated[Path | None, typer.Option("--session-file", help="File with one session ID per line")] = None,
    model: Annotated[str, typer.Option(help="Model profile from config.toml")] = "doubao",
    output_dir: Annotated[Path, typer.Option("--output", help="HuggingFace dataset output directory")] = Path("data"),
    index_output: Annotated[Path | None, typer.Option("--index-output", help="Write final index JSON to this path")] = None,
    split: Annotated[str, typer.Option(help="Dataset split name")] = "train",
    chunk_size: Annotated[
        str,
        typer.Option(help="Messages per chunk: '3' (fixed) or '2-5' (random range)"),
    ] = DEFAULT_CHUNK_SIZE_SPEC,
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
    all_sessions = list(session or [])
    if session_file:
        all_sessions.extend(
            line.strip() for line in session_file.read_text().splitlines() if line.strip()
        )
    asyncio.run(_collect_async(
        paths or [], all_sessions, model, output_dir, index_output,
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


type ChunkResults = list[ExtractedChunk]


_SPAN_SYMBOL_RE: Final = re.compile(r"^(?:span\s+)?s\d+$", re.IGNORECASE)


def _symbol_namespace(run_id: str, sym: dict[str, Any]) -> str:
    """Scope trajectory-local span IDs to the source run."""
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
    from .index import Step, StepRole

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

    # LLM-extracted symbols
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
        )

    # Programmatic references
    refs, _rels = _build_references(all_syms, msgs)
    for ref in refs:
        resolved = index.resolve_symbol_by_name(
            ref.symbol_name,
            namespace=namespaces.get(ref.symbol_name, ""),
        )
        ref_step = steps_by_id.get(ref.turn_id)
        if resolved and ref_step:
            index.add_reference(symbol=resolved, step=ref_step, text=ref.text, kind=ref.kind, start=ref.start)


def _build_index_from_chunks(all_chunks: list[ChunkResults]) -> Any:
    """Rebuild a TrajectoryIndex from extraction chunk results."""
    from .index import TrajectoryIndex

    index = TrajectoryIndex()
    for chunk_list in all_chunks:
        for extracted in chunk_list:
            _build_index_from_chunks_into(index, extracted)
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
        extracted: ExtractedChunk,
        _original_msgs: list[dict[str, JsonValue]],
    ) -> None:
        if live_index is None:
            return
        _build_index_from_chunks_into(live_index, extracted)
        live_index.dump(index_output)

    async def _process_one(src: TrajectorySource) -> ChunkResults:
        async with sem:
            label, msgs = src["label"], src["messages"]
            if len(msgs) < min_messages:
                logger.debug(f"{label}: {len(msgs)} msgs < {min_messages}, skipped")
                return []
            try:
                chunks = await extract_incremental(
                    msgs,
                    provider,
                    chunk_size,
                    on_chunk=_on_chunk,
                    run_id=label,
                )
            except Exception:
                logger.exception(f"{label}: extraction failed")
                return []
            if not chunks:
                logger.warning(f"{label}: no successful chunks")
                return []

            total_sym = sum(len(chunk.result.symbols) for chunk in chunks)
            logger.info(
                f"{label}: {len(msgs)} msgs -> {len(chunks)} chunks, "
                f"{total_sym} sym"
            )
            return chunks

    tasks = [_process_one(src) for src in sources]
    results = await asyncio.gather(*tasks)

    all_chunks = [cr for cr in results if cr]
    examples = [
        build_sft_example(chunk.prompt_input, chunk.result)
        for cr in all_chunks
        for chunk in cr
    ]
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
