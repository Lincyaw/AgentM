"""Runtime data utilities for trajectory-index extraction.

Trace message cleaning, JSON extraction, vocabulary validation, and
programmatic reference generation — used by the trajectory-index atom
during live agent sessions.

There is no CLI here: offline teacher extraction and evaluation live in
the ``agentm_eval.benchmarks.index_eval`` benchmark (adapter name
``index``), which drives this module's functions programmatically.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from importlib.resources import files
from typing import Any, Final

from loguru import logger

from ..agents.entity_extractor.schema import ExtractionResult

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

type JsonValue = str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]
type ProviderSpec = tuple[str, dict[str, JsonValue]]

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


def message_parts(msg: dict[str, JsonValue]) -> tuple[list[tuple[str, str]], str | None]:
    """THE single message walk: (content_part, view_part) pairs + tool name.

    Both the step content (``index._message_step_content``) and the
    extractor's view (:func:`view_body_with_map`) derive from these pairs
    — one walk, so the two representations that offset alignment depends
    on cannot drift apart.

    View parts differ from content parts in exactly one length-preserving
    way: literal annotation delimiters (``⟦``/``⟧``, should they ever
    occur in recorded content) are substituted with plain brackets so the
    markup parser never meets an unbalanced literal — the substitution
    preserves length, offsets map 1:1, and stored step content keeps the
    original characters. The view is otherwise the FULL text: gap elision
    keeps the model's output bounded, so the prompt window needs no
    truncation and every character is annotatable.
    """
    import json as _json

    from .markup import CLOSE, OPEN

    def _view(text: str) -> str:
        return text.replace(OPEN, "[").replace(CLOSE, "]")

    pairs: list[tuple[str, str]] = []
    tool_name: str | None = None
    blocks = msg.get("content", [])
    if isinstance(blocks, list):
        for block in blocks:
            if not isinstance(block, dict):
                continue
            btype = block.get("type", "")
            if btype == "text":
                text = str(block.get("text", ""))
                pairs.append((text, _view(text)))
            elif btype == "tool_call":
                name = block.get("name")
                tool_name = str(name) if name is not None else None
                args = block.get("arguments", block.get("input", {}))
                arg_text = (
                    _json.dumps(args, ensure_ascii=False)
                    if isinstance(args, dict) else str(args)
                )
                part = f"[tool_call: {tool_name}]\n{arg_text}"
                pairs.append((part, _view(part)))
            elif btype == "tool_result":
                sub = block.get("content", [])
                if isinstance(sub, list):
                    for s in sub:
                        if isinstance(s, dict):
                            text = str(s.get("text", ""))
                            pairs.append((text, _view(text)))
                else:
                    text = str(sub)
                    pairs.append((text, _view(text)))
    return pairs, tool_name


def view_body_with_map(msg: dict[str, JsonValue]) -> tuple[str, list[int | None]]:
    """The extractor's view of a message body + a view→content offset map.

    Takes the UNTRUNCATED serialized message dict. The view is what the
    extraction prompt shows; the map gives, for each view offset (plus one
    end-of-string entry), the corresponding offset into the step content —
    or None for synthesized characters (the truncation ellipsis). Both
    sides come from :func:`message_parts`, non-empty parts joined with a
    newline.
    """
    pairs, _tool = message_parts(msg)

    view_parts: list[str] = []
    mapping: list[int | None] = []
    content_off = 0
    first = True
    for content_part, view_part in pairs:
        if not content_part:
            continue
        if not first:
            view_parts.append("\n")
            mapping.append(content_off)   # the join newline
            content_off += 1
        first = False
        keep = (
            len(view_part) if len(view_part) == len(content_part)
            else len(view_part) - 3       # truncated: trailing "..." is synthesized
        )
        for k in range(len(view_part)):
            mapping.append(content_off + k if k < keep else None)
        view_parts.append(view_part)
        content_off += len(content_part)
    mapping.append(content_off)          # end-of-string
    return "".join(view_parts), mapping


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


def _reindex_messages(
    msgs: list[dict[str, JsonValue]],
    start: int = 0,
) -> list[dict[str, JsonValue]]:
    """Replace message IDs with absolute sequential integers for extraction."""
    return [{**msg, "id": str(i)} for i, msg in enumerate(msgs, start=start)]


# ---------------------------------------------------------------------------
# JSON extraction
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


# ---------------------------------------------------------------------------
# Vocabulary validation
# ---------------------------------------------------------------------------


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


def _validate_vocabulary(result: ExtractionResult, vocabulary: str = "default") -> str | None:
    """Validate annotated-message markup: well-formedness + sym attributes.

    Runs at parse time so a malformed response triggers the retry loop.
    Content fidelity (strip == original body) is checked later at populate
    time, where the originals are available.
    """
    from ..ir.models import _ENTITY_CLASS_VALUES
    from .markup import MarkupError, parse

    symbol_values, _reference_values, _relation_values = _load_vocabulary_values(vocabulary)

    errors: list[str] = []
    for am in result.annotated:
        try:
            _plain, annotations = parse(am.text)
        except MarkupError as exc:
            errors.append(f"message {am.message_id}: {exc}")
            continue
        for a in annotations:
            if a.tag != "sym":
                continue
            kind = a.attrs.get("kind", "unknown")
            if kind not in symbol_values:
                errors.append(f"message {am.message_id}: sym kind '{kind}' invalid")
            klass = a.attrs.get("class", "identifier")
            if klass not in _ENTITY_CLASS_VALUES:
                errors.append(f"message {am.message_id}: sym class '{klass}' invalid")
    if errors:
        valid_kinds = ", ".join(v for v in symbol_values if v != "unknown")
        return (
            f"Markup errors: {'; '.join(errors[:8])}. "
            f"Valid sym kinds: {valid_kinds}. Valid sym class: identifier, value, unknown."
        )
    return None


def _try_parse_response(
    messages: list[Any],
    vocabulary: str = "default",
) -> tuple[ExtractionResult | None, str | None]:
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
                vocab_error = _validate_vocabulary(result, vocabulary)
                if vocab_error:
                    return None, vocab_error
                return result, None
            return None, f"Could not extract JSON from response ({len(block.text)} chars)"
    return None, "No assistant text in response"


# ---------------------------------------------------------------------------
# Structural symbol extraction (code, no LLM)
# ---------------------------------------------------------------------------

_SQL_TABLE_RE: Final = re.compile(
    r"\b(?:FROM|JOIN|INTO|UPDATE|TABLE)\s+([A-Za-z_][A-Za-z0-9_.]*)",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class StructuralSymbol:
    name: str
    kind: str


def extract_structural_symbols(
    messages: list[dict[str, JsonValue]],
) -> list[StructuralSymbol]:
    """Extract symbols from structured message positions — no LLM needed.

    Two sources, both decidable, neither semantic recognition:

    * tool names — structural-position lookup (the ``name`` field of
      ``tool_call`` blocks);
    * SQL table names — formal-grammar parsing: the token after
      FROM/JOIN/INTO/UPDATE/TABLE is a table reference by the grammar of
      SQL itself, not by guessing what the text means. Registry-priming
      only (recall hint for the extractor's ⟦known⟧ marks); the model
      remains the authority on what gets declared a symbol.
    """
    seen: set[str] = set()
    out: list[StructuralSymbol] = []

    def _add(name: str, kind: str) -> None:
        key = name.lower()
        if key not in seen:
            seen.add(key)
            out.append(StructuralSymbol(name=name, kind=kind))

    def _scan_sql(text: str) -> None:
        for m in _SQL_TABLE_RE.finditer(text):
            table = m.group(1)
            if len(table) >= 2 and not table.startswith("("):
                _add(table, "table")

    for msg in messages:
        blocks = msg.get("content", [])
        if not isinstance(blocks, list):
            continue
        for block in blocks:
            if not isinstance(block, dict):
                continue
            btype = block.get("type", "")

            if btype == "tool_call":
                name = block.get("name")
                if isinstance(name, str) and name:
                    _add(name, "tool")
                args = block.get("arguments", block.get("input", {}))
                if isinstance(args, dict):
                    args_text = json.dumps(args, ensure_ascii=False)
                else:
                    args_text = str(args)
                _scan_sql(args_text)

            elif btype == "tool_result":
                sub = block.get("content", [])
                if isinstance(sub, list):
                    for s in sub:
                        if isinstance(s, dict):
                            _scan_sql(str(s.get("text", "")))
                else:
                    _scan_sql(str(sub))

    return out


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


_IDENT_CHAR_RE: Final = re.compile(r"[A-Za-z0-9_.\-/]")


def _usable_reference_term(term: str) -> bool:
    """Drop tiny aliases that create noisy substring matches."""
    norm = re.sub(r"\W+", "", term, flags=re.UNICODE)
    return len(norm) >= 2


def _ref_at_word_boundary(text: str, start: int, end: int) -> bool:
    """Check that the match is not inside a longer identifier."""
    if start > 0 and _IDENT_CHAR_RE.match(text[start - 1]):
        return False
    return not (end < len(text) and _IDENT_CHAR_RE.match(text[end]))


def _symbol_aliases(sym: dict[str, Any]) -> list[str]:
    aliases = sym.get("aliases", [])
    if not isinstance(aliases, list):
        return []
    return [alias for alias in aliases if isinstance(alias, str)]


def _reference_segments(msg: dict[str, JsonValue]) -> list[tuple[str, int, str]]:
    """``(segment_text, search_from, ref_kind)`` mirroring :func:`message_parts`.

    Reference ``start`` offsets must land in the stored ``step.content``, which
    :func:`_message_step_content` builds by ``"\\n".join`` over the SAME
    ``message_parts`` segments. So this walk reproduces those segment texts
    exactly — including the ``"[tool_call: name]\\n"`` header and one segment
    per tool_result sub-block — rather than a per-block concatenation of its
    own. ``search_from`` skips the tool_call header so a symbol is never
    matched inside ``[tool_call: name]`` while the header still counts toward
    offsets.
    """
    out: list[tuple[str, int, str]] = []
    blocks = msg.get("content", [])
    if not isinstance(blocks, list):
        return out
    for block in blocks:
        if not isinstance(block, dict):
            continue
        btype = block.get("type", "")
        if btype == "text":
            out.append((str(block.get("text", "")), 0, "mention"))
        elif btype == "tool_call":
            name = block.get("name")
            tool_name = str(name) if name is not None else None
            args = block.get("arguments", block.get("input", {}))
            arg_text = json.dumps(args, ensure_ascii=False) if isinstance(args, dict) else str(args)
            header = f"[tool_call: {tool_name}]\n"
            out.append((header + arg_text, len(header), "tool_input"))
        elif btype == "tool_result":
            kind = "tool_output" if block.get("deterministic", True) else "mention"
            sub = block.get("content", [])
            if isinstance(sub, list):
                for s in sub:
                    if isinstance(s, dict):
                        out.append((str(s.get("text", "")), 0, kind))
            else:
                out.append((str(sub), 0, kind))
    return out


def _build_references(
    symbols: list[dict[str, Any]],
    messages: list[dict[str, JsonValue]],
) -> list[GeneratedReference]:
    """Grep symbol names + aliases in messages to produce references
    (exact name/alias matching — deterministic given Pass 1 names)."""
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
    names.sort(key=lambda x: -len(x[0]))

    refs: list[GeneratedReference] = []
    seen: set[tuple[str, str, str]] = set()

    for msg in messages:
        mid = str(msg.get("id", ""))
        # Segments and their join mirror _message_step_content exactly, so
        # ``seg_offset`` tracks the true offset into step.content.
        seg_offset = 0
        first = True
        for seg_text, search_from, kind in _reference_segments(msg):
            if not seg_text:
                continue
            if not first:
                seg_offset += 1   # the "\n" that _message_step_content joins with
            first = False
            hay = seg_text.lower()

            for search_term, canonical in names:
                pos = hay.find(search_term, search_from)
                if pos < 0:
                    continue
                end = pos + len(search_term)
                if not _ref_at_word_boundary(seg_text, pos, end):
                    continue
                dedup_key = (canonical, mid, kind)
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)

                refs.append(GeneratedReference(
                    symbol_name=canonical,
                    turn_id=mid,
                    text=seg_text[pos:end][:50],
                    kind=kind,
                    start=pos + seg_offset,
                ))

            seg_offset += len(seg_text)

    return refs
