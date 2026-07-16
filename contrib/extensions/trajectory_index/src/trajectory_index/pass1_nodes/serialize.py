"""Runtime data utilities for trajectory-index extraction.

JSON extraction, vocabulary validation, programmatic reference generation,
and the single message walk that both step content and the extractor view
derive from.
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

# ---------------------------------------------------------------------------
# Message walk — the single source for step content + extractor view
# ---------------------------------------------------------------------------


def message_parts(msg: dict[str, JsonValue]) -> tuple[list[tuple[str, str]], str | None]:
    """THE single message walk: (content_part, view_part) pairs + tool name.

    Both the step content and the extractor's view derive from these pairs
    — one walk, so the two representations that offset alignment depends
    on cannot drift apart.

    View parts differ from content parts in exactly one length-preserving
    way: literal annotation delimiters are substituted with plain brackets
    so the markup parser never meets an unbalanced literal.
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
    """The extractor's view of a message body + a view→content offset map."""
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
            mapping.append(content_off)
            content_off += 1
        first = False
        keep = (
            len(view_part) if len(view_part) == len(content_part)
            else len(view_part) - 3
        )
        for k in range(len(view_part)):
            mapping.append(content_off + k if k < keep else None)
        view_parts.append(view_part)
        content_off += len(content_part)
    mapping.append(content_off)
    return "".join(view_parts), mapping


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


# ---------------------------------------------------------------------------
# Extraction result parsing
# ---------------------------------------------------------------------------


def _load_vocabulary_values(vocab_name: str = "default") -> tuple[set[str], set[str], set[str]]:
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
    from ..ir.models import _ENTITY_CLASS_VALUES

    symbol_values, _reference_values, _relation_values = _load_vocabulary_values(vocabulary)
    for sym in (result.symbols or []):
        kind = (sym.kind or "unknown").lower()
        if kind not in symbol_values:
            sym.kind = "unknown"
        entity_class = getattr(sym, "entity_class", "identifier")
        if entity_class not in _ENTITY_CLASS_VALUES:
            sym.entity_class = "identifier"
    return None


def _parse_tags_to_result(text: str) -> ExtractionResult | None:
    from .markup import OPEN, MarkupError, parse

    if OPEN not in text:
        return None
    try:
        _plain, annotations = parse(text)
    except MarkupError:
        return None

    from ..agents.entity_extractor.schema import (
        ExtractedClaim,
        ExtractedConstraint,
        ExtractedObs,
        ExtractedSymbol,
        ExtractedValue,
    )

    symbols: list[ExtractedSymbol] = []
    claims: list[ExtractedClaim] = []
    observations: list[ExtractedObs] = []
    constraints: list[ExtractedConstraint] = []
    values: list[ExtractedValue] = []

    for ann in annotations:
        if ann.depth > 0:
            continue
        content = _plain[ann.start:ann.end]
        if ann.tag == "sym":
            kind = ann.attrs.get("kind", "unknown")
            name_attr = ann.attrs.get("name", "")
            entity_class = ann.attrs.get("class", "identifier")
            canonical = name_attr if name_attr else content
            aliases = [content] if name_attr and content != name_attr else []
            symbols.append(ExtractedSymbol(
                name=canonical, kind=kind, aliases=aliases, entity_class=entity_class,
            ))
        elif ann.tag == "claim":
            role = ann.attrs.get("role", "")
            parts = content.split("…", 1)
            head = parts[0].strip()
            tail = parts[1].strip() if len(parts) > 1 else ""
            if head:
                claims.append(ExtractedClaim(head=head, tail=tail, role=role))
        elif ann.tag == "val":
            sym_name = ann.attrs.get("sym", "")
            val_text = content.strip()
            if sym_name and val_text:
                values.append(ExtractedValue(sym=sym_name, value=val_text))
        elif ann.tag == "obs":
            parts = content.split("…", 1)
            head = parts[0].strip()
            tail = parts[1].strip() if len(parts) > 1 else ""
            if head:
                observations.append(ExtractedObs(head=head, tail=tail))
        elif ann.tag == "constraint":
            parts = content.split("…", 1)
            head = parts[0].strip()
            tail = parts[1].strip() if len(parts) > 1 else ""
            if head:
                constraints.append(ExtractedConstraint(head=head, tail=tail))

    if not symbols and not claims and not observations and not constraints and not values:
        return None
    return ExtractionResult(
        symbols=symbols, claims=claims, observations=observations,
        constraints=constraints, values=values,
    )


def _try_parse_response(
    messages: list[Any],
    vocabulary: str = "default",
) -> tuple[ExtractionResult | None, str | None]:
    from agentm.core.abi import TextContent

    if not messages:
        return None, "No messages"
    msg = messages[-1]
    content = getattr(msg, "content", None)
    if not content:
        return None, "Last message has no content"
    blocks = content if isinstance(content, list) else [content]
    for block in blocks:
        if not isinstance(block, TextContent):
            continue
        text = block.text
        result = _parse_tags_to_result(text)
        if result is None:
            obj = extract_json(text)
            if obj:
                try:
                    result = ExtractionResult.model_validate(obj)
                except Exception as exc:
                    logger.debug("data: caught exception: {}", exc)
                    return None, f"JSON validation error: {exc}"
        if result is None:
            return None, f"No tags or JSON found in response ({len(text)} chars)"
        vocab_error = _validate_vocabulary(result, vocabulary)
        if vocab_error:
            return None, vocab_error
        return result, None
    return None, "Last message has no text content"


# ---------------------------------------------------------------------------
# Programmatic reference generation
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class GeneratedReference:
    symbol_name: str
    turn_id: str
    text: str
    kind: str
    start: int


_IDENT_CHAR_RE: Final = re.compile(r"[A-Za-z0-9_.\-/]")


def _ref_kind_for_block(block: dict[str, Any]) -> str:
    btype = block.get("type", "")
    if btype == "tool_call":
        return "tool_input"
    if btype == "tool_result":
        return "tool_output"
    return "mention"


def _build_references(
    symbols: list[dict[str, Any]],
    messages: list[dict[str, JsonValue]],
) -> list[GeneratedReference]:
    """Grep symbol names in messages to produce references."""
    terms: list[tuple[str, str]] = []
    seen_terms: set[tuple[str, str]] = set()
    for sym in symbols:
        canonical = str(sym["name"])
        aliases = sym.get("aliases", [])
        if not isinstance(aliases, list):
            aliases = []
        for candidate in [canonical, *aliases]:
            if not isinstance(candidate, str):
                continue
            norm = re.sub(r"\W+", "", candidate, flags=re.UNICODE)
            if len(norm) < 2:
                continue
            key = (candidate.lower(), canonical)
            if key not in seen_terms:
                seen_terms.add(key)
                terms.append(key)
    terms.sort(key=lambda x: -len(x[0]))

    refs: list[GeneratedReference] = []
    seen: set[tuple[str, str, str]] = set()

    for msg in messages:
        mid = str(msg.get("id", ""))
        parts, _ = message_parts(msg)
        blocks = msg.get("content", [])
        block_list = blocks if isinstance(blocks, list) else []
        block_kinds = [_ref_kind_for_block(b) for b in block_list if isinstance(b, dict)]

        seg_offset = 0
        first = True
        kind_idx = 0
        for content_part, _view_part in parts:
            if not content_part:
                continue
            if not first:
                seg_offset += 1
            first = False

            kind = block_kinds[kind_idx] if kind_idx < len(block_kinds) else "mention"
            hay = content_part.lower()

            for search_term, canonical in terms:
                pos = hay.find(search_term)
                if pos < 0:
                    continue
                end = pos + len(search_term)
                if pos > 0 and _IDENT_CHAR_RE.match(content_part[pos - 1]):
                    continue
                if end < len(content_part) and _IDENT_CHAR_RE.match(content_part[end]):
                    continue
                dedup_key = (canonical, mid, kind)
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)
                refs.append(GeneratedReference(
                    symbol_name=canonical,
                    turn_id=mid,
                    text=content_part[pos:end][:50],
                    kind=kind,
                    start=pos + seg_offset,
                ))

            seg_offset += len(content_part)
            kind_idx += 1

    return refs
