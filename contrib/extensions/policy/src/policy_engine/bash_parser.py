# code-health: ignore-file[AM022] -- tree-sitter exposes dynamic parser/node objects
"""Bash command parsing helpers for policy trace views.

This module owns shell syntax parsing only. Policy-specific labels such as
"query" or "write" are assigned by the caller from the structured segments.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from loguru import logger


@dataclass(frozen=True, slots=True)
class BashRedirect:
    kind: str
    operator: str
    destination: str
    descriptor: str | None
    text: str


@dataclass(frozen=True, slots=True)
class BashSegment:
    argv: tuple[str, ...]
    text: str
    redirects: tuple[BashRedirect, ...]
    parser: str
    pipeline_index: int | None = None
    depth: int = 0
    start_byte: int = 0
    end_byte: int = 0

    @property
    def command(self) -> str:
        return self.argv[0] if self.argv else ""


@dataclass(frozen=True, slots=True)
class _Context:
    redirects: tuple[BashRedirect, ...] = ()
    pipeline_index: int | None = None
    depth: int = 0


def parse_bash_segments(source: str) -> tuple[BashSegment, ...]:
    """Parse shell source into command segments.

    tree-sitter-bash is preferred because it preserves shell structure such as
    pipelines, redirects, heredocs, and command substitutions. A small shlex
    fallback keeps the view usable if optional parser packages are unavailable.
    """

    if not source.strip():
        return ()
    parser = _tree_sitter_parser()
    if parser is not None:
        try:
            return _tree_sitter_segments(parser, source)
        except Exception as exc:
            logger.debug("tree-sitter bash parse failed; falling back: {}", exc)
            return _fallback_segments(source)
    return _fallback_segments(source)


@lru_cache(maxsize=1)
def _tree_sitter_parser() -> Any | None:
    try:
        from tree_sitter import Language, Parser
        import tree_sitter_bash
    except Exception as exc:
        logger.debug("tree-sitter bash parser unavailable: {}", exc)
        return None
    try:
        return Parser(Language(tree_sitter_bash.language()))
    except Exception as exc:
        logger.debug("tree-sitter bash parser initialization failed: {}", exc)
        return None


def _tree_sitter_segments(parser: Any, source: str) -> tuple[BashSegment, ...]:
    source_bytes = source.encode()
    tree = parser.parse(source_bytes)
    segments: list[BashSegment] = []
    _collect_segments(tree.root_node, source_bytes, _Context(), segments)
    return tuple(segments)


def _collect_segments(
    node: Any,
    source: bytes,
    context: _Context,
    segments: list[BashSegment],
) -> None:
    if node.type == "pipeline":
        index = 0
        for child in node.children:
            if not child.is_named:
                continue
            _collect_segments(
                child,
                source,
                _Context(
                    redirects=context.redirects,
                    pipeline_index=index,
                    depth=context.depth,
                ),
                segments,
            )
            if _contains_command(child):
                index += 1
        return

    if node.type == "redirected_statement":
        redirects = tuple(
            redirect
            for child in node.children
            if _field_name(node, child) == "redirect"
            for redirect in _redirect_from_node(child, source)
        )
        for child in node.children:
            if _field_name(node, child) == "body":
                _collect_redirected_body(
                    child,
                    source,
                    context,
                    redirects,
                    segments,
                )
        return

    if node.type == "command":
        segment = _segment_from_command(node, source, context)
        if segment is not None:
            segments.append(segment)
        for child in node.children:
            if child.type in {
                "command_substitution",
                "process_substitution",
                "subshell",
            }:
                _collect_segments(
                    child,
                    source,
                    _Context(
                        redirects=(),
                        pipeline_index=None,
                        depth=context.depth + 1,
                    ),
                    segments,
                )
            elif _contains_node_type(
                child,
                {"command_substitution", "process_substitution", "subshell"},
            ):
                _collect_nested_shell(child, source, context.depth + 1, segments)
        return

    if node.type in {"command_substitution", "process_substitution", "subshell"}:
        for child in node.children:
            if child.is_named:
                _collect_segments(
                    child,
                    source,
                    _Context(depth=context.depth + 1),
                    segments,
                )
        return

    for child in node.children:
        if child.is_named:
            _collect_segments(child, source, context, segments)


def _collect_redirected_body(
    node: Any,
    source: bytes,
    context: _Context,
    redirects: tuple[BashRedirect, ...],
    segments: list[BashSegment],
) -> None:
    if node.type in {"list", "pipeline"}:
        _collect_compound_with_tail_redirects(
            node,
            source,
            context,
            redirects,
            segments,
        )
        return
    _collect_segments(
        node,
        source,
        _Context(
            redirects=context.redirects + redirects,
            pipeline_index=context.pipeline_index,
            depth=context.depth,
        ),
        segments,
    )


def _collect_compound_with_tail_redirects(
    node: Any,
    source: bytes,
    context: _Context,
    redirects: tuple[BashRedirect, ...],
    segments: list[BashSegment],
) -> None:
    command_children = [
        child for child in node.children if child.is_named and _contains_command(child)
    ]
    if not command_children:
        return
    tail_id = command_children[-1].id
    pipeline_index = 0
    for child in node.children:
        if not child.is_named:
            continue
        child_redirects = context.redirects + (redirects if child.id == tail_id else ())
        child_pipeline_index = (
            pipeline_index
            if node.type == "pipeline" and _contains_command(child)
            else None
        )
        _collect_segments(
            child,
            source,
            _Context(
                redirects=child_redirects,
                pipeline_index=child_pipeline_index,
                depth=context.depth,
            ),
            segments,
        )
        if node.type == "pipeline" and _contains_command(child):
            pipeline_index += 1


def _collect_nested_shell(
    node: Any,
    source: bytes,
    depth: int,
    segments: list[BashSegment],
) -> None:
    if node.type in {"command_substitution", "process_substitution", "subshell"}:
        _collect_segments(node, source, _Context(depth=depth), segments)
        return
    for child in node.children:
        if child.is_named:
            _collect_nested_shell(child, source, depth, segments)


def _segment_from_command(
    node: Any,
    source: bytes,
    context: _Context,
) -> BashSegment | None:
    name_node = _child_by_field(node, "name")
    command = _command_name_text(name_node, source)
    if not command:
        return None
    argv = [command]
    for child in node.children:
        if _field_name(node, child) == "argument":
            argv.append(_node_text(child, source))
    return BashSegment(
        argv=tuple(argv),
        text=_node_text(node, source),
        redirects=context.redirects,
        parser="tree-sitter-bash",
        pipeline_index=context.pipeline_index,
        depth=context.depth,
        start_byte=node.start_byte,
        end_byte=node.end_byte,
    )


def _command_name_text(node: Any | None, source: bytes) -> str:
    if node is None:
        return ""
    for child in node.children:
        if child.type == "word":
            return _node_text(child, source)
    return ""


def _redirect_from_node(node: Any, source: bytes) -> tuple[BashRedirect, ...]:
    if node.type == "file_redirect":
        destination = _child_by_field(node, "destination")
        descriptor = _child_by_field(node, "descriptor")
        return (
            BashRedirect(
                kind="file",
                operator=_redirect_operator(node, source),
                destination=_node_text(destination, source) if destination else "",
                descriptor=_node_text(descriptor, source) if descriptor else None,
                text=_node_text(node, source),
            ),
        )
    if node.type == "heredoc_redirect":
        start = next(
            (child for child in node.children if child.type == "heredoc_start"),
            None,
        )
        return (
            BashRedirect(
                kind="heredoc",
                operator=_redirect_operator(node, source),
                destination=_node_text(start, source) if start else "",
                descriptor=None,
                text=_node_text(node, source),
            ),
        )
    return ()


def _redirect_operator(node: Any, source: bytes) -> str:
    for child in node.children:
        if child.is_named or child.type == "file_descriptor":
            continue
        return _node_text(child, source)
    return ""


def _child_by_field(node: Any, field: str) -> Any | None:
    for child in node.children:
        if _field_name(node, child) == field:
            return child
    return None


def _field_name(parent: Any, child: Any) -> str | None:
    for index, candidate in enumerate(parent.children):
        if candidate.id == child.id:
            return parent.field_name_for_child(index)
    return None


def _contains_command(node: Any) -> bool:
    return node.type in {"command", "redirected_statement"} or _contains_node_type(
        node,
        {"command"},
    )


def _contains_node_type(node: Any, types: set[str]) -> bool:
    if node.type in types:
        return True
    return any(_contains_node_type(child, types) for child in node.children)


def _node_text(node: Any | None, source: bytes) -> str:
    if node is None:
        return ""
    return source[node.start_byte : node.end_byte].decode("utf-8", "replace")


def _fallback_segments(source: str) -> tuple[BashSegment, ...]:
    try:
        lexer = shlex.shlex(
            source.replace("\n", " ; "), posix=True, punctuation_chars=True
        )
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        tokens = source.split()

    segments: list[BashSegment] = []
    current: list[str] = []
    for token in tokens:
        if token in {";", "&&", "||", "|", "|&"}:
            if current:
                segments.append(_fallback_segment(current))
                current = []
            continue
        current.append(token)
    if current:
        segments.append(_fallback_segment(current))
    return tuple(segments)


def _fallback_segment(tokens: list[str]) -> BashSegment:
    argv: list[str] = []
    redirects: list[BashRedirect] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in {">", ">>", "<", "<<", "<<-", "<<<", "<>", ">|", "&>", "&>>"}:
            destination = tokens[index + 1] if index + 1 < len(tokens) else ""
            redirects.append(
                BashRedirect(
                    kind="heredoc" if token in {"<<", "<<-"} else "file",
                    operator=token,
                    destination=destination,
                    descriptor=None,
                    text=f"{token} {destination}".strip(),
                )
            )
            index += 2
            continue
        argv.append(token)
        index += 1
    return BashSegment(
        argv=tuple(argv),
        text=" ".join(tokens),
        redirects=tuple(redirects),
        parser="shlex-fallback",
    )
