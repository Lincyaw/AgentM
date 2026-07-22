# code-health: ignore-file[AM022] -- ast-grep exposes dynamic parser/node objects
# code-health: ignore-file[AM025] -- parser boundaries normalize ast-grep and outline JSON shapes
"""ast-grep source parsing helpers for shell commands and code files.

The shared layer owns shell AST traversal and code outline extraction.
Code symbols come from ast-grep's bundled outline extractors; this module only
maps outline entries into IFG facts.
"""

from __future__ import annotations

import posixpath
import shlex
import json
import shutil
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from loguru import logger


@dataclass(frozen=True, slots=True)
class HostExecResult:
    stdout: str
    stderr: str
    returncode: int


HostExec = Callable[[Sequence[str], float | None], HostExecResult]

_active_host_exec: HostExec | None = None


def set_host_exec(host_exec: HostExec) -> None:
    global _active_host_exec  # noqa: PLW0603
    _active_host_exec = host_exec


SYMBOL_EXTRACTOR_VERSION = "ast-grep-outline-v2"


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

    ast-grep's built-in Bash grammar preserves shell structure such as
    pipelines, redirects, heredocs, and command substitutions. A small shlex
    fallback keeps the view usable if the optional parser package is missing.
    """

    if not source.strip():
        return ()
    try:
        root = parse_ast_grep_root(source, "bash")
        segments: list[BashSegment] = []
        _collect_segments(root, _Context(), segments)
        return tuple(segments)
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        logger.debug("ast-grep bash parse failed; falling back: {}", exc)
        return _fallback_segments(source)


def parse_ast_grep_root(source: str, language: str) -> Any:
    """Parse source text with ast-grep and return its root node."""

    from ast_grep_py import SgRoot

    return SgRoot(source, language).root()


def _collect_segments(
    node: Any,
    context: _Context,
    segments: list[BashSegment],
) -> None:
    kind = node.kind()
    if kind == "pipeline":
        _collect_pipeline(node, context, segments, tail_redirects=())
        return

    if kind == "redirected_statement":
        redirects = _node_redirects(node)
        body_children = [
            child
            for child in node.children()
            if child.is_named()
            and child.kind() not in {"file_redirect", "heredoc_redirect"}
        ]
        if len(body_children) == 1 and body_children[0].kind() == "pipeline":
            _collect_pipeline(
                body_children[0],
                context,
                segments,
                tail_redirects=redirects,
            )
            return
        for child in body_children:
            _collect_segments(
                child,
                _Context(
                    redirects=context.redirects + redirects,
                    pipeline_index=context.pipeline_index,
                    depth=context.depth,
                ),
                segments,
            )
        return

    if kind == "command":
        segment = _segment_from_command(node, context)
        if segment is not None:
            segments.append(segment)
        for child in node.children():
            if child.kind() in {
                "command_substitution",
                "process_substitution",
                "subshell",
            }:
                _collect_segments(
                    child,
                    _Context(depth=context.depth + 1),
                    segments,
                )
        return

    if kind in {"command_substitution", "process_substitution", "subshell"}:
        for child in node.children():
            if child.is_named():
                _collect_segments(
                    child,
                    _Context(depth=context.depth + 1),
                    segments,
                )
        return

    for child in node.children():
        if child.is_named():
            _collect_segments(child, context, segments)


def _collect_pipeline(
    node: Any,
    context: _Context,
    segments: list[BashSegment],
    *,
    tail_redirects: tuple[BashRedirect, ...],
) -> None:
    command_children = [
        child
        for child in node.children()
        if child.is_named() and _contains_command(child)
    ]
    if not command_children:
        return
    tail_key = _node_key(command_children[-1])
    pipeline_index = 0
    for child in node.children():
        if not child.is_named() or not _contains_command(child):
            continue
        redirects = tail_redirects if _node_key(child) == tail_key else ()
        _collect_segments(
            child,
            _Context(
                redirects=context.redirects + redirects,
                pipeline_index=pipeline_index,
                depth=context.depth,
            ),
            segments,
        )
        pipeline_index += 1


def _segment_from_command(node: Any, context: _Context) -> BashSegment | None:
    command_node = _command_name_node(node)
    command = _command_name_text(command_node)
    if not command:
        return None
    argv = [command]
    for child in node.children():
        kind = child.kind()
        if kind in {"command_name", "file_redirect", "heredoc_redirect"}:
            continue
        if child.is_named():
            argv.append(child.text())
    text_range = node.range()
    return BashSegment(
        argv=tuple(argv),
        text=node.text(),
        redirects=context.redirects + _node_redirects(node),
        parser="ast-grep-bash",
        pipeline_index=context.pipeline_index,
        depth=context.depth,
        start_byte=int(text_range.start.index),
        end_byte=int(text_range.end.index),
    )


def _command_name_node(node: Any) -> Any | None:
    for child in node.children():
        if child.kind() == "command_name":
            return child
    return None


def _command_name_text(node: Any | None) -> str:
    if node is None:
        return ""
    for child in node.children():
        if child.kind() == "word":
            return child.text()
    return node.text()


def _node_redirects(node: Any) -> tuple[BashRedirect, ...]:
    redirects: list[BashRedirect] = []
    for child in node.children():
        if child.kind() == "file_redirect":
            redirects.append(_file_redirect(child))
        elif child.kind() == "heredoc_redirect":
            redirects.append(_heredoc_redirect(child))
    return tuple(redirects)


def _file_redirect(node: Any) -> BashRedirect:
    descriptor = None
    operator = ""
    destination = ""
    for child in node.children():
        kind = child.kind()
        if kind == "file_descriptor":
            descriptor = child.text()
        elif child.is_named():
            destination = child.text()
        elif not operator:
            operator = child.text()
    return BashRedirect(
        kind="file",
        operator=operator,
        destination=destination,
        descriptor=descriptor,
        text=node.text(),
    )


def _heredoc_redirect(node: Any) -> BashRedirect:
    operator = ""
    destination = ""
    for child in node.children():
        kind = child.kind()
        if kind == "heredoc_start":
            destination = child.text()
        elif not child.is_named() and not operator:
            operator = child.text()
    return BashRedirect(
        kind="heredoc",
        operator=operator,
        destination=destination,
        descriptor=None,
        text=node.text(),
    )


def _contains_command(node: Any) -> bool:
    if node.kind() in {"command", "redirected_statement"}:
        return True
    return any(
        child.is_named() and _contains_command(child) for child in node.children()
    )


def _node_key(node: Any) -> tuple[int, int]:
    text_range = node.range()
    return int(text_range.start.index), int(text_range.end.index)


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


@dataclass(frozen=True, slots=True)
class SymbolExtractionInput:
    session_id: str
    extractor_version: str
    action_id: str
    source_unit_id: str
    path: str
    relation: str
    turn: int
    event_id: int | None
    tool_name: str
    content_hash: str | None
    unit_hash: str | None
    content_text: str | None
    metadata: Mapping[str, object]
    raw_evidence: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class SymbolFact:
    session_id: str
    extractor_version: str
    action_id: str
    source_unit_id: str
    path: str
    stable_key: str
    kind: str
    qualified_name: str
    file_relation: str
    action_relation: str
    symbol_relation: str | None
    turn: int
    event_id: int | None
    source: str
    confidence: str
    metadata: Mapping[str, object]
    raw_evidence: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class SymbolExtractionResult:
    symbols: tuple[SymbolFact, ...]
    errors: tuple[Mapping[str, object], ...]


def extract_symbols_from_source_units(
    source_units: Sequence[SymbolExtractionInput],
    *,
    extractor_version: str,
) -> SymbolExtractionResult:
    """Extract symbol facts from ast-grep's bundled outline rules."""

    units = tuple(unit for unit in source_units if unit.content_text)
    if not units:
        return SymbolExtractionResult(symbols=(), errors=())
    binary = _ast_grep_binary()
    if binary is None:
        return SymbolExtractionResult(
            symbols=(),
            errors=tuple(
                _symbol_error(
                    unit,
                    "symbol_extraction:ast_grep_cli_unavailable",
                    raw_evidence={"path": unit.path},
                )
                for unit in units
            ),
        )

    with tempfile.TemporaryDirectory(prefix="agentm-ifg-outline-") as tmp:
        tmp_path = Path(tmp)
        path_to_unit: dict[str, SymbolExtractionInput] = {}
        for index, unit in enumerate(units):
            suffix = _source_suffix(unit.path)
            temp_file = tmp_path / f"unit_{index:05d}{suffix}"
            temp_file.write_text(unit.content_text or "", encoding="utf-8")
            path_to_unit[str(temp_file)] = unit
            path_to_unit[str(temp_file.resolve())] = unit
            path_to_unit[temp_file.name] = unit

        argv = [
            binary,
            "outline",
            str(tmp_path),
            "--json=compact",
            "--items",
            "all",
            "--view",
            "expanded",
            "--threads",
            "1",
        ]
        try:
            completed = _run_outline(argv, timeout=30)
        except (OSError, TimeoutError) as exc:
            return SymbolExtractionResult(
                symbols=(),
                errors=tuple(
                    _symbol_error(
                        unit,
                        f"symbol_extraction:outline_failed:{type(exc).__name__}:{exc}",
                        raw_evidence={"path": unit.path},
                    )
                    for unit in units
                ),
            )
        if completed.returncode != 0:
            return SymbolExtractionResult(
                symbols=(),
                errors=tuple(
                    _symbol_error(
                        unit,
                        "symbol_extraction:outline_failed",
                        raw_evidence={
                            "path": unit.path,
                            "stderr": _preview(completed.stderr),
                        },
                    )
                    for unit in units
                ),
            )
        try:
            outlined = json.loads(completed.stdout or "[]")
        except json.JSONDecodeError as exc:
            return SymbolExtractionResult(
                symbols=(),
                errors=tuple(
                    _symbol_error(
                        unit,
                        f"symbol_extraction:outline_json_failed:{exc}",
                        raw_evidence={
                            "path": unit.path,
                            "stdout": _preview(completed.stdout),
                        },
                    )
                    for unit in units
                ),
            )

    return SymbolExtractionResult(
        symbols=tuple(
            _outline_symbol_facts(
                outlined,
                path_to_unit=path_to_unit,
                extractor_version=extractor_version,
            )
        ),
        errors=(),
    )


def extract_symbols_from_repository_files(
    source_units: Sequence[SymbolExtractionInput],
    *,
    extractor_version: str,
) -> SymbolExtractionResult:
    """Extract validated symbols from full files available in the live repository."""

    units = tuple(unit for unit in source_units if Path(unit.path).is_file())
    if not units:
        return SymbolExtractionResult(symbols=(), errors=())
    binary = _ast_grep_binary()
    if binary is None:
        return SymbolExtractionResult(
            symbols=(),
            errors=tuple(
                _symbol_error(
                    unit,
                    "repository_validation:ast_grep_cli_unavailable",
                    raw_evidence={"path": unit.path},
                )
                for unit in units
            ),
        )

    path_to_unit: dict[str, SymbolExtractionInput] = {}
    paths: list[str] = []
    for unit in units:
        path = Path(unit.path)
        resolved = str(path.resolve())
        paths.append(resolved)
        path_to_unit[unit.path] = unit
        path_to_unit[resolved] = unit
        path_to_unit[path.name] = unit

    argv = [
        binary,
        "outline",
        *paths,
        "--json=compact",
        "--items",
        "all",
        "--view",
        "expanded",
        "--threads",
        "1",
    ]
    try:
        completed = _run_outline(argv, timeout=30)
    except (OSError, TimeoutError) as exc:
        return SymbolExtractionResult(
            symbols=(),
            errors=tuple(
                _symbol_error(
                    unit,
                    f"repository_validation:outline_failed:{type(exc).__name__}:{exc}",
                    raw_evidence={"path": unit.path},
                )
                for unit in units
            ),
        )
    if completed.returncode != 0:
        return SymbolExtractionResult(
            symbols=(),
            errors=tuple(
                _symbol_error(
                    unit,
                    "repository_validation:outline_failed",
                    raw_evidence={
                        "path": unit.path,
                        "stderr": _preview(completed.stderr),
                    },
                )
                for unit in units
            ),
        )
    try:
        outlined = json.loads(completed.stdout or "[]")
    except json.JSONDecodeError as exc:
        return SymbolExtractionResult(
            symbols=(),
            errors=tuple(
                _symbol_error(
                    unit,
                    f"repository_validation:outline_json_failed:{exc}",
                    raw_evidence={
                        "path": unit.path,
                        "stdout": _preview(completed.stdout),
                    },
                )
                for unit in units
            ),
        )

    facts = _outline_symbol_facts(
        outlined,
        path_to_unit=path_to_unit,
        extractor_version=extractor_version,
    )
    return SymbolExtractionResult(
        symbols=tuple(_repository_validated_fact(fact) for fact in facts),
        errors=(),
    )


def extract_symbols_from_repository_outline(
    source_units: Sequence[SymbolExtractionInput],
    *,
    documents_by_path: Mapping[str, Mapping[str, object]],
    extractor_version: str,
) -> SymbolExtractionResult:
    """Map cached repository outline documents onto IFG source units."""

    facts: list[SymbolFact] = []
    for unit in source_units:
        document = documents_by_path.get(posixpath.normpath(unit.path))
        if document is None:
            continue
        raw_path = _mapping_str(document, "path")
        path_to_unit = {
            unit.path: unit,
            posixpath.normpath(unit.path): unit,
            Path(unit.path).name: unit,
        }
        if raw_path:
            path_to_unit[raw_path] = unit
        outlined_facts = _outline_symbol_facts(
            (document,),
            path_to_unit=path_to_unit,
            extractor_version=extractor_version,
        )
        facts.extend(_repository_validated_fact(fact) for fact in outlined_facts)
    return SymbolExtractionResult(symbols=tuple(facts), errors=())


def _repository_validated_fact(fact: SymbolFact) -> SymbolFact:
    language = fact.metadata.get("language") or "unknown"
    return replace(
        fact,
        source=f"ast-grep:repository-outline:{language}",
        confidence="high",
        metadata={
            **dict(fact.metadata),
            "content_scope": "full_file",
            "validation": "repository_present",
            "link_action": False,
        },
        raw_evidence={
            **dict(fact.raw_evidence),
            "validation": "repository_present",
        },
    )


def _outline_symbol_facts(
    outlined: object,
    *,
    path_to_unit: Mapping[str, SymbolExtractionInput],
    extractor_version: str,
) -> Iterable[SymbolFact]:
    if not isinstance(outlined, Sequence) or isinstance(outlined, (str, bytes)):
        return ()
    facts: list[SymbolFact] = []
    for document in outlined:
        if not isinstance(document, Mapping):
            continue
        unit = _unit_for_outline_document(document, path_to_unit)
        if unit is None:
            continue
        language = _mapping_str(document, "language")
        for entry, parent_name in _outline_entries(document.get("items")):
            if not isinstance(entry, Mapping):
                continue
            name = _mapping_str(entry, "name")
            if not name:
                continue
            kind = _outline_kind(entry)
            qualified_name = f"{parent_name}.{name}" if parent_name else name
            file_relation = _outline_file_relation(entry)
            content_scope = unit.metadata.get("content_scope")
            if not isinstance(content_scope, str) or not content_scope:
                content_scope = "fragment"
            repo, file_key = _repo_and_file_key(unit.path, unit.metadata)
            stable_key = _stable_symbol_key(
                repo=repo,
                file_key=file_key,
                kind=kind,
                qualified_name=qualified_name,
            )
            span = _outline_span(entry.get("range"))
            facts.append(
                SymbolFact(
                    session_id=unit.session_id,
                    extractor_version=extractor_version,
                    action_id=unit.action_id,
                    source_unit_id=unit.source_unit_id,
                    path=unit.path,
                    stable_key=stable_key,
                    kind=kind,
                    qualified_name=qualified_name,
                    file_relation=file_relation,
                    action_relation=_action_symbol_relation(unit.relation),
                    symbol_relation=None,
                    turn=unit.turn,
                    event_id=unit.event_id,
                    source=f"ast-grep:outline:{language or 'unknown'}",
                    confidence=("high" if content_scope == "full_file" else "medium"),
                    metadata={
                        "repo": repo,
                        "file_key": file_key,
                        "language": language,
                        "role": _mapping_str(entry, "role"),
                        "symbol_type": _mapping_str(entry, "symbolType"),
                        "short_name": name,
                        "parent_name": parent_name,
                        "signature": _mapping_str(entry, "signature"),
                        "ast_kind": _mapping_str(entry, "astKind"),
                        "is_import": _mapping_bool(entry, "isImport"),
                        "is_exported": _mapping_bool(entry, "isExported"),
                        "is_public": _mapping_bool(entry, "isPublic"),
                        "span": span,
                        "content_hash": unit.content_hash,
                        "unit_hash": unit.unit_hash,
                        "source_unit_relation": unit.relation,
                        "content_scope": content_scope,
                        "validation": "trajectory_observed",
                        "link_action": True,
                    },
                    raw_evidence={
                        "source_unit_id": unit.source_unit_id,
                        "path": unit.path,
                        "tool_name": unit.tool_name,
                        "outline_path": _mapping_str(document, "path"),
                        "outline_entry": dict(entry),
                        "raw_source_unit": unit.raw_evidence,
                    },
                )
            )
    return tuple(facts)


def _unit_for_outline_document(
    document: Mapping[object, object],
    path_to_unit: Mapping[str, SymbolExtractionInput],
) -> SymbolExtractionInput | None:
    raw_path = _mapping_str(document, "path")
    if not raw_path:
        return None
    candidates = [raw_path]
    try:
        candidates.append(str(Path(raw_path).resolve()))
    except OSError:
        pass
    candidates.append(Path(raw_path).name)
    for candidate in candidates:
        unit = path_to_unit.get(candidate)
        if unit is not None:
            return unit
    return None


def _outline_entries(
    items: object,
    *,
    parent_name: str | None = None,
) -> Iterable[tuple[Mapping[str, object], str | None]]:
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        return ()
    entries: list[tuple[Mapping[str, object], str | None]] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        item_dict = dict(item)
        entries.append((item_dict, parent_name))
        name = _mapping_str(item_dict, "name")
        next_parent = f"{parent_name}.{name}" if parent_name and name else name
        entries.extend(
            _outline_entries(item_dict.get("members"), parent_name=next_parent)
        )
    return tuple(entries)


def _outline_kind(entry: Mapping[str, object]) -> str:
    raw = _mapping_str(entry, "symbolType")
    if not raw:
        return "symbol"
    return raw[:1].lower() + raw[1:]


def _outline_file_relation(entry: Mapping[str, object]) -> str:
    if _mapping_bool(entry, "isImport"):
        return "imports"
    if _mapping_bool(entry, "isExported"):
        return "exports"
    return "defines"


def _outline_span(raw_range: object) -> Mapping[str, int]:
    if not isinstance(raw_range, Mapping):
        return {}
    start = raw_range.get("start")
    end = raw_range.get("end")
    byte_offset = raw_range.get("byteOffset")
    span: dict[str, int] = {}
    if isinstance(start, Mapping):
        start_line = _int_value(start.get("line"))
        start_column = _int_value(start.get("column"))
        if start_line is not None:
            span["start_line"] = start_line + 1
        if start_column is not None:
            span["start_column"] = start_column + 1
    if isinstance(end, Mapping):
        end_line = _int_value(end.get("line"))
        end_column = _int_value(end.get("column"))
        if end_line is not None:
            span["end_line"] = end_line + 1
        if end_column is not None:
            span["end_column"] = end_column + 1
    if isinstance(byte_offset, Mapping):
        start_index = _int_value(byte_offset.get("start"))
        end_index = _int_value(byte_offset.get("end"))
        if start_index is not None:
            span["start_index"] = start_index
        if end_index is not None:
            span["end_index"] = end_index
    return span


def _ast_grep_binary() -> str | None:
    return shutil.which("ast-grep") or shutil.which("sg")


def _run_outline(
    argv: Sequence[str],
    *,
    timeout: float | None,
) -> HostExecResult:
    if _active_host_exec is not None:
        return _active_host_exec(argv, timeout)
    import subprocess  # noqa: PLC0415

    try:
        completed = subprocess.run(
            list(argv),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(str(exc)) from exc
    return HostExecResult(
        stdout=completed.stdout,
        stderr=completed.stderr,
        returncode=completed.returncode,
    )


def _source_suffix(path: str) -> str:
    lower = path.lower()
    for suffix in (".d.ts", ".test.tsx", ".spec.tsx", ".tsx", ".ts", ".jsx", ".js"):
        if lower.endswith(suffix):
            return suffix
    suffix = Path(path).suffix
    return suffix if suffix else ".txt"


def _repo_and_file_key(
    path: str,
    metadata: Mapping[str, object],
) -> tuple[str, str]:
    cwd = metadata.get("cwd")
    repo = str(cwd) if isinstance(cwd, str) and cwd.startswith("/") else ""
    normalized_path = posixpath.normpath(path)
    if repo and normalized_path == repo:
        return repo, "."
    if repo and normalized_path.startswith(f"{repo}/"):
        return repo, posixpath.relpath(normalized_path, repo)
    return repo, normalized_path


def _stable_symbol_key(
    *,
    repo: str,
    file_key: str,
    kind: str,
    qualified_name: str,
) -> str:
    return f"repo:{repo}\x1ffile:{file_key}\x1fkind:{kind}\x1fname:{qualified_name}"


def _action_symbol_relation(file_relation: str) -> str:
    if file_relation in {"read", "reference"}:
        return "read"
    if file_relation in {"write", "create", "edit", "delete"}:
        return "write"
    return "read"


def _mapping_str(value: Mapping[object, object], key: str) -> str | None:
    raw = value.get(key)
    return raw if isinstance(raw, str) and raw else None


def _mapping_bool(value: Mapping[str, object], key: str) -> bool:
    raw = value.get(key)
    return raw if isinstance(raw, bool) else False


def _int_value(value: object) -> int | None:
    if isinstance(value, int):
        return value
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _preview(text: str, *, limit: int = 500) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


def _symbol_error(
    source_unit: SymbolExtractionInput,
    error: str,
    *,
    raw_evidence: Mapping[str, object],
) -> Mapping[str, object]:
    return {
        "turn": source_unit.turn,
        "event_id": source_unit.event_id,
        "tool_call_id": source_unit.action_id,
        "tool_name": source_unit.tool_name,
        "error": error,
        "raw_evidence": {
            "action_id": source_unit.action_id,
            "source_unit_id": source_unit.source_unit_id,
            **dict(raw_evidence),
        },
    }
