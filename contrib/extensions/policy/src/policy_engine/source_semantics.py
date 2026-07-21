# code-health: ignore-file[AM025] -- command schema is intentionally untyped
"""Schema-driven read/write extraction for parsed bash segments.

The parser layer owns shell syntax. This module interprets parsed argv,
redirects, and pipeline position using a small YAML command schema plus a
generic fallback for unknown commands. Core file relations are intentionally
coarse: read or write.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from policy_engine.source_parser import BashRedirect, BashSegment


BASH_SEMANTICS_EXTRACTOR_VERSION = "bash-semantics-v4"
EXTRACTOR_VERSION = BASH_SEMANTICS_EXTRACTOR_VERSION

_WRITE_REDIRECTS = frozenset({">", ">>", "<>", ">|", "&>", "&>>"})
_READ_REDIRECTS = frozenset({"<", "<>"})
_HEX_RE = re.compile(r"[0-9a-fA-F]{7,64}")
_NUMERIC_RE = re.compile(r"\d+(?:\.\d+)?")
_SHORT_NUMERIC_FLAG_RE = re.compile(r"-\d+")
_SED_RANGE_RE = re.compile(r"['\"]?\d+(?:,\d+)?[pP]['\"]?")
_IDENTIFIER_RE = re.compile(r"\b[A-Za-z_$][A-Za-z0-9_$]{2,}\b")
_PATH_ROLES = frozenset({"path", "path-pattern"})
_READ_LIKE_ACTIONS = frozenset({"read", "reference"})
_WRITE_LIKE_ACTIONS = frozenset({"write", "edit", "create", "delete"})


@dataclass(frozen=True, slots=True)
class BashPathReference:
    path: str
    path_kind: str
    relation: str
    source: str
    confidence: str


@dataclass(frozen=True, slots=True)
class BashSymbolMention:
    text: str
    source: str
    confidence: str


@dataclass(frozen=True, slots=True)
class BashSegmentSemantics:
    command: str
    action_kind: str
    family: str
    confidence: str
    template_tokens: tuple[str, ...]
    template: str
    path_refs: tuple[BashPathReference, ...]
    symbol_mentions: tuple[BashSymbolMention, ...]


def analyze_bash_segment(
    segment: BashSegment,
    *,
    schema: Mapping[str, object] | None = None,
) -> BashSegmentSemantics:
    """Classify one parsed bash segment for IFG extraction."""

    command = command_name(segment.argv)
    schema = load_bash_command_schema() if schema is None else schema
    command_schema = _command_schema(schema, command)
    action_kind = _segment_action_kind(segment, schema=schema)
    if _is_stdin_filter(segment, action_kind, schema=schema):
        action_kind = "filter"
    confidence = _segment_confidence(segment, command_schema=command_schema)
    template_tokens = _segment_template_tokens(segment, schema=schema)
    path_refs = tuple(
        _dedupe_path_refs(
            _segment_path_refs(
                segment,
                action_kind,
                schema=schema,
                command_confidence=confidence,
            )
        )
    )
    return BashSegmentSemantics(
        command=command,
        action_kind=action_kind,
        family=_family_for_action(action_kind),
        confidence=confidence,
        template_tokens=template_tokens,
        template=" ".join(template_tokens),
        path_refs=path_refs,
        symbol_mentions=tuple(_symbol_mentions(segment, schema=schema)),
    )


@lru_cache(maxsize=1)
def load_bash_command_schema() -> Mapping[str, object]:
    path = Path(__file__).with_name("bash_command_schema.yaml")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, Mapping) else {}


def command_name(command: Sequence[str]) -> str:
    if not command:
        return ""
    return Path(command[0]).name


def _segment_action_kind(
    segment: BashSegment,
    *,
    schema: Mapping[str, object],
) -> str:
    if _has_write_redirect(segment):
        return "write"
    return _command_action_kind(segment.argv, schema=schema)


def _command_action_kind(
    command: Sequence[str],
    *,
    schema: Mapping[str, object],
) -> str:
    name = command_name(command)
    if not name:
        return "reference"
    command_schema = _command_schema(schema, name)
    strategy = str(command_schema.get("strategy") or "generic")
    if strategy == "timeout":
        return _command_action_kind(_timeout_inner_command(command), schema=schema)
    if strategy == "git":
        subcommand = command[1] if len(command) > 1 else ""
        read_subcommands = _string_set(command_schema.get("read_subcommands"))
        read_subcommands |= _string_set(command_schema.get("query_subcommands"))
        return (
            "read" if subcommand in read_subcommands else _schema_action(command_schema)
        )
    if strategy == "package":
        test_subcommands = _string_set(command_schema.get("test_subcommands"))
        if any(token in test_subcommands for token in command[1:]):
            return "test"
    if _schema_option_triggers_edit(command, command_schema):
        return "write"
    if command_schema:
        return _schema_action(command_schema)
    if _generic_path_operands(command):
        return "reference"
    return "exec"


def _schema_action(command_schema: Mapping[str, object]) -> str:
    action = command_schema.get("action")
    return action if isinstance(action, str) and action else "exec"


def _schema_option_triggers_edit(
    command: Sequence[str],
    command_schema: Mapping[str, object],
) -> bool:
    prefixes = tuple(_string_set(command_schema.get("edit_when_option_prefix")))
    contains = tuple(_string_set(command_schema.get("edit_when_option_contains")))
    for token in command[1:]:
        if prefixes and any(token.startswith(prefix) for prefix in prefixes):
            return True
        if (
            contains
            and token.startswith("-")
            and any(marker in token for marker in contains)
        ):
            return True
    return False


def _family_for_action(action_kind: str) -> str:
    if action_kind in _WRITE_LIKE_ACTIONS:
        return "write"
    if action_kind in _READ_LIKE_ACTIONS or action_kind == "filter":
        return "read"
    if action_kind in {"control", "exec", "test"}:
        return action_kind
    return "read"


def _segment_confidence(
    segment: BashSegment,
    *,
    command_schema: Mapping[str, object],
) -> str:
    if _has_write_redirect(segment):
        return "medium"
    return "medium" if command_schema else "low"


def _is_stdin_filter(
    segment: BashSegment,
    action: str,
    *,
    schema: Mapping[str, object],
) -> bool:
    if action not in _READ_LIKE_ACTIONS:
        return False
    if segment.pipeline_index is None or segment.pipeline_index == 0:
        return False
    if _has_file_operand(segment.argv, schema=schema):
        return False
    if any(
        redirect.kind == "file" and redirect.operator in _READ_REDIRECTS
        for redirect in segment.redirects
    ):
        return False
    return True


def _has_file_operand(
    command: Sequence[str],
    *,
    schema: Mapping[str, object],
) -> bool:
    roles = _command_token_roles(command, schema=schema)
    return any(role in _PATH_ROLES for role in roles.values())


def _has_write_redirect(segment: BashSegment) -> bool:
    for redirect in segment.redirects:
        if redirect.kind != "file" or redirect.operator not in _WRITE_REDIRECTS:
            continue
        if redirect.descriptor == "2":
            continue
        if _is_virtual_path(_clean_token(redirect.destination) or ""):
            continue
        return True
    return False


def _segment_template_tokens(
    segment: BashSegment,
    *,
    schema: Mapping[str, object],
) -> tuple[str, ...]:
    tokens = list(_command_template_tokens(segment.argv, schema=schema))
    for redirect in segment.redirects:
        operator = redirect.operator
        if redirect.descriptor:
            operator = f"{redirect.descriptor}{operator}"
        if operator:
            tokens.append(operator)
        if redirect.kind == "heredoc":
            tokens.append("<heredoc>")
        elif redirect.destination:
            tokens.append(_normalize_token(redirect.destination, "path"))
    return tuple(tokens)


def _command_template_tokens(
    command: Sequence[str],
    *,
    schema: Mapping[str, object],
) -> tuple[str, ...]:
    roles = _command_token_roles(command, schema=schema)
    return tuple(
        _normalize_token(token, roles.get(index, "auto"))
        for index, token in enumerate(command)
    )


def _command_token_roles(
    command: Sequence[str],
    *,
    schema: Mapping[str, object],
) -> Mapping[int, str]:
    name = command_name(command)
    if not command:
        return {}
    command_schema = _command_schema(schema, name)
    strategy = str(command_schema.get("strategy") or "generic")
    if strategy == "timeout":
        return _timeout_token_roles(command, schema=schema)
    if strategy == "git":
        return _git_token_roles(command, command_schema=command_schema)
    if strategy == "search":
        return _search_token_roles(
            command, schema=schema, command_schema=command_schema
        )
    if strategy == "find":
        return _find_token_roles(command, command_schema=command_schema)
    if strategy == "sed":
        return _sed_token_roles(command)
    if strategy in {"file_operands", "mixed_operands", "path_scope"}:
        return _file_operand_roles(command)
    if strategy == "source_dest":
        return _source_dest_roles(command)
    if strategy == "package":
        return _package_token_roles(command)
    return _generic_token_roles(command)


def _generic_token_roles(command: Sequence[str]) -> dict[int, str]:
    roles: dict[int, str] = {0: "preserve"}
    for index, token in enumerate(command[1:], start=1):
        if token.startswith("-") and "=" not in token:
            roles[index] = "preserve"
        elif _is_path_like(token):
            roles[index] = "path-pattern" if _has_glob(token) else "path"
    return roles


def _timeout_token_roles(
    command: Sequence[str],
    *,
    schema: Mapping[str, object],
) -> dict[int, str]:
    roles: dict[int, str] = {0: "preserve"}
    inner = _timeout_inner_index(command)
    for index in range(1, inner):
        roles[index] = "num" if _looks_numeric(command[index]) else "preserve"
    for offset, role in _command_token_roles(command[inner:], schema=schema).items():
        roles[inner + offset] = role
    return roles


def _git_token_roles(
    command: Sequence[str],
    *,
    command_schema: Mapping[str, object],
) -> dict[int, str]:
    roles = _generic_token_roles(command)
    if len(command) > 1:
        roles[1] = "preserve"
    after_double_dash = False
    for index, token in enumerate(command[2:], start=2):
        if token == "--":
            roles[index] = "preserve"
            after_double_dash = True
            continue
        if after_double_dash:
            roles[index] = "path"
        elif _SHORT_NUMERIC_FLAG_RE.fullmatch(token):
            roles[index] = "num-flag"
        elif token.startswith("-"):
            roles[index] = "preserve"
        elif _HEX_RE.fullmatch(_strip_quotes(token)):
            roles[index] = "hash"
    if (
        command_schema.get("read_subcommands")
        or command_schema.get("query_subcommands")
    ) and len(command) > 1:
        roles[1] = "preserve"
    return roles


def _search_token_roles(
    command: Sequence[str],
    *,
    schema: Mapping[str, object],
    command_schema: Mapping[str, object],
) -> dict[int, str]:
    roles = _generic_token_roles(command)
    option_roles = _option_value_roles(schema, command_schema)
    pattern_seen = False
    skip_next = False
    for index, token in enumerate(command[1:], start=1):
        if skip_next:
            skip_next = False
            continue
        if token == "--":
            roles[index] = "preserve"
            continue
        key = token.split("=", 1)[0]
        if key in option_roles and "=" not in token:
            roles[index] = "preserve"
            if index + 1 < len(command):
                roles[index + 1] = option_roles[key]
                skip_next = True
            continue
        if token.startswith("-") and "=" not in token:
            roles[index] = "preserve"
            continue
        if token.startswith("-") and "=" in token:
            roles[index] = "flag-assignment"
            continue
        if not pattern_seen:
            roles[index] = "pattern"
            pattern_seen = True
        else:
            roles[index] = "path-pattern" if _has_glob(token) else "path"
    return roles


def _find_token_roles(
    command: Sequence[str],
    *,
    command_schema: Mapping[str, object],
) -> dict[int, str]:
    roles = _generic_token_roles(command)
    pattern_flags = _string_set(command_schema.get("pattern_flags"))
    numeric_flags = _string_set(command_schema.get("numeric_flags"))
    in_predicates = False
    skip_next = False
    for index, token in enumerate(command[1:], start=1):
        if skip_next:
            skip_next = False
            continue
        if token in pattern_flags:
            roles[index] = "preserve"
            if index + 1 < len(command):
                roles[index + 1] = "pattern"
                skip_next = True
            in_predicates = True
            continue
        if token in numeric_flags:
            roles[index] = "preserve"
            if index + 1 < len(command):
                roles[index + 1] = "num"
                skip_next = True
            in_predicates = True
            continue
        if token.startswith("-") or token in {"!", "(", ")"}:
            roles[index] = "preserve"
            in_predicates = True
            continue
        if not in_predicates:
            roles[index] = "path"
    return roles


def _sed_token_roles(command: Sequence[str]) -> dict[int, str]:
    roles = _generic_token_roles(command)
    script_seen = False
    for index, token in enumerate(command[1:], start=1):
        if token.startswith("-") and "=" not in token:
            roles[index] = "preserve"
            continue
        if not script_seen:
            roles[index] = (
                "range" if _SED_RANGE_RE.fullmatch(_strip_quotes(token)) else "arg"
            )
            script_seen = True
        else:
            roles[index] = "path"
    return roles


def _file_operand_roles(command: Sequence[str]) -> dict[int, str]:
    roles = _generic_token_roles(command)
    for index, token in enumerate(command[1:], start=1):
        if token.startswith("-") and "=" not in token:
            roles[index] = (
                "num-flag" if _SHORT_NUMERIC_FLAG_RE.fullmatch(token) else "preserve"
            )
        else:
            roles[index] = "path-pattern" if _has_glob(token) else "path"
    return roles


def _source_dest_roles(command: Sequence[str]) -> dict[int, str]:
    roles = _generic_token_roles(command)
    for index, token in enumerate(command[1:], start=1):
        if token.startswith("-") and "=" not in token:
            roles[index] = "preserve"
        else:
            roles[index] = "path-pattern" if _has_glob(token) else "path"
    return roles


def _package_token_roles(command: Sequence[str]) -> dict[int, str]:
    roles = _generic_token_roles(command)
    for index, token in enumerate(command[1:3], start=1):
        if not token.startswith("-"):
            roles[index] = "preserve"
    for index, token in enumerate(command[3:], start=3):
        if token.startswith("-") and "=" not in token:
            roles[index] = "preserve"
        elif _is_path_like(token):
            roles[index] = "path-pattern" if _has_glob(token) else "path"
    return roles


def _option_value_roles(
    schema: Mapping[str, object],
    command_schema: Mapping[str, object],
) -> dict[str, str]:
    roles: dict[str, str] = {}
    defaults = schema.get("defaults")
    if isinstance(defaults, Mapping):
        roles.update(_string_mapping(defaults.get("option_value_roles")))
    roles.update(_string_mapping(command_schema.get("option_value_roles")))
    return roles


def _normalize_token(token: str, role: str) -> str:
    if role == "preserve":
        return token
    if role == "flag-assignment":
        key, value = token.split("=", 1)
        value_role = _option_value_roles(load_bash_command_schema(), {}).get(
            key, "auto"
        )
        return f"{key}={_normalize_token(value, value_role)}"
    if role == "num-flag":
        return "-<num>"
    if role in {"path", "path-pattern", "pattern", "num", "hash", "range", "arg"}:
        return f"<{role}>"
    if token.startswith("-") and "=" in token:
        key, value = token.split("=", 1)
        return f"{key}={_normalize_auto(value)}"
    if token.startswith("-") and _SHORT_NUMERIC_FLAG_RE.fullmatch(token):
        return "-<num>"
    if token.startswith("-"):
        return token
    return _normalize_auto(token)


def _normalize_auto(token: str) -> str:
    clean = _strip_quotes(token)
    if not clean:
        return "<arg>"
    if _NUMERIC_RE.fullmatch(clean):
        return "<num>"
    if _HEX_RE.fullmatch(clean):
        return "<hash>"
    if _has_glob(clean) and _is_path_like(clean):
        return "<path-pattern>"
    if _has_glob(clean):
        return "<pattern>"
    if _is_path_like(clean):
        return "<path>"
    if any(ch.isspace() for ch in clean) or len(clean) > 80:
        return "<arg>"
    if any(ch in clean for ch in "{}()[],:"):
        return "<arg>"
    return clean


def _segment_path_refs(
    segment: BashSegment,
    action_kind: str,
    *,
    schema: Mapping[str, object],
    command_confidence: str,
) -> Iterable[BashPathReference]:
    for redirect in segment.redirects:
        yield from _redirect_path_refs(redirect)

    command = list(segment.argv)
    command_schema = _command_schema(schema, command_name(command))
    strategy = str(command_schema.get("strategy") or "generic")
    if strategy == "timeout":
        fake = BashSegment(
            argv=tuple(_timeout_inner_command(command)),
            text=segment.text,
            redirects=(),
            parser=segment.parser,
            pipeline_index=segment.pipeline_index,
            depth=segment.depth,
            start_byte=segment.start_byte,
            end_byte=segment.end_byte,
        )
        yield from _segment_path_refs(
            fake,
            _command_action_kind(fake.argv, schema=schema),
            schema=schema,
            command_confidence=command_confidence,
        )
        return

    if action_kind == "filter":
        return
    if strategy == "source_dest":
        operands = _path_operands(command, schema=schema)
        if operands:
            yield BashPathReference(
                operands[0],
                _operand_path_kind(operands[0], strategy=strategy),
                "read",
                "cmd.arg.source",
                command_confidence,
            )
        for path in operands[1:]:
            relation = "write" if action_kind in _WRITE_LIKE_ACTIONS else action_kind
            yield BashPathReference(
                path,
                _operand_path_kind(path, strategy=strategy),
                relation,
                "cmd.arg.dest",
                command_confidence,
            )
        return

    relation = _file_relation_for_action(action_kind)
    if relation is None:
        relation = "reference" if not command_schema else None
    if relation is None:
        return
    for path in _path_operands(command, schema=schema):
        confidence = command_confidence if command_schema else "low"
        yield BashPathReference(
            path,
            _operand_path_kind(path, strategy=strategy),
            relation,
            f"cmd.{relation}",
            confidence,
        )


def _redirect_path_refs(redirect: BashRedirect) -> Iterable[BashPathReference]:
    if redirect.kind != "file":
        return ()
    path = _clean_token(redirect.destination)
    if not path or _is_virtual_path(path):
        return ()
    if redirect.operator in _READ_REDIRECTS:
        return (BashPathReference(path, "file", "read", "cmd.redirect.read", "medium"),)
    if redirect.operator in _WRITE_REDIRECTS and redirect.descriptor != "2":
        return (
            BashPathReference(path, "file", "write", "cmd.redirect.write", "medium"),
        )
    return ()


def _file_relation_for_action(action_kind: str) -> str | None:
    if action_kind in _READ_LIKE_ACTIONS:
        return "read"
    if action_kind in _WRITE_LIKE_ACTIONS:
        return "write"
    return None


def _path_operands(
    command: Sequence[str],
    *,
    schema: Mapping[str, object],
) -> list[str]:
    roles = _command_token_roles(command, schema=schema)
    paths: list[str] = []
    for index, token in enumerate(command):
        if roles.get(index) not in _PATH_ROLES:
            continue
        clean = _clean_token(token)
        if clean and not _is_virtual_path(clean):
            paths.append(clean)
    return paths


def _generic_path_operands(command: Sequence[str]) -> list[str]:
    paths: list[str] = []
    for token in command[1:]:
        clean = _clean_token(token)
        if clean and _is_path_like(clean):
            paths.append(clean)
    return paths


def _dedupe_path_refs(refs: Iterable[BashPathReference]) -> Iterable[BashPathReference]:
    seen: set[tuple[str, str, str, str]] = set()
    for ref in refs:
        key = (ref.path, ref.path_kind, ref.relation, ref.source)
        if key in seen:
            continue
        seen.add(key)
        yield ref


def _operand_path_kind(path: str, *, strategy: str) -> str:
    if _has_glob(path):
        return "pattern"
    if strategy in {"find", "path_scope"}:
        return "directory"
    if strategy in {"search", "git", "source_dest", "mixed_operands", "generic"}:
        return "file" if _looks_like_concrete_file(path) else "directory"
    return "file"


def _looks_like_concrete_file(path: str) -> bool:
    clean = _clean_token(path)
    if not clean or clean.endswith("/") or clean in {".", "..", "/"}:
        return False
    name = Path(clean).name
    if name in {
        "BUILD",
        "BUILD.bazel",
        "Dockerfile",
        "Gemfile",
        "Makefile",
        "Rakefile",
        "WORKSPACE",
    }:
        return True
    if name in {
        ".dockerignore",
        ".editorconfig",
        ".eslintignore",
        ".eslintrc",
        ".gitignore",
        ".npmrc",
        ".prettierignore",
        ".prettierrc",
    }:
        return True
    return not name.startswith(".") and bool(Path(name).suffix)


def _symbol_mentions(
    segment: BashSegment,
    *,
    schema: Mapping[str, object],
) -> Iterable[BashSymbolMention]:
    roles = _command_token_roles(segment.argv, schema=schema)
    for index, token in enumerate(segment.argv):
        if roles.get(index) != "pattern":
            continue
        clean = _strip_quotes(token)
        for match in _IDENTIFIER_RE.finditer(clean):
            yield BashSymbolMention(match.group(0), "cmd.pattern", "low")


def _command_schema(
    schema: Mapping[str, object],
    command: str,
) -> Mapping[str, object]:
    commands = schema.get("commands")
    if not isinstance(commands, Mapping):
        return {}
    raw = commands.get(command)
    return raw if isinstance(raw, Mapping) else {}


def _timeout_inner_command(command: Sequence[str]) -> Sequence[str]:
    return command[_timeout_inner_index(command) :]


def _timeout_inner_index(command: Sequence[str]) -> int:
    for index, token in enumerate(command[1:], start=1):
        if token.startswith("-") or _looks_numeric(token):
            continue
        return index
    return len(command)


def _looks_numeric(token: str) -> bool:
    return _NUMERIC_RE.fullmatch(_strip_quotes(token)) is not None


def _clean_token(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    token = value.strip().strip("\"'`,;:()[]{}")
    if not token:
        return None
    if token == "--" or token.startswith("-") or "://" in token:
        return None
    return token


def _strip_quotes(value: str) -> str:
    token = value.strip()
    if len(token) >= 2 and token[0] == token[-1] and token[0] in {"'", '"'}:
        return token[1:-1]
    return token


def _has_glob(token: str) -> bool:
    return any(char in _strip_quotes(token) for char in "*?[]")


def _is_path_like(token: str) -> bool:
    clean = _clean_token(token)
    if not clean or _is_virtual_path(clean):
        return False
    return (
        clean in {".", "..", "/"}
        or clean.startswith(("/", "./", "../"))
        or "/" in clean
        or "." in clean
    )


def _is_virtual_path(path: str) -> bool:
    parts = [part for part in path.split("/") if part]
    return bool(path.startswith("/") and parts and parts[0] in {"dev", "proc", "sys"})


def _string_set(value: object) -> set[str]:
    if isinstance(value, str):
        return {value}
    if isinstance(value, Sequence):
        return {item for item in value if isinstance(item, str)}
    return set()


def _string_mapping(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): raw
        for key, raw in value.items()
        if isinstance(key, str) and isinstance(raw, str)
    }
