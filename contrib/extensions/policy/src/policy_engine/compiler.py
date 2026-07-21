# code-health: ignore-file[AM025] -- policy engine normalizes untyped YAML, SQLite, and runtime event payloads
"""Policy engine compiler — YAML parse + expression compilation."""

from __future__ import annotations

import ast
import types as pytypes

import yaml
from loguru import logger

from .types import EffectSpec, Guard, RuleInstance


# ---------------------------------------------------------------------------
# AST allowlist
# ---------------------------------------------------------------------------

ALLOWED_NODES: frozenset[type] = frozenset(
    {
        ast.Expression,
        ast.BoolOp,
        ast.BinOp,
        ast.UnaryOp,
        ast.Compare,
        ast.IfExp,
        ast.Call,
        ast.Attribute,
        ast.Subscript,
        ast.Starred,
        ast.Dict,
        ast.List,
        ast.Tuple,
        ast.Set,
        ast.GeneratorExp,
        ast.ListComp,
        ast.comprehension,
        ast.Name,
        ast.Constant,
        ast.JoinedStr,
        ast.FormattedValue,
        ast.And,
        ast.Or,
        ast.Not,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Mod,
        ast.FloorDiv,
        ast.Pow,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.Is,
        ast.IsNot,
        ast.In,
        ast.NotIn,
        ast.UAdd,
        ast.USub,
        ast.Load,
        ast.Store,
        ast.Slice,
        ast.keyword,
    }
)

FORBIDDEN_NAMES: frozenset[str] = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "__import__",
        "open",
        "getattr",
        "setattr",
        "delattr",
        "type",
        "vars",
        "dir",
        "globals",
        "locals",
        "breakpoint",
        "exit",
        "quit",
        "input",
        "print",
    }
)

NAMESPACE_NAMES: frozenset[str] = frozenset(
    {
        "event",
        "session",
        "labels",
        "tool_log",
        "file_state",
        "effect_log",
        "ifg",
        "streak",
        "trend",
        "ratio",
        "sequence",
        "group",
        "diff",
        "lookup",
        "entity_evidence",
        "fingerprint",
        "hash",
        "max",
        "min",
        "abs",
        "len",
        "any",
        "all",
        "sorted",
        "True",
        "False",
        "None",
        "EMPTY",
        "not",
        "and",
        "or",
        "in",
    }
)


# ---------------------------------------------------------------------------
# AST validation
# ---------------------------------------------------------------------------


class CompileError(Exception):
    """Raised when a rule expression fails to compile."""


def _validate_ast(tree: ast.AST, rule_id: str) -> None:
    """Walk AST and reject disallowed node types."""
    for node in ast.walk(tree):
        if type(node) not in ALLOWED_NODES:
            raise CompileError(
                f"rule '{rule_id}': disallowed AST node {type(node).__name__}"
            )
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise CompileError(
                f"rule '{rule_id}': dunder attribute access '{node.attr}' forbidden"
            )
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise CompileError(f"rule '{rule_id}': forbidden name '{node.id}'")


# ---------------------------------------------------------------------------
# Expression compilation
# ---------------------------------------------------------------------------


def compile_expression(expr_str: str, rule_id: str) -> pytypes.CodeType:
    """Compile a restricted Python expression to a code object.

    Returns a code object suitable for eval() with an injected namespace.
    Raises CompileError on syntax errors or disallowed constructs.
    """
    expr_str = " ".join(line.strip() for line in expr_str.splitlines() if line.strip())
    if not expr_str:
        raise CompileError(f"rule '{rule_id}': empty expression")

    try:
        tree = ast.parse(expr_str, mode="eval")
    except SyntaxError as e:
        raise CompileError(f"rule '{rule_id}': syntax error: {e}") from e

    _validate_ast(tree, rule_id)

    try:
        code = compile(tree, filename=f"<rule:{rule_id}>", mode="eval")
    except Exception as e:
        raise CompileError(f"rule '{rule_id}': compile failed: {e}") from e

    return code


# ---------------------------------------------------------------------------
# Guard compilation (from `match` clause)
# ---------------------------------------------------------------------------


def compile_guard(match_clause: dict[str, object] | None) -> Guard:
    """Compile a match clause into a Guard for fast-reject."""
    if not match_clause:
        return Guard()

    tool_names: frozenset[str] | None = None
    field_patterns: list[tuple[str, str]] = []

    for key, pattern in match_clause.items():
        if key == "tool":
            if "|" in str(pattern):
                tool_names = frozenset(str(pattern).split("|"))
            else:
                tool_names = frozenset({str(pattern)})
        else:
            field_patterns.append((key, str(pattern)))

    return Guard(
        tool_names=tool_names,
        field_patterns=tuple(field_patterns),
    )


# ---------------------------------------------------------------------------
# Rule compilation (YAML dict → RuleInstance)
# ---------------------------------------------------------------------------


def compile_rule(rule_dict: dict[str, object]) -> RuleInstance:
    """Compile a single rule definition into a RuleInstance."""
    rule_id = str(rule_dict.get("name", "unnamed"))

    channel_map = {
        "tool_call_pre": "tool_call",
        "tool_call_post": "tool_result",
        "turn_end": "turn_committed",
        "session_spawn": "child_session_start",
        "context": "context",
    }
    # YAML parses bare `on:` as boolean True (reserved word).
    on_value = rule_dict.get("on") or rule_dict.get(True, "tool_call_pre")  # type: ignore[call-overload]
    channel = channel_map.get(str(on_value), str(on_value))

    match_raw = rule_dict.get("match")
    guard = compile_guard(match_raw if isinstance(match_raw, dict) else None)

    predicate = None
    when_expr = rule_dict.get("when")
    if when_expr:
        predicate = compile_expression(str(when_expr), rule_id)

    effect_type = str(rule_dict.get("effect", "notify"))
    reason = str(rule_dict.get("reason", ""))
    raw_cooldown = rule_dict.get("cooldown", 0)
    if isinstance(raw_cooldown, str) and "turns" in raw_cooldown:
        cooldown = int(raw_cooldown.split()[0])
    else:
        cooldown = int(str(raw_cooldown))

    escalate_ctx = None
    if rule_dict.get("escalate_context"):
        escalate_ctx = compile_expression(
            str(rule_dict["escalate_context"]), f"{rule_id}:ctx"
        )

    mode = str(rule_dict.get("mode", "observe"))
    if effect_type == "abort":
        mode = "enforce"

    layer = _infer_layer(rule_dict)

    return RuleInstance(
        rule_id=rule_id,
        layer=layer,
        mode=mode,  # type: ignore[arg-type]
        channel=channel,
        guard=guard,
        predicate=predicate,
        effect=EffectSpec(
            effect=effect_type,  # type: ignore[arg-type]
            reason_template=reason,
        ),
        escalate_context_expr=escalate_ctx,
        cooldown_turns=int(cooldown),
    )


def _infer_layer(rule_dict: dict[str, object]) -> int:
    """Infer the rule layer from its structure."""
    when_str = str(rule_dict.get("when", ""))
    if "effect_log" in when_str:
        return 3  # Meta
    if "labels." in when_str or "taint" in when_str:
        return 2  # IFC
    return 1  # Query


# ---------------------------------------------------------------------------
# Policy file compilation
# ---------------------------------------------------------------------------


def compile_policy_file(content: str) -> tuple[list[RuleInstance], list[str]]:
    """Parse a YAML policy file and compile all rules.

    Returns (compiled_rules, disabled_rule_names).
    """
    try:
        data = yaml.safe_load(content)
    except yaml.YAMLError as e:
        logger.error("policy YAML parse error: {}", e)
        return [], []

    if not isinstance(data, dict):
        logger.error("policy file must be a YAML mapping")
        return [], []

    disabled: list[str] = data.get("disable", [])
    if not isinstance(disabled, list):
        disabled = []

    rules_data = data.get("rules", [])
    if not isinstance(rules_data, list):
        logger.error("'rules' must be a list")
        return [], disabled

    compiled: list[RuleInstance] = []
    for rule_dict in rules_data:
        if not isinstance(rule_dict, dict):
            continue
        try:
            instance = compile_rule(rule_dict)
            compiled.append(instance)
        except CompileError as e:
            logger.warning("skipping rule: {}", e)

    return compiled, disabled


def compose_rules(
    layers: list[tuple[list[RuleInstance], list[str]]],
) -> list[RuleInstance]:
    """Compose multiple policy layers with override semantics.

    Later layers can add rules (tighten) and disable rules they authored
    or rules marked overridable. Platform-level safety rules (abort effect)
    cannot be disabled by child layers.
    """
    all_rules: dict[str, RuleInstance] = {}
    disabled: set[str] = set()

    for rules, disable_list in layers:
        for rule in rules:
            all_rules[rule.rule_id] = rule
        for name in disable_list:
            existing = all_rules.get(name)
            if existing and existing.effect.effect == "abort":
                logger.warning(
                    "cannot disable abort-effect rule '{}' — safety rules are not overridable",
                    name,
                )
                continue
            disabled.add(name)

    return [r for r in all_rules.values() if r.rule_id not in disabled]
