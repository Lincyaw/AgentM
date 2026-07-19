"""Policy engine effects — diagnostic construction and injection."""

from __future__ import annotations

import re

from .types import EffectSpec, EvidenceList, ToolArgs


# ---------------------------------------------------------------------------
# Reason template interpolation
# ---------------------------------------------------------------------------


def interpolate_reason(template: str, event_args: ToolArgs) -> str:
    """Interpolate a reason template with event args.

    Supports both simple {key} and nested {event.args[key]} patterns.
    """
    def _resolve_nested(match: re.Match[str]) -> str:
        expr = match.group(1)
        if "event.args" in expr:
            key_match = re.search(r"\[(['\"]?)(\w+)\1\]", expr)
            if key_match:
                key = key_match.group(2)
                return str(event_args.get(key, f"{{{expr}}}"))
        return match.group(0)

    resolved = re.sub(r"\{(event\.[^}]+)\}", _resolve_nested, template)
    try:
        return resolved.format_map(_SafeDict(event_args))
    except (KeyError, IndexError, ValueError, AttributeError):
        return resolved


class _SafeDict(dict):  # type: ignore[type-arg]
    """Dict that returns '{key}' for missing keys during format_map."""

    def __missing__(self, key: str) -> str:
        return f"{{{key}}}"


# ---------------------------------------------------------------------------
# Escalation context serialization
# ---------------------------------------------------------------------------


def format_escalate_context(value: object) -> str:
    """Render an escalate_context return value to human-readable text."""
    if isinstance(value, EvidenceList):
        return value.format_for_diagnostic()
    if isinstance(value, dict):
        return "\n".join(f"  {k}: {v}" for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return "\n".join(f"  - {item}" for item in value)
    if value is None or not value:
        return "(no context available)"
    return str(value)


# ---------------------------------------------------------------------------
# Diagnostic message construction
# ---------------------------------------------------------------------------


def build_diagnostic(
    rule_id: str,
    effect: EffectSpec,
    event_args: ToolArgs,
    escalate_context_value: object = None,
) -> str:
    """Build the complete diagnostic message for injection."""
    reason = interpolate_reason(effect.reason_template, event_args)

    severity_prefix = {
        "notify": "info",
        "escalate": "warning",
        "block": "error",
        "abort": "critical",
    }.get(effect.effect, "info")

    parts = [f"[Policy:{rule_id}] ({severity_prefix}) {reason}"]

    if escalate_context_value is not None:
        ctx_text = format_escalate_context(escalate_context_value)
        if ctx_text and ctx_text != "(no context available)":
            parts.append(f"\nEvidence:\n{ctx_text}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Effect resolution (priority ordering)
# ---------------------------------------------------------------------------

_EFFECT_PRIORITY = {"abort": 0, "block": 1, "escalate": 2, "notify": 3}


def resolve_effects(
    fired: list[tuple[str, EffectSpec, str]],
) -> list[tuple[str, EffectSpec, str]]:
    """Sort fired effects by priority. abort > block > escalate > notify."""
    return sorted(fired, key=lambda x: _EFFECT_PRIORITY.get(x[1].effect, 99))
