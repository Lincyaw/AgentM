"""§11.4.12 declared-handler-effects ordering gate.

Locks the load-time rule that turns silent handler-ordering bugs (a reader
of ``api.tools`` registered before the atom that filters it) into a
fail-fast error. Part of the load-bearing "§11 extension contract
validator" position.
"""

from __future__ import annotations

from agentm.extensions import ChannelEffects, ExtensionManifest
from agentm.extensions.validate import validate_effects_ordering


def _manifest(name: str, effects: dict[str, ChannelEffects]) -> ExtensionManifest:
    return ExtensionManifest(name=name, description=name, effects=effects)


def test_effects_ordering_rule() -> None:
    mutator = _manifest(
        "tool_filter", {"before_agent_start": ChannelEffects(mutates=("tools",))}
    )
    reader = _manifest(
        "tool_index",
        {"before_agent_start": ChannelEffects(reads=("tools",), appends=("system",))},
    )
    undeclared = ExtensionManifest(name="plain", description="plain")

    # Reader before mutator on the same channel -> error naming both atoms,
    # the channel, the resource, and the fix.
    issues = validate_effects_ordering([("m.reader", reader), ("m.mutator", mutator)])
    errors = [i for i in issues if i.severity == "error"]
    assert len(errors) == 1
    assert errors[0].rule == "11.4.12-effects-read-after-mutate"
    for fragment in ("tool_index", "tool_filter", "before_agent_start", "tools", "move"):
        assert fragment in errors[0].message

    # Mutator before reader -> clean.
    assert validate_effects_ordering([("m.mutator", mutator), ("m.reader", reader)]) == []

    # Undeclared extensions are unconstrained regardless of position.
    assert (
        validate_effects_ordering(
            [("m.plain", undeclared), ("m.mutator", mutator), ("m.reader", reader)]
        )
        == []
    )

    # Two mutators of the same resource on the same channel -> warning only.
    other_mutator = _manifest(
        "tool_gate", {"before_agent_start": ChannelEffects(mutates=("tools",))}
    )
    issues = validate_effects_ordering(
        [("m.mutator", mutator), ("m.other", other_mutator)]
    )
    assert [i.severity for i in issues] == ["warning"]
    assert issues[0].rule == "11.4.12-effects-mutate-mutate"

    # Appenders are commutative: no constraint among appenders, and an
    # appender is not a mutator for the read-after-mutate rule.
    appender_a = _manifest(
        "sys_a", {"before_agent_start": ChannelEffects(appends=("system",))}
    )
    appender_b = _manifest(
        "sys_b", {"before_agent_start": ChannelEffects(appends=("system",))}
    )
    sys_reader = _manifest(
        "sys_reader", {"before_agent_start": ChannelEffects(reads=("system",))}
    )
    assert (
        validate_effects_ordering(
            [("m.sr", sys_reader), ("m.sa", appender_a), ("m.sb", appender_b)]
        )
        == []
    )
