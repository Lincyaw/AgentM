"""``loop_budget`` atom — registers a LoopConfig + fail-fast validation.

Load-bearing because a scenario's turn/tool budget feeds reproducible eval
runs (e.g. rca pins ``max_turns``). A typo'd or non-positive value must fail at
install time, not be silently dropped — a silently-ignored budget would let
two "identical" eval runs diverge. The atom is the on-axiom way to express the
budget: a scenario lists it like any other capability rather than the
substrate special-casing a privileged manifest field.
"""

from __future__ import annotations

from typing import Any

import pytest

from agentm.core.abi import LoopConfig
from agentm.core.abi.roles import LOOP_BUDGET_SERVICE
from agentm.extensions.builtin import loop_budget


class _FakeAPI:
    """Minimal stand-in capturing ``set_service`` calls."""

    def __init__(self) -> None:
        self.services: dict[str, Any] = {}

    def set_service(self, name: str, obj: Any) -> None:
        self.services[name] = obj


def _install(config: dict[str, Any]) -> LoopConfig:
    api = _FakeAPI()
    loop_budget.install(api, config)  # type: ignore[arg-type]
    registered = api.services[LOOP_BUDGET_SERVICE]
    assert isinstance(registered, LoopConfig)
    return registered


def test_registers_both_caps() -> None:
    cfg = _install({"max_turns": 64, "max_tool_calls": 200})
    assert cfg.max_turns == 64
    assert cfg.max_tool_calls == 200


def test_empty_config_means_unbounded() -> None:
    cfg = _install({})
    assert cfg.max_turns is None
    assert cfg.max_tool_calls is None


def test_partial_config_leaves_other_uncapped() -> None:
    cfg = _install({"max_tool_calls": 50})
    assert cfg.max_turns is None
    assert cfg.max_tool_calls == 50


def test_explicit_null_is_uncapped() -> None:
    cfg = _install({"max_turns": None})
    assert cfg.max_turns is None


@pytest.mark.parametrize(
    "config",
    [
        {"max_turns": 0},       # not positive
        {"max_turns": -3},      # negative
        {"max_turns": True},    # bool masquerading as int
        {"max_tool_calls": 0},  # not positive
        {"max_turns": "8"},     # wrong type
    ],
)
def test_invalid_value_raises(config: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        _install(config)
