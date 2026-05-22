"""Fail-stop for ``ChildSessionExtendingEvent`` — locks down the
substrate's "extensions can contribute atoms to child sessions" contract.

The dedupe helper (``apply_child_session_contributions``) is the only
substrate-side code path that touches handler return values; if its
ordering or dedupe drifts, every atom that auto-attaches to children
(``live_inspector`` today, more later) silently regresses. We test the
helper directly rather than standing up a real session — handler return
shapes are the contract surface.

Coverage:
- ``None`` returns are ignored.
- A single handler's contribution is appended.
- Two handlers' contributions are appended in registration order.
- Re-contributing a ``(module, _)`` already on the base extensions list is
  dropped (operator-override wins).
- Re-contributing a module that an earlier handler already contributed is
  dropped (first-contribution wins).
- Malformed entries (non-tuple, wrong arity, non-string module, non-dict
  config) are silently skipped — we don't want a misbehaving atom to wedge
  spawn.
"""

from __future__ import annotations

from agentm.core.runtime.session_factory import apply_child_session_contributions


def test_no_handler_returns_leaves_base_unchanged() -> None:
    base = [("m.a", {"x": 1})]
    out = apply_child_session_contributions(base, [])
    assert out == base
    # Result is a fresh list — mutating it must not affect the caller's
    # original (so the substrate can hand it to the factory without
    # worrying about post-spawn mutation aliasing back).
    out.append(("m.b", {}))
    assert base == [("m.a", {"x": 1})]


def test_none_returns_are_ignored() -> None:
    out = apply_child_session_contributions([], [None, None])
    assert out == []


def test_single_handler_contribution_appended() -> None:
    out = apply_child_session_contributions(
        [], [[("contrib.extensions.live_inspector", {"role": "child"})]]
    )
    assert out == [("contrib.extensions.live_inspector", {"role": "child"})]


def test_two_handlers_contributions_appended_in_order() -> None:
    out = apply_child_session_contributions(
        [],
        [
            [("m.first", {"k": 1})],
            [("m.second", {"k": 2})],
        ],
    )
    assert out == [
        ("m.first", {"k": 1}),
        ("m.second", {"k": 2}),
    ]


def test_module_already_on_base_extensions_wins_against_contribution() -> None:
    """Operator override on the child config beats any handler contribution
    for the same module — handlers can declare intent, but the operator's
    explicit listing is authoritative.
    """
    base = [("m.x", {"operator_chosen_port": 9999})]
    out = apply_child_session_contributions(
        base, [[("m.x", {"port": 0})]]
    )
    assert out == [("m.x", {"operator_chosen_port": 9999})]


def test_first_contribution_wins_across_handlers() -> None:
    """If two handlers contribute the same module, the first one wins."""
    out = apply_child_session_contributions(
        [],
        [
            [("m.shared", {"from": "first"})],
            [("m.shared", {"from": "second"})],
        ],
    )
    assert out == [("m.shared", {"from": "first"})]


def test_malformed_entries_are_skipped() -> None:
    """A misbehaving handler must not wedge spawn."""
    out = apply_child_session_contributions(
        [],
        [
            "not iterable in a useful way at all",
            [
                "bare-string-not-a-tuple",
                ("only-one-element",),
                ("ok.module", {"role": "kept"}),
                (123, {}),  # non-string module
                ("bad.config", "not-a-dict"),
            ],
        ],
    )
    # Permissive: iterating "not iterable" produces single characters; each
    # one is dropped by the tuple-arity check. Strings are iterable, so the
    # outer ``list(ret)`` succeeds but every element fails downstream.
    assert out == [("ok.module", {"role": "kept"})]
