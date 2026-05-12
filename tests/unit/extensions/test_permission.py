"""Permission atom: glob matching + allow-overrides-deny precedence.

Fail-stop position: the atom is the load-bearing policy gate between
scenarios and tools. If glob matching silently misfires or the precedence
flips, a scenario that thought it carved an exception out of a broad
deny rule would actually be denied (or vice versa, a denied tool would
sneak through). Both are real-world security failures.

These tests drive ``permission.install`` against a stub ExtensionAPI,
emit synthetic ``ToolCallEvent``s, and assert the handler's verdict.
"""

from __future__ import annotations

from typing import Any

import pytest

from agentm.core.abi.events import ToolCallEvent
from agentm.extensions.builtin import permission


class _StubAPI:
    """Bare-minimum ExtensionAPI surface ``permission.install`` touches."""

    def __init__(self) -> None:
        self.handler: Any = None

    def on(self, channel: str, handler: Any) -> None:
        assert channel == ToolCallEvent.CHANNEL
        self.handler = handler


def _verdict(api: _StubAPI, name: str) -> dict[str, Any] | None:
    event = ToolCallEvent(tool_call_id="call-1", tool_name=name, args={})
    return api.handler(event)  # type: ignore[no-any-return]


def test_pure_allow_is_positive_list() -> None:
    """With only ``allow`` set: a name passes iff it matches the list."""

    api = _StubAPI()
    permission.install(api, {"allow": ["read", "edit"]})  # type: ignore[arg-type]

    assert _verdict(api, "read") is None
    not_listed = _verdict(api, "bash")
    assert not_listed is not None and "allowlist" in not_listed["reason"]


def test_pure_deny_is_negative_list() -> None:
    """With only ``deny`` set: anything not matching ``deny`` is permitted."""

    api = _StubAPI()
    permission.install(api, {"deny": ["dangerous_*"]})  # type: ignore[arg-type]

    blocked = _verdict(api, "dangerous_delete")
    assert blocked is not None and "denied" in blocked["reason"]
    assert _verdict(api, "read") is None
    assert _verdict(api, "mcp__anything__here") is None


def test_both_lists_treats_allow_as_carve_out() -> None:
    """The motivating case: deny mcp__* but let mcp__fetch__* through,
    and leave native AgentM tools untouched."""

    api = _StubAPI()
    permission.install(
        api,  # type: ignore[arg-type]
        {"deny": ["mcp__*"], "allow": ["mcp__fetch__*"]},
    )

    # Allowed by the carve-out even though deny pattern also matches.
    assert _verdict(api, "mcp__fetch__fetch") is None
    # Denied: matches deny, not allow.
    blocked = _verdict(api, "mcp__github__create_issue")
    assert blocked is not None and "denied" in blocked["reason"]
    # Native tools match neither list → permitted (NOT forced into a
    # positive-list regime when both lists are set).
    assert _verdict(api, "read") is None
    assert _verdict(api, "bash") is None
    assert _verdict(api, "tool_eval_run") is None


def test_empty_config_installs_no_handler() -> None:
    """Atom is a no-op when neither list is configured."""

    api = _StubAPI()
    permission.install(api, {})  # type: ignore[arg-type]
    assert api.handler is None


@pytest.mark.parametrize(
    "pattern,name,should_match",
    [
        ("mcp__*", "mcp__fetch__fetch", True),
        ("mcp__fetch__*", "mcp__github__issue", False),
        ("tool_?", "tool_x", True),
        ("tool_?", "tool_xx", False),
        ("file[0-9]", "file3", True),
        ("file[0-9]", "fileA", False),
    ],
)
def test_glob_metacharacters(pattern: str, name: str, should_match: bool) -> None:
    """fnmatch semantics — sanity check on `*`, `?`, `[seq]`."""

    api = _StubAPI()
    permission.install(api, {"deny": [pattern]})  # type: ignore[arg-type]
    verdict = _verdict(api, name)
    if should_match:
        assert verdict is not None and verdict["block"] is True
    else:
        assert verdict is None
