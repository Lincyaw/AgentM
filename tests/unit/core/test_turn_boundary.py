"""Turn-boundary marker: resume sheds a crash-left half-turn.

Fail-stop position (transactional reload atomicity): entries persist
incrementally mid-turn, so a process killed mid-turn leaves a half-turn on
disk. A clean ``agent_end`` appends a ``turn_committed`` marker; cold load
truncates back to the last marker so the rebuilt context never ends in a
dangling tool_call.
"""

from __future__ import annotations

from agentm.core.abi.messages import TextContent, UserMessage
from agentm.core.abi.session import ENTRY_TYPE_TURN_COMMITTED
from agentm.core.runtime.session_manager import SessionManager, _entry_to_record


def _user(text: str) -> UserMessage:
    return UserMessage(
        role="user", content=[TextContent(type="text", text=text)], timestamp=0.0
    )


def _commit(mgr: SessionManager) -> None:
    mgr.append_custom_entry(ENTRY_TYPE_TURN_COMMITTED, {"cause": "end_turn"})


def test_truncate_sheds_trailing_half_turn() -> None:
    mgr = SessionManager.in_memory()
    mgr.append_message(_user("turn 1"))
    _commit(mgr)  # clean boundary after turn 1
    marker_id = mgr.get_leaf_id()
    mgr.append_message(_user("turn 2 user"))  # crash mid-turn-2: no marker after
    assert mgr.get_leaf_entry() is not None
    assert mgr.get_leaf_entry().type == "message"

    mgr._truncate_to_last_boundary()

    assert mgr.get_leaf_id() == marker_id
    # The active branch now ends at the marker — the half-turn is off-path.
    types = [e.type for e in mgr.get_branch()]
    assert types[-1] == ENTRY_TYPE_TURN_COMMITTED
    assert types.count("message") == 1  # only turn-1's user message remains active


def test_truncate_noop_on_clean_ending() -> None:
    mgr = SessionManager.in_memory()
    mgr.append_message(_user("turn 1"))
    _commit(mgr)
    leaf_before = mgr.get_leaf_id()

    mgr._truncate_to_last_boundary()

    assert mgr.get_leaf_id() == leaf_before  # leaf already at the marker


def test_truncate_noop_without_markers_legacy_log() -> None:
    # Pre-feature logs carry no markers — resume must NOT shed anything.
    mgr = SessionManager.in_memory()
    mgr.append_message(_user("a"))
    mgr.append_message(_user("b"))
    leaf_before = mgr.get_leaf_id()

    mgr._truncate_to_last_boundary()

    assert mgr.get_leaf_id() == leaf_before


def test_from_records_applies_truncation() -> None:
    # Build a tree with a trailing half-turn, serialize, reload via from_records
    # (the ClickHouse resume path) and assert it sheds the half-turn.
    src = SessionManager.in_memory()
    src.append_message(_user("turn 1"))
    _commit(src)
    src.append_message(_user("turn 2 user"))  # half-turn, no trailing marker
    records = [_entry_to_record(src.get_entry(eid)) for eid in src._order]
    header = src.get_header()
    assert header is not None

    reloaded = SessionManager.from_records(header, records)

    leaf = reloaded.get_leaf_entry()
    assert leaf is not None
    assert leaf.type == ENTRY_TYPE_TURN_COMMITTED
