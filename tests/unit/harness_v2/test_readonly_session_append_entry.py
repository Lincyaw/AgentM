from __future__ import annotations

from agentm.core.kernel import text_message
from agentm.harness.session import _SessionView
from agentm.harness.session_manager import InMemorySessionManager, message_entry


def test_readonly_session_append_entry_returns_id_and_chains_parent_links() -> None:
    manager = InMemorySessionManager()
    root = message_entry(text_message("root", timestamp=0.0), parent_id=None)
    manager.append(root)
    view = _SessionView(manager)

    first_id = view.append_entry("compaction", {"step": 1})
    second_id = view.append_entry("compaction", {"step": 2}, parent_id=first_id)

    assert first_id != ""
    assert second_id != ""

    branch = manager.get_active_branch()
    assert [entry.type for entry in branch] == ["message", "compaction", "compaction"]
    assert branch[-2].id == first_id
    assert branch[-1].id == second_id
    assert branch[-1].parent_id == first_id
