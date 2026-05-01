from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentm.core.kernel import AgentMessage, EventBus, TextContent
from agentm.harness.extension import ProviderConfig, _ExtensionAPIImpl
from agentm_rca.trajectory_judger.reader import TrajectoryReader, install


class _SessionView:
    def __init__(self) -> None:
        self.entries: list[tuple[str, object]] = []

    def get_messages(self) -> list[AgentMessage]:
        return []

    def append_entry(self, type: str, payload: object, parent_id: str | None = None) -> str:
        del parent_id
        self.entries.append((type, payload))
        return str(len(self.entries))


def _make_api() -> _ExtensionAPIImpl:
    session = _SessionView()
    providers: dict[str, ProviderConfig] = {}
    return _ExtensionAPIImpl(
        bus=EventBus(),
        cwd=".",
        session=session,
        tools=[],
        commands={},
        providers=providers,
        renderers={},
        pending_user_messages=[],
        model_getter=lambda: None,
        provider_getter=lambda: None,
    )


def test_reader_register_and_read_skips_malformed_jsonl_lines(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.jsonl"
    path.write_text(
        '{"_meta": {"thread_id": "case-42"}}\n'
        'not-json\n'
        + json.dumps({"channel": "tool_call", "name": "read"})
        + "\n"
        + json.dumps({"channel": "tool_result", "name": "read"})
        + "\n",
        encoding="utf-8",
    )

    reader = TrajectoryReader()
    case_id = reader.register(path)

    assert case_id == "case-42"
    assert reader.read(case_id, ". | length", raw=True) == "3"
    payload = json.loads(reader.read(case_id, "."))
    assert [item.get("channel") for item in payload[1:]] == ["tool_call", "tool_result"]


@pytest.mark.asyncio
async def test_reader_tool_atom_reads_a_fixture_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.jsonl"
    path.write_text(
        json.dumps({"_meta": {"thread_id": "case-99"}}) + "\n"
        + json.dumps({"channel": "assistant", "text": "done"})
        + "\n",
        encoding="utf-8",
    )

    api = _make_api()
    install(api, {})
    tool = next(tool for tool in api.tools if tool.name == "jq_query")
    result = await tool.execute({"path": str(path), "expression": ".[1].channel", "raw": True})

    assert result.is_error is False
    assert isinstance(result.content[0], TextContent)
    assert result.content[0].text == "assistant"
