from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import types
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from agentm.core.abi import AssistantStreamEvent, MessageEnd, Model, TextContent
from agentm.core.abi.messages import AssistantMessage
from agentm.core.abi.extension import ProviderConfig
from agentm.core.abi.session_config import AgentSessionConfig
from agentm.core.runtime.session import AgentSession

_PROVIDER_MODULE = "agentm._tests.artifact_store_provider"
_HYPOTHESIS_MODULE = "agentm._tests.rca_hypothesis_tools"


def _install_provider_module() -> str:
    if _PROVIDER_MODULE in sys.modules:
        return _PROVIDER_MODULE
    module = types.ModuleType(_PROVIDER_MODULE)

    async def _stream(
        *,
        messages: list[Any],
        model: Model,
        tools: list[Any],
        system: str | None = None,
        signal: Any = None,
        thinking: str = "off",
    ) -> AsyncIterator[AssistantStreamEvent]:
        del messages, model, tools, system, signal, thinking
        yield MessageEnd(
            message=AssistantMessage(
                role="assistant",
                content=[TextContent(type="text", text="noop")],
                timestamp=0.0,
                stop_reason="end_turn",
            )
        )

    def install(api: Any, _config: dict[str, Any]) -> None:
        api.register_provider(
            "artifact-test",
            ProviderConfig(
                stream_fn=_stream,
                model=Model(
                    id="artifact-test",
                    provider="fake",
                    context_window=16_000,
                    max_output_tokens=2_000,
                ),
                name="artifact-test",
            ),
        )

    module.install = install  # type: ignore[attr-defined]
    sys.modules[_PROVIDER_MODULE] = module
    return _PROVIDER_MODULE


def _install_hypothesis_module() -> str:
    if _HYPOTHESIS_MODULE in sys.modules:
        return _HYPOTHESIS_MODULE
    repo_root = Path(__file__).resolve().parents[2]
    file_path = repo_root / "contrib" / "scenarios" / "rca" / "src" / "agentm_rca" / "tools" / "hypothesis_tools.py"
    spec = importlib.util.spec_from_file_location(_HYPOTHESIS_MODULE, file_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[_HYPOTHESIS_MODULE] = module
    spec.loader.exec_module(module)
    return _HYPOTHESIS_MODULE


async def _create_session(
    tmp_path: Path,
    *,
    root_session_id: str | None = None,
    task_id: str | None = None,
    extensions: list[tuple[str, dict[str, Any]]] | None = None,
) -> AgentSession:
    provider_module = _install_provider_module()
    return await AgentSession.create(
        AgentSessionConfig(
            cwd=str(tmp_path),
            provider=(provider_module, {}),
            extensions=list(extensions or []),
            root_session_id=root_session_id,
            task_id=task_id,
        )
    )


def _tool(session: AgentSession, name: str) -> Any:
    for tool in session.tools:
        if tool.name == name:
            return tool
    raise AssertionError(f"missing tool {name}")


async def _tool_json(tool: Any, args: dict[str, Any]) -> dict[str, Any]:
    result = await tool.execute(args)
    assert result.is_error is False, result.content[0].text
    return json.loads(result.content[0].text)


@pytest.mark.asyncio
async def test_artifact_write_is_append_only_for_same_title(tmp_path: Path) -> None:
    session = await _create_session(
        tmp_path,
        extensions=[

            ("agentm.extensions.builtin.operations_local", {}),

            ("agentm.extensions.builtin.artifact_store", {})],
    )
    try:
        write = _tool(session, "artifact_write")
        first = await _tool_json(
            write,
            {"kind": "finding", "title": "Repeated Title", "body": "first body"},
        )
        second = await _tool_json(
            write,
            {"kind": "finding", "title": "Repeated Title", "body": "second body"},
        )

        assert first["artifact_id"] != second["artifact_id"]
        first_path = Path(first["path"])
        second_path = Path(second["path"])
        assert first_path.read_text(encoding="utf-8") == "first body"
        assert second_path.read_text(encoding="utf-8") == "second body"
        assert first_path != second_path
    finally:
        await session.shutdown()


@pytest.mark.asyncio
async def test_artifact_store_shares_by_root_session_and_isolates_other_roots(
    tmp_path: Path,
) -> None:
    shared_a = await _create_session(
        tmp_path,
        root_session_id="root-shared",
        task_id="task-a",
        extensions=[

            ("agentm.extensions.builtin.operations_local", {}),

            ("agentm.extensions.builtin.artifact_store", {})],
    )
    shared_b = await _create_session(
        tmp_path,
        root_session_id="root-shared",
        task_id="task-b",
        extensions=[

            ("agentm.extensions.builtin.operations_local", {}),

            ("agentm.extensions.builtin.artifact_store", {})],
    )
    isolated = await _create_session(
        tmp_path,
        root_session_id="root-isolated",
        task_id="task-c",
        extensions=[

            ("agentm.extensions.builtin.operations_local", {}),

            ("agentm.extensions.builtin.artifact_store", {})],
    )
    try:
        shared_a_write = _tool(shared_a, "artifact_write")
        shared_b_write = _tool(shared_b, "artifact_write")
        isolated_list = _tool(isolated, "artifact_list")
        isolated_read = _tool(isolated, "artifact_read")
        shared_a_list = _tool(shared_a, "artifact_list")

        first = await _tool_json(
            shared_a_write,
            {"kind": "finding", "title": "Shared A", "body": "alpha"},
        )
        second = await _tool_json(
            shared_b_write,
            {"kind": "finding", "title": "Shared B", "body": "beta"},
        )

        shared_listing = await _tool_json(shared_a_list, {})
        assert [item["id"] for item in shared_listing["artifacts"]] == [
            second["artifact_id"],
            first["artifact_id"],
        ]

        isolated_listing = await _tool_json(isolated_list, {})
        assert isolated_listing["artifacts"] == []

        missing = await isolated_read.execute({"artifact_id": first["artifact_id"]})
        assert missing.is_error is True
        assert "unknown artifact_id" in missing.content[0].text
    finally:
        await shared_a.shutdown()
        await shared_b.shutdown()
        await isolated.shutdown()


@pytest.mark.asyncio
async def test_artifact_store_concurrent_same_root_allocates_unique_ids(
    tmp_path: Path,
) -> None:
    first = await _create_session(
        tmp_path,
        root_session_id="root-concurrent",
        task_id="task-a",
        extensions=[

            ("agentm.extensions.builtin.operations_local", {}),

            ("agentm.extensions.builtin.artifact_store", {})],
    )
    second = await _create_session(
        tmp_path,
        root_session_id="root-concurrent",
        task_id="task-b",
        extensions=[

            ("agentm.extensions.builtin.operations_local", {}),

            ("agentm.extensions.builtin.artifact_store", {})],
    )
    try:
        first_write = _tool(first, "artifact_write")
        second_write = _tool(second, "artifact_write")
        writes = [
            _tool_json(
                first_write if index % 2 == 0 else second_write,
                {"kind": "finding", "title": f"Concurrent {index}", "body": str(index)},
            )
            for index in range(12)
        ]

        payloads = await asyncio.gather(*writes)

        ids = [payload["artifact_id"] for payload in payloads]
        assert sorted(ids) == [f"art_{index:03d}" for index in range(1, 13)]
        assert len(set(ids)) == len(ids)
    finally:
        await first.shutdown()
        await second.shutdown()


@pytest.mark.asyncio
async def test_artifact_store_service_is_per_session_instance(tmp_path: Path) -> None:
    first = await _create_session(
        tmp_path,
        root_session_id="root-service-a",
        extensions=[

            ("agentm.extensions.builtin.operations_local", {}),

            ("agentm.extensions.builtin.artifact_store", {})],
    )
    second = await _create_session(
        tmp_path,
        root_session_id="root-service-b",
        extensions=[

            ("agentm.extensions.builtin.operations_local", {}),

            ("agentm.extensions.builtin.artifact_store", {})],
    )
    try:
        first_write = _tool(first, "artifact_write")
        second_write = _tool(second, "artifact_write")
        second_list = _tool(second, "artifact_list")
        first_list = _tool(first, "artifact_list")
        written = await _tool_json(
            first_write,
            {"kind": "finding", "title": "First Only", "body": "alpha"},
        )

        second_written = await _tool_json(
            second_write,
            {"kind": "finding", "title": "Second Only", "body": "beta"},
        )

        assert written["artifact_id"] == "art_001"
        assert second_written["artifact_id"] == "art_001"
        assert [item["id"] for item in (await _tool_json(first_list, {}))["artifacts"]] == [
            written["artifact_id"]
        ]
        assert [item["id"] for item in (await _tool_json(second_list, {}))["artifacts"]] == [
            second_written["artifact_id"]
        ]
    finally:
        await first.shutdown()
        await second.shutdown()


@pytest.mark.asyncio
async def test_artifact_read_truncates_large_body_without_range(tmp_path: Path) -> None:
    session = await _create_session(
        tmp_path,
        extensions=[

            ("agentm.extensions.builtin.operations_local", {}),

            ("agentm.extensions.builtin.artifact_store", {})],
    )
    try:
        write = _tool(session, "artifact_write")
        read = _tool(session, "artifact_read")
        body = "x" * (8 * 1024 + 256)
        written = await _tool_json(
            write,
            {"kind": "query_result", "title": "Large Body", "body": body},
        )

        payload = await _tool_json(read, {"artifact_id": written["artifact_id"]})
        assert payload["body"].startswith("x" * (8 * 1024))
        assert "[Truncated:" in payload["body"]
        assert body not in payload["body"]
    finally:
        await session.shutdown()


@pytest.mark.asyncio
async def test_artifact_list_filters_by_kind_and_created_by_task(tmp_path: Path) -> None:
    first = await _create_session(
        tmp_path,
        root_session_id="root-filter",
        task_id="task-1",
        extensions=[

            ("agentm.extensions.builtin.operations_local", {}),

            ("agentm.extensions.builtin.artifact_store", {})],
    )
    second = await _create_session(
        tmp_path,
        root_session_id="root-filter",
        task_id="task-2",
        extensions=[

            ("agentm.extensions.builtin.operations_local", {}),

            ("agentm.extensions.builtin.artifact_store", {})],
    )
    try:
        first_write = _tool(first, "artifact_write")
        second_write = _tool(second, "artifact_write")
        list_tool = _tool(first, "artifact_list")

        await _tool_json(
            first_write,
            {"kind": "finding", "title": "First", "body": "one", "tags": ["shared"]},
        )
        target = await _tool_json(
            second_write,
            {"kind": "hypothesis", "title": "Second", "body": "two", "tags": ["shared", "focus"]},
        )
        await _tool_json(
            second_write,
            {"kind": "finding", "title": "Third", "body": "three", "tags": ["other"]},
        )

        filtered = await _tool_json(
            list_tool,
            {"kind": "hypothesis", "created_by_task": "task-2"},
        )
        assert [item["id"] for item in filtered["artifacts"]] == [target["artifact_id"]]
        assert filtered["artifacts"][0]["kind"] == "hypothesis"
        assert filtered["artifacts"][0]["created_by"]["task_id"] == "task-2"
    finally:
        await first.shutdown()
        await second.shutdown()


@pytest.mark.asyncio
async def test_hypothesis_tools_persist_append_only_artifacts(tmp_path: Path) -> None:
    hypothesis_module = _install_hypothesis_module()
    session = await _create_session(
        tmp_path,
        root_session_id="root-hypothesis",
        task_id="task-hypothesis",
        extensions=[

            ("agentm.extensions.builtin.operations_local", {}),
            ("agentm.extensions.builtin.artifact_store", {}),
            (hypothesis_module, {}),
        ],
    )
    try:
        update = _tool(session, "update_hypothesis")
        remove = _tool(session, "remove_hypothesis")
        list_tool = _tool(session, "artifact_list")

        first = await _tool_json(
            update,
            {
                "id": "hyp-1",
                "description": "Initial hypothesis",
                "status": "formed",
            },
        )
        second = await _tool_json(
            update,
            {
                "id": "hyp-1",
                "description": "Refined hypothesis",
                "status": "refined",
            },
        )
        removed = await _tool_json(remove, {"id": "hyp-1"})
        listing = await _tool_json(list_tool, {"kind": "hypothesis", "limit": 10})

        assert first["artifact_id"] != second["artifact_id"]
        assert removed["artifact_id"] != second["artifact_id"]
        assert len(listing["artifacts"]) == 3
        assert all(item["kind"] == "hypothesis" for item in listing["artifacts"])
    finally:
        await session.shutdown()
