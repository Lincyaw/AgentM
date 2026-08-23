# code-health: ignore-file[AM025] -- legacy wire data is validated at the trust boundary
# code-health: ignore-file[AM022] -- legacy persisted JSON is heterogeneous by definition
"""Raw version-2 trajectory record migrations.

Version 2 stored multiple provider/tool rounds inside one Turn. Version 3
stores one provider response per Turn. The migration preserves replay order by
flattening legacy rounds into injected messages when a lossless structural
mapping is not possible.
"""

from __future__ import annotations

from typing import Any

from agentm.core.lib.codec_primitives import (
    expect_array,
    expect_integer,
    expect_object,
    expect_only_fields,
    expect_string,
)


def _legacy_injected_messages(
    data: Any,
    *,
    round_count: int,
    path: str,
) -> dict[int, list[dict[str, Any]]]:
    by_round: dict[int, list[dict[str, Any]]] = {}
    for index, raw_entry in enumerate(expect_array(data, path)):
        entry_path = f"{path}[{index}]"
        entry = expect_object(raw_entry, entry_path)
        expect_only_fields(entry, {"after_round", "messages"}, entry_path)
        after_round = expect_integer(
            entry.get("after_round"),
            f"{entry_path}.after_round",
            minimum=-1,
        )
        if after_round >= round_count:
            raise ValueError(
                f"{entry_path}.after_round must reference a materialized round"
            )
        messages = [
            expect_object(message, f"{entry_path}.messages[{message_index}]")
            for message_index, message in enumerate(
                expect_array(entry.get("messages"), f"{entry_path}.messages")
            )
        ]
        by_round.setdefault(after_round, []).extend(messages)
    return by_round


def _legacy_round_messages(
    rounds_data: Any,
    injected_data: Any,
    *,
    path: str,
    injected_path: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rounds = [
        expect_object(raw_round, f"{path}.rounds[{index}]")
        for index, raw_round in enumerate(expect_array(rounds_data, f"{path}.rounds"))
    ]
    injected = _legacy_injected_messages(
        injected_data,
        round_count=len(rounds),
        path=injected_path,
    )
    messages = list(injected.get(-1, ()))
    for round_index, round_data in enumerate(rounds):
        round_path = f"{path}.rounds[{round_index}]"
        expect_only_fields(round_data, {"response", "tool_results"}, round_path)
        messages.append(
            expect_object(round_data.get("response"), f"{round_path}.response")
        )
        raw_results = expect_array(
            round_data.get("tool_results"),
            f"{round_path}.tool_results",
        )
        if raw_results:
            result_blocks: list[dict[str, Any]] = []
            for result_index, raw_record in enumerate(raw_results):
                record_path = f"{round_path}.tool_results[{result_index}]"
                record = expect_object(raw_record, record_path)
                expect_only_fields(
                    record,
                    {"call", "result", "backgrounded"},
                    record_path,
                )
                result_blocks.append(
                    expect_object(record.get("result"), f"{record_path}.result")
                )
            messages.append(
                {
                    "role": "tool_result",
                    "content": result_blocks,
                    "timestamp": 0.0,
                }
            )
        messages.extend(injected.get(round_index, ()))
    return rounds, messages


def migrate_v2_turn(
    data: dict[str, Any],
    *,
    target_version: int,
) -> dict[str, Any]:
    """Convert one version-2 Turn record into the current raw shape."""

    expect_only_fields(
        data,
        {
            "schema_version",
            "index",
            "id",
            "trigger",
            "rounds",
            "outcome",
            "timestamp",
            "meta",
            "trigger_metadata",
        },
        "legacy turn",
    )
    raw_outcome = expect_object(data.get("outcome"), "legacy turn.outcome")
    expect_only_fields(raw_outcome, {"cause", "injected"}, "legacy turn.outcome")
    rounds, replay_messages = _legacy_round_messages(
        data.get("rounds"),
        raw_outcome.get("injected", []),
        path="legacy turn",
        injected_path="legacy turn.outcome.injected",
    )
    turn_id = expect_string(data.get("id"), "legacy turn.id", allow_empty=False)

    response: object = None
    tool_results: object = []
    injected: object = replay_messages
    if len(rounds) == 1:
        legacy_injected = _legacy_injected_messages(
            raw_outcome.get("injected", []),
            round_count=1,
            path="legacy turn.outcome.injected",
        )
        if not legacy_injected.get(-1):
            response = rounds[0].get("response")
            tool_results = rounds[0].get("tool_results")
            injected = legacy_injected.get(0, [])

    return {
        "schema_version": target_version,
        "index": data.get("index"),
        "id": turn_id,
        "run_id": f"legacy-v2:{turn_id}",
        "run_step": 0,
        "trigger": data.get("trigger"),
        "response": response,
        "tool_results": tool_results,
        "outcome": {
            "cause": raw_outcome.get("cause"),
            "injected": injected,
        },
        "timestamp": data.get("timestamp"),
        "meta": data.get("meta"),
        "trigger_metadata": data.get("trigger_metadata"),
    }


def migrate_v2_checkpoint(
    data: dict[str, Any],
    *,
    target_version: int,
) -> dict[str, Any]:
    """Convert one version-2 TurnCheckpoint record into the current raw shape."""

    expect_only_fields(
        data,
        {
            "schema_version",
            "index",
            "id",
            "trigger",
            "rounds",
            "updated_at",
            "meta",
            "injected",
            "trigger_metadata",
        },
        "legacy turn checkpoint",
    )
    _, replay_messages = _legacy_round_messages(
        data.get("rounds"),
        data.get("injected", []),
        path="legacy turn checkpoint",
        injected_path="legacy turn checkpoint.injected",
    )
    checkpoint_id = expect_string(
        data.get("id"),
        "legacy turn checkpoint.id",
        allow_empty=False,
    )
    return {
        "schema_version": target_version,
        "index": data.get("index"),
        "id": checkpoint_id,
        "run_id": f"legacy-v2:{checkpoint_id}",
        "run_step": 0,
        "trigger": data.get("trigger"),
        "response": None,
        "tool_results": [],
        "updated_at": data.get("updated_at"),
        "meta": data.get("meta"),
        "injected": replay_messages,
        "trigger_metadata": data.get("trigger_metadata"),
    }


__all__ = ["migrate_v2_checkpoint", "migrate_v2_turn"]
