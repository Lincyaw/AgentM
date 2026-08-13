# code-health: ignore-file[AM025] -- ABI DTOs and codecs enforce runtime invariants at trust boundaries
"""Strict portable wire codec for resource mutations embedded in turns."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import cast

from agentm.core.abi.resource import (
    ResourceMutation,
    ResourceMutationOp,
    ResourceRef,
    ResourceTransactionRef,
)
from agentm.core.lib.codec_primitives import (
    expect_object,
    expect_only_fields,
    expect_optional_string,
    expect_string,
)


def serialize_resource_mutations(
    mutations: Sequence[ResourceMutation],
) -> list[dict[str, object]]:
    payload: list[dict[str, object]] = []
    for mutation in mutations:
        transaction = mutation.transaction
        transaction_payload: dict[str, object] | None = None
        if transaction is not None:
            transaction_payload = {
                "id": transaction.id,
                "session_id": transaction.session_id,
                "turn_id": transaction.turn_id,
                "turn_index": transaction.turn_index,
            }
        payload.append(
            {
                "ref": {
                    "namespace": mutation.ref.namespace,
                    "path": mutation.ref.path,
                },
                "op": mutation.op,
                "transaction": transaction_payload,
                "before_version": mutation.before_version,
                "after_version": mutation.after_version,
                "metadata": dict(mutation.metadata),
            }
        )
    return payload


def deserialize_resource_mutations(data: object) -> tuple[ResourceMutation, ...]:
    if not isinstance(data, list):
        raise ValueError("resource_mutations must be a list")
    mutations: list[ResourceMutation] = []
    for index, raw_item in enumerate(cast(Sequence[object], data)):
        label = f"resource_mutations[{index}]"
        item = expect_object(raw_item, label)
        expect_only_fields(
            item,
            {
                "ref",
                "op",
                "transaction",
                "before_version",
                "after_version",
                "metadata",
            },
            label,
        )
        ref_data = expect_object(item.get("ref"), f"{label}.ref")
        expect_only_fields(ref_data, {"namespace", "path"}, f"{label}.ref")
        op_value = item.get("op")
        if op_value not in {"create", "write", "replace", "delete"}:
            raise ValueError(f"{label}.op is invalid")
        transaction_data = item.get("transaction")
        transaction = (
            None
            if transaction_data is None
            else _transaction_ref(transaction_data, f"{label}.transaction")
        )
        mutations.append(
            ResourceMutation(
                ref=ResourceRef(
                    namespace=expect_string(
                        ref_data.get("namespace"),
                        f"{label}.ref.namespace",
                        allow_empty=False,
                    ),
                    path=expect_string(
                        ref_data.get("path"),
                        f"{label}.ref.path",
                        allow_empty=False,
                    ),
                ),
                op=cast(ResourceMutationOp, op_value),
                transaction=transaction,
                before_version=expect_optional_string(
                    item.get("before_version"),
                    f"{label}.before_version",
                    allow_empty=False,
                ),
                after_version=expect_optional_string(
                    item.get("after_version"),
                    f"{label}.after_version",
                    allow_empty=False,
                ),
                metadata=_metadata(item.get("metadata", {}), f"{label}.metadata"),
            )
        )
    return tuple(mutations)


def _transaction_ref(value: object, label: str) -> ResourceTransactionRef:
    data = expect_object(value, label)
    expect_only_fields(data, {"id", "session_id", "turn_id", "turn_index"}, label)
    turn_index = data.get("turn_index")
    if (
        not isinstance(turn_index, int)
        or isinstance(turn_index, bool)
        or turn_index < 0
    ):
        raise ValueError(f"{label}.turn_index must be a non-negative integer")
    return ResourceTransactionRef(
        id=expect_string(data.get("id"), f"{label}.id", allow_empty=False),
        session_id=expect_string(
            data.get("session_id"),
            f"{label}.session_id",
            allow_empty=False,
        ),
        turn_id=expect_string(
            data.get("turn_id"),
            f"{label}.turn_id",
            allow_empty=False,
        ),
        turn_index=turn_index,
    )


def _metadata(
    value: object,
    label: str,
) -> dict[str, str | int | float | bool | None]:
    data = expect_object(value, label)
    metadata: dict[str, str | int | float | bool | None] = {}
    for key, item in data.items():
        if item is not None and not isinstance(item, (str, int, float, bool)):
            raise ValueError(f"{label}[{key!r}] must be a JSON scalar")
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError(f"{label}[{key!r}] must be finite")
        metadata[key] = item
    return metadata


__all__ = ["deserialize_resource_mutations", "serialize_resource_mutations"]
