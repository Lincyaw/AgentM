"""Strict portable wire codec for resource mutations embedded in turns."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import cast

from agentm.core.abi.resource import (
    ResourceMutation,
    ResourceMutationOp,
    ResourceRef,
    ResourceTransactionRef,
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
        item = _object(raw_item, label)
        _only_fields(
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
        ref_data = _object(item.get("ref"), f"{label}.ref")
        _only_fields(ref_data, {"namespace", "path"}, f"{label}.ref")
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
                    namespace=_required_string(
                        ref_data.get("namespace"),
                        f"{label}.ref.namespace",
                    ),
                    path=_required_string(
                        ref_data.get("path"),
                        f"{label}.ref.path",
                    ),
                ),
                op=cast(ResourceMutationOp, op_value),
                transaction=transaction,
                before_version=_optional_string(
                    item.get("before_version"),
                    f"{label}.before_version",
                ),
                after_version=_optional_string(
                    item.get("after_version"),
                    f"{label}.after_version",
                ),
                metadata=_metadata(item.get("metadata", {}), f"{label}.metadata"),
            )
        )
    return tuple(mutations)


def _transaction_ref(value: object, label: str) -> ResourceTransactionRef:
    data = _object(value, label)
    _only_fields(data, {"id", "session_id", "turn_id", "turn_index"}, label)
    turn_index = data.get("turn_index")
    if (
        not isinstance(turn_index, int)
        or isinstance(turn_index, bool)
        or turn_index < 0
    ):
        raise ValueError(f"{label}.turn_index must be a non-negative integer")
    return ResourceTransactionRef(
        id=_required_string(data.get("id"), f"{label}.id"),
        session_id=_required_string(
            data.get("session_id"),
            f"{label}.session_id",
        ),
        turn_id=_required_string(data.get("turn_id"), f"{label}.turn_id"),
        turn_index=turn_index,
    )


def _object(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be an object")
    return cast(Mapping[str, object], value)


def _only_fields(
    value: Mapping[str, object],
    allowed: set[str],
    label: str,
) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"{label} has unknown fields: {sorted(unknown)!r}")


def _required_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _optional_string(value: object, label: str) -> str | None:
    if value is None:
        return None
    return _required_string(value, label)


def _metadata(
    value: object,
    label: str,
) -> dict[str, str | int | float | bool | None]:
    data = _object(value, label)
    metadata: dict[str, str | int | float | bool | None] = {}
    for key, item in data.items():
        if item is not None and not isinstance(item, (str, int, float, bool)):
            raise ValueError(f"{label}[{key!r}] must be a JSON scalar")
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError(f"{label}[{key!r}] must be finite")
        metadata[key] = item
    return metadata


__all__ = ["deserialize_resource_mutations", "serialize_resource_mutations"]
