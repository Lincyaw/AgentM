"""Execution-error ledger shared by verifier graph state."""

from __future__ import annotations

from collections.abc import Callable

from .lib.schema import ExecutionError


class ExecutionLedgerMixin:
    """Record and clear verifier execution failures by stable stage key."""

    log: Callable[[str], None]
    execution_errors: dict[str, ExecutionError]

    @staticmethod
    def execution_key(stage: str, item: str) -> str:
        return stage + ":" + item

    def record_error(self, stage: str, item: str, reason: str) -> None:
        self.execution_errors[self.execution_key(stage, item)] = {
            "stage": stage,
            "item": item,
            "reason": reason,
        }
        self.log(f"⚠ {stage} {item}: {reason}")

    def clear_error(self, stage: str, item: str) -> None:
        self.execution_errors.pop(self.execution_key(stage, item), None)


__all__ = ["ExecutionLedgerMixin"]
