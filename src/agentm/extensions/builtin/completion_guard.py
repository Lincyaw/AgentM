"""Double-confirm before the model ends the task.

Terminal-Bench's reference agent (terminus_2) requires the model to assert
completion twice: the first ``task_complete`` signal triggers a confirmation
prompt ("Are you sure? This will trigger grading"), and only a second
assertion ends the run. Our native tool-calling loop ends as soon as the
model stops emitting tool calls (``ModelEndTurn``) with no such safety net,
so a model that gives up early loses whatever it left unfinished.

This atom restores that safety net for parity in benchmark comparisons: on
``ModelEndTurn`` it injects a confirmation prompt and continues the loop,
letting the run stop only after ``confirmations`` consecutive finishes. Any
resumed work (the model calls a tool again, i.e. a ``Step``) resets the
counter, so a later finish is confirmed afresh. It is a self-contained eval
aid -- not the ``goal`` oversight machinery, which spawns checker sessions.
"""

from __future__ import annotations

from loguru import logger
from pydantic import BaseModel

from agentm.core.abi import (
    DecideTurnActionEvent,
    ExtensionAPI,
    Inject,
    ModelEndTurn,
    Stop,
    text_message,
)
from agentm.extensions import ExtensionManifest

_DEFAULT_PROMPT = (
    "Are you sure the task is complete? Ending now submits your work for "
    "grading and cannot be undone. If anything the task asked for is "
    "unfinished, untested, or unverified, keep working and check it. If you "
    "are certain everything is done and verified, end your turn again to "
    "submit."
)


class CompletionGuardConfig(BaseModel):
    confirmations: int = 1
    prompt: str = _DEFAULT_PROMPT


MANIFEST = ExtensionManifest(
    name="completion_guard",
    description=(
        "Inject a confirmation prompt on ModelEndTurn and require the model "
        "to re-assert completion before the loop stops (mirrors terminus_2's "
        "double-confirm; prevents premature give-up in benchmarks)."
    ),
    registers=("event:decide_turn_action",),
    config_schema=CompletionGuardConfig,
    requires=(),
)


class _CompletionGuardRuntime:
    def __init__(self, api: ExtensionAPI, config: CompletionGuardConfig) -> None:
        self._api = api
        self._confirmations = max(0, config.confirmations)
        self._prompt = config.prompt
        self._used = 0

    def install(self) -> None:
        self._api.on(DecideTurnActionEvent.CHANNEL, self.on_decide)

    def on_decide(self, event: DecideTurnActionEvent) -> Inject | None:
        default = event.observation.default_action

        # Only guard a voluntary end-turn. A Step (still working) or an
        # explicit terminal-tool Stop is left untouched, and resets the
        # counter so the next voluntary finish is confirmed afresh.
        if not (isinstance(default, Stop) and isinstance(default.cause, ModelEndTurn)):
            self._used = 0
            return None

        if self._used >= self._confirmations:
            self._used = 0
            return None

        self._used += 1
        logger.info(
            "completion_guard: confirming completion ({}/{})",
            self._used,
            self._confirmations,
        )
        return Inject(messages=[text_message(self._prompt)])


def install(api: ExtensionAPI, config: CompletionGuardConfig) -> None:
    _CompletionGuardRuntime(api, config).install()
