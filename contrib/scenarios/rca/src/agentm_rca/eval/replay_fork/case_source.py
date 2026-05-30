"""Baseline-case sources for the replay-fork experiment.

The driver consumes :class:`ReplayCase` objects and never touches storage
directly, so swapping where recorded baselines live -- eval.db today, a
JSONL dump or a different schema tomorrow -- means writing another
:class:`CaseSource`, not editing the driver. Knowledge of the eval.db
column layout is confined to :class:`EvalDbCaseSource`.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from agentm.core.abi.messages import AgentMessage

from .trajectory import openai_chat_to_agentm

_logger = logging.getLogger(__name__)

__all__ = ["CaseSource", "EvalDbCaseSource", "ReplayCase"]


@dataclass(frozen=True)
class ReplayCase:
    """One recorded baseline rollout, ready to replay-and-fork.

    ``backbone_messages`` is the recorded control trajectory (the main agent
    is not re-run for control). ``data_dir`` is the resolved dataset case
    directory holding both the parquet telemetry (for intervention
    continuations) and ``injection.json`` (the ground truth the judge
    reads) -- so judging never depends on the storage the backbone came
    from. ``control_response`` / ``control_correct`` carry the baseline's
    own submission + pre-computed correctness when the source has them, so
    cases where the auditor stays silent need no re-judging.
    """

    case_id: str
    system_prompt: str
    backbone_messages: list[AgentMessage]
    data_dir: str
    control_response: str | None = None
    control_correct: bool | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class CaseSource(Protocol):
    """A stream of recorded baseline cases to replay-and-fork."""

    def cases(self) -> Iterator[ReplayCase]:
        ...


class EvalDbCaseSource:
    """Recorded baselines pulled from an ``evaluation_data`` SQLite table.

    Reads the rows for one ``exp_id`` and rehydrates each row's
    ``trajectories`` JSON into a :class:`ReplayCase`. The ``data_dir`` is
    resolved through any symlink to the canonical dataset case directory, so
    a later cleanup of the run's staging symlinks does not break replay and
    the ground-truth ``injection.json`` is always reachable.
    """

    def __init__(
        self,
        db_path: str | os.PathLike[str],
        exp_id: str,
        *,
        limit: int | None = None,
        case_ids: Sequence[str] | None = None,
        skip_case_ids: Sequence[str] | None = None,
    ) -> None:
        self._db_path = str(db_path)
        self._exp_id = exp_id
        self._limit = limit
        self._case_ids = set(case_ids) if case_ids else None
        self._skip_case_ids = set(skip_case_ids) if skip_case_ids else None

    def cases(self) -> Iterator[ReplayCase]:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            sql = (
                "SELECT source, meta, trajectories, response, correct, correct_answer "
                "FROM evaluation_data WHERE exp_id = ? AND trajectories IS NOT NULL "
                "ORDER BY dataset_index"
            )
            params: list[Any] = [self._exp_id]
            if self._limit is not None:
                sql += " LIMIT ?"
                params.append(self._limit)
            for row in conn.execute(sql, params):
                case = self._row_to_case(row)
                if case is None:
                    continue
                if self._case_ids is not None and case.case_id not in self._case_ids:
                    continue
                if self._skip_case_ids is not None and case.case_id in self._skip_case_ids:
                    continue
                yield case
        finally:
            conn.close()

    def _row_to_case(self, row: sqlite3.Row) -> ReplayCase | None:
        case_id = str(row["source"] or "")
        messages = _extract_messages(row["trajectories"])
        if messages is None:
            _logger.warning("EvalDbCaseSource: case %r has no usable trajectory; skipping", case_id)
            return None
        system_prompt, backbone = openai_chat_to_agentm(messages)
        if not backbone:
            _logger.warning("EvalDbCaseSource: case %r rehydrated to 0 messages; skipping", case_id)
            return None
        meta = _parse_meta(row["meta"])
        data_dir = _resolve_data_dir(meta.get("path"))
        if data_dir is None:
            _logger.warning(
                "EvalDbCaseSource: case %r has no resolvable data_dir (meta.path=%r); skipping",
                case_id,
                meta.get("path"),
            )
            return None
        correct = row["correct"]
        return ReplayCase(
            case_id=case_id,
            system_prompt=system_prompt,
            backbone_messages=backbone,
            data_dir=data_dir,
            control_response=row["response"],
            control_correct=bool(correct) if correct is not None else None,
            meta=meta,
        )


def _extract_messages(trajectories_raw: Any) -> list[dict[str, Any]] | None:
    """Pull the flat OpenAI message list out of the eval.db trajectories blob.

    Shape: ``{"trajectories": [{"trajectory_id", "agent_name",
    "messages": [...]}]}``. The first (``main``) trajectory's messages are
    used. Returns ``None`` on any structural mismatch.
    """
    if not isinstance(trajectories_raw, str):
        return None
    try:
        obj = json.loads(trajectories_raw)
    except json.JSONDecodeError:
        return None
    trajs = obj.get("trajectories") if isinstance(obj, dict) else None
    if not isinstance(trajs, list) or not trajs:
        return None
    first = trajs[0]
    msgs = first.get("messages") if isinstance(first, dict) else None
    if not isinstance(msgs, list):
        return None
    return [m for m in msgs if isinstance(m, dict)]


def _parse_meta(meta_raw: Any) -> dict[str, Any]:
    if not isinstance(meta_raw, str):
        return {}
    try:
        obj = json.loads(meta_raw)
    except json.JSONDecodeError:
        return {}
    return obj if isinstance(obj, dict) else {}


def _resolve_data_dir(path: Any) -> str | None:
    """Resolve a recorded data_dir to the canonical case directory.

    The recorded path is typically a per-run staging symlink
    (``eval-data/<exp>/data_<hash>``) pointing at the durable dataset case
    dir (``datasets/.../batch-<id>``). ``realpath`` follows it; the result
    must exist (it holds the parquet + injection.json the continuation and
    judge need).
    """
    if not isinstance(path, str) or not path:
        return None
    resolved = os.path.realpath(path)
    if not os.path.isdir(resolved):
        return None
    return resolved
