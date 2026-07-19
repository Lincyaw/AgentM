"""JSONL-file TrajectoryStore — one file per session, append-only.

File layout for ``{session_id}.jsonl``::

    <line 0>  SessionMeta as JSON
    <line 1>  Turn as JSON
    <line 2>  Turn as JSON
    ...

Each Turn is one complete JSON line, written in append mode.  A crash
mid-write leaves a trailing partial (unterminated / unparseable) line,
which ``load`` skips — so recovery needs no truncation or markers.

Serialization is delegated to :class:`CodecRegistry` from ``codec.py``
so custom Trigger types round-trip correctly when their atom registers
a codec.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from agentm.core.abi.codec import CodecRegistry, DEFAULT_CODEC
from agentm.core.abi.store import SessionMeta
from agentm.core.abi.trajectory import Turn, TurnRef
from agentm.core.runtime.stores.memory import _prefix_cut

logger = logging.getLogger(__name__)


class JsonlTrajectoryStore:
    """A ``TrajectoryStore`` persisted as one JSONL file per session."""

    __slots__ = ("_dir", "_codec")

    def __init__(
        self,
        directory: Path,
        codec: CodecRegistry | None = None,
    ) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._codec = codec or DEFAULT_CODEC

    def _path(self, session_id: str) -> Path:
        return self._dir / f"{session_id}.jsonl"

    def file_path(self, session_id: str) -> Path:
        return self._path(session_id)

    def create_session(self, meta: SessionMeta) -> None:
        path = self._path(meta.id)
        try:
            fh = path.open("x", encoding="utf-8")
        except FileExistsError:
            raise ValueError(f"session already exists: {meta.id}") from None
        with fh:
            fh.write(json.dumps(self._codec.serialize_session_meta(meta)) + "\n")

    def append(self, session_id: str, turn: Turn) -> None:
        path = self._path(session_id)
        if not path.exists():
            raise KeyError(session_id)
        line = json.dumps(self._codec.serialize_turn(turn), default=str) + "\n"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()

    def load(self, session_id: str) -> tuple[SessionMeta, list[Turn]]:
        return self._read(session_id)

    def load_prefix(
        self,
        session_id: str,
        up_to: TurnRef,
    ) -> tuple[SessionMeta, list[Turn]]:
        meta, turns = self._read(session_id)
        cut = _prefix_cut(turns, up_to)
        return (meta, turns[: cut + 1])

    def session_children(self, session_id: str) -> list[str]:
        children: list[str] = []
        for path in self._dir.glob("*.jsonl"):
            meta = self._read_meta(path)
            if meta is not None and meta.parent_id == session_id:
                children.append(meta.id)
        return children

    def session_exists(self, session_id: str) -> bool:
        return self._path(session_id).exists()

    def list_sessions(self) -> list[SessionMeta]:
        result: list[SessionMeta] = []
        for path in self._dir.glob("*.jsonl"):
            meta = self._read_meta(path)
            if meta is not None:
                result.append(meta)
        return result

    def _read(self, session_id: str) -> tuple[SessionMeta, list[Turn]]:
        path = self._path(session_id)
        if not path.exists():
            raise KeyError(session_id)
        meta: SessionMeta | None = None
        turns: list[Turn] = []
        with path.open("r", encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh):
                line = raw.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(
                        "skipping malformed line %d in %s", lineno, path
                    )
                    continue
                if meta is None:
                    try:
                        meta = self._codec.deserialize_session_meta(data)
                    except Exception as exc:
                        raise KeyError(
                            f"corrupt session meta for {session_id}: {exc}"
                        ) from exc
                    continue
                try:
                    turns.append(self._codec.deserialize_turn(data))
                except (KeyError, TypeError, ValueError) as exc:
                    logger.warning(
                        "skipping malformed turn on line %d in %s: %s",
                        lineno,
                        path,
                        exc,
                    )
        if meta is None:
            raise KeyError(session_id)
        return (meta, turns)

    def _read_meta(self, path: Path) -> SessionMeta | None:
        try:
            with path.open("r", encoding="utf-8") as fh:
                first = fh.readline().strip()
        except OSError:
            return None
        if not first:
            return None
        try:
            return self._codec.deserialize_session_meta(json.loads(first))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None


__all__ = ["JsonlTrajectoryStore"]
