"""Crash-aware JSONL trajectory store with one file per session."""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

from agentm.core.abi.codec import CodecRegistry
from agentm.core.abi.store import SessionMeta
from agentm.core.abi.trajectory import Turn, TurnRef
from agentm.core.lib.trajectory_store import (
    turn_prefix_cut,
    validate_turn_append,
    validate_turn_sequence,
)


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
        self._codec = codec if codec is not None else CodecRegistry()

    @property
    def codec(self) -> CodecRegistry:
        return self._codec

    def _path(self, session_id: str) -> Path:
        _validate_session_id(session_id)
        return self._dir / f"{session_id}.jsonl"

    def file_path(self, session_id: str) -> Path:
        return self._path(session_id)

    def create_session(self, meta: SessionMeta) -> None:
        self.create_session_with_turns(meta, ())

    def create_session_with_turns(
        self, meta: SessionMeta, turns: Sequence[Turn]
    ) -> None:
        copied = list(turns)
        validate_turn_sequence(copied)
        records = [self._codec.serialize_session_meta(meta)]
        records.extend(self._codec.serialize_turn(turn) for turn in copied)
        payload = b"".join(self._encode_record(record) for record in records)
        path = self._path(meta.id)
        fd, temp_name = tempfile.mkstemp(prefix=f".{meta.id}.", dir=self._dir)
        temp_path = Path(temp_name)
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            try:
                os.link(temp_path, path)
            except FileExistsError:
                raise ValueError(f"session already exists: {meta.id}") from None
            self._fsync_directory()
        finally:
            temp_path.unlink(missing_ok=True)

    def append(self, session_id: str, turn: Turn) -> None:
        path = self._path(session_id)
        try:
            fh = path.open("r+b")
        except FileNotFoundError:
            raise KeyError(session_id) from None
        record = self._encode_record(self._codec.serialize_turn(turn))
        with fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            raw = fh.read()
            _, turns, valid_bytes, terminated = self._decode_records(
                raw,
                path,
                allow_trailing_torn=True,
            )
            validate_turn_append(turns, turn)
            fh.truncate(valid_bytes)
            fh.seek(valid_bytes)
            if not terminated:
                fh.write(b"\n")
            fh.write(record)
            fh.flush()
            os.fsync(fh.fileno())

    def load(self, session_id: str) -> tuple[SessionMeta, list[Turn]]:
        return self._read(session_id)

    def load_prefix(
        self,
        session_id: str,
        up_to: TurnRef,
    ) -> tuple[SessionMeta, list[Turn]]:
        meta, turns = self._read(session_id)
        cut = turn_prefix_cut(turns, up_to)
        return (meta, turns[: cut + 1])

    def session_children(self, session_id: str) -> list[str]:
        return [
            meta.id
            for meta in self.list_sessions()
            if meta.parent_id == session_id
        ]

    def session_exists(self, session_id: str) -> bool:
        return self._path(session_id).exists()

    def list_sessions(self) -> list[SessionMeta]:
        return [
            self._read_meta(path)
            for path in sorted(self._dir.glob("*.jsonl"))
        ]

    def _read(self, session_id: str) -> tuple[SessionMeta, list[Turn]]:
        path = self._path(session_id)
        if not path.exists():
            raise KeyError(session_id)
        meta, turns, _, _ = self._read_records(path, allow_trailing_torn=True)
        return meta, turns

    def _read_records(
        self,
        path: Path,
        *,
        allow_trailing_torn: bool,
    ) -> tuple[SessionMeta, list[Turn], int, bool]:
        return self._decode_records(
            path.read_bytes(),
            path,
            allow_trailing_torn=allow_trailing_torn,
        )

    def _decode_records(
        self,
        raw: bytes,
        path: Path,
        *,
        allow_trailing_torn: bool,
    ) -> tuple[SessionMeta, list[Turn], int, bool]:
        if not raw:
            raise ValueError(f"corrupt empty session file: {path}")
        chunks = raw.splitlines(keepends=True)
        meta: SessionMeta | None = None
        turns: list[Turn] = []
        valid_bytes = 0
        valid_terminated = False
        for lineno, chunk in enumerate(chunks, start=1):
            terminated = chunk.endswith(b"\n") or chunk.endswith(b"\r")
            try:
                data = json.loads(chunk)
                if meta is None:
                    meta = self._codec.deserialize_session_meta(data)
                else:
                    turns.append(self._codec.deserialize_turn(data))
            except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                is_last = lineno == len(chunks)
                if allow_trailing_torn and is_last and not terminated and meta is not None:
                    break
                raise ValueError(
                    f"corrupt record {lineno} in {path}: {exc}"
                ) from exc
            valid_bytes += len(chunk)
            valid_terminated = terminated
        if meta is None:
            raise ValueError(f"corrupt session metadata in {path}")
        validate_turn_sequence(turns)
        return meta, turns, valid_bytes, valid_terminated

    def _read_meta(self, path: Path) -> SessionMeta:
        meta, _, _, _ = self._read_records(path, allow_trailing_torn=True)
        return meta

    @staticmethod
    def _encode_record(record: object) -> bytes:
        return (
            json.dumps(record, separators=(",", ":")) + "\n"
        ).encode("utf-8")

    def _fsync_directory(self) -> None:
        fd = os.open(self._dir, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _validate_session_id(session_id: str) -> None:
    if (
        not session_id
        or session_id in {".", ".."}
        or Path(session_id).name != session_id
        or "\\" in session_id
        or "\x00" in session_id
    ):
        raise ValueError(f"session_id is not a valid path token: {session_id!r}")


__all__ = ["JsonlTrajectoryStore"]
