"""Export AgentM traces to HuggingFace-compatible Parquet datasets.

Provides a programmatic SDK and backs the ``agentm trace export-dataset`` CLI.
Two backends: ClickHouse (bulk SQL, native Parquet) and local JSONL (parallel
file reads via :class:`TraceReader`).

Quick start::

    from agentm.dataset_export import DatasetExporter

    exporter = DatasetExporter.auto()
    count = exporter.export_parquet("traces.parquet", scenarios={"chatbot"})

    # Streaming iterator — bounded memory:
    for conv in exporter.iter_conversations(scenarios={"chatbot"}):
        print(conv.session_id, len(conv.messages))

    # Raw ClickHouse → Parquet (no Python transform, maximum speed):
    exporter.export_raw_parquet("raw.parquet")

Output schema (one row per session)::

    session_id      VARCHAR
    messages        JSON      -- OpenAI-compatible chat messages
    scenario        VARCHAR
    purpose         VARCHAR
    model           VARCHAR
    turns           INTEGER
    input_tokens    BIGINT
    output_tokens   BIGINT
    tool_call_count INTEGER

Performance: ClickHouse backend uses three ``groupArray`` SQL queries per
batch (O(sessions) rows returned, not O(messages)); local backend uses
``ThreadPoolExecutor`` for parallel file reads.  ``export_raw_parquet``
bypasses Python entirely — ClickHouse produces Parquet in a single HTTP
round-trip.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Protocol, runtime_checkable

from loguru import logger


@dataclass(slots=True)
class Conversation:
    """One exported conversation."""

    session_id: str
    messages: list[dict[str, Any]]
    scenario: str | None = None
    purpose: str | None = None
    model: str | None = None
    turns: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    tool_call_count: int = 0


# ---------------------------------------------------------------------------
# Message format conversion
# ---------------------------------------------------------------------------


def _entry_to_chat_messages(
    entry: dict[str, Any],
    *,
    include_thinking: bool = False,
) -> list[dict[str, Any]]:
    """Convert a SessionEntry dict to OpenAI chat format message(s)."""
    if entry.get("type") != "message":
        return []
    payload = entry.get("payload")
    if not isinstance(payload, dict):
        return []

    role = payload.get("role")
    content_blocks = payload.get("content") or []

    if role == "system":
        text = "\n".join(
            block.get("text", "")
            for block in content_blocks
            if isinstance(block, dict) and block.get("type") == "text"
        )
        if not text:
            return []
        return [{"role": "system", "content": text}]

    if role == "user":
        text = "\n".join(
            block.get("text", "")
            for block in content_blocks
            if isinstance(block, dict) and block.get("type") == "text"
        )
        if not text:
            return []
        return [{"role": "user", "content": text}]

    if role == "assistant":
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for block in content_blocks:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                text_parts.append(block.get("text", ""))
            elif btype == "tool_call":
                args = block.get("arguments")
                tool_calls.append({
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": (
                            json.dumps(args, ensure_ascii=False)
                            if isinstance(args, dict) else str(args or "")
                        ),
                    },
                })
            elif btype == "thinking" and include_thinking:
                text_parts.append(f"<think>{block.get('text', '')}</think>")
        content = "\n".join(text_parts)
        msg: dict[str, Any] = {"role": "assistant"}
        msg["content"] = content if content else None
        if tool_calls:
            msg["tool_calls"] = tool_calls
        return [msg]

    if role == "tool_result":
        out: list[dict[str, Any]] = []
        for block in content_blocks:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            result_content = block.get("content") or []
            text = "\n".join(
                b.get("text", "")
                for b in result_content
                if isinstance(b, dict) and b.get("type") == "text"
            )
            out.append({
                "role": "tool",
                "content": text,
                "tool_call_id": block.get("tool_call_id", ""),
            })
        return out

    return []


def _entries_to_conversation(
    entries: list[dict[str, Any]],
    *,
    include_thinking: bool = False,
) -> list[dict[str, Any]]:
    """Convert a list of SessionEntry dicts to OpenAI chat messages."""
    messages: list[dict[str, Any]] = []
    for entry in entries:
        messages.extend(
            _entry_to_chat_messages(entry, include_thinking=include_thinking)
        )
    return messages


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class _Backend(Protocol):
    def list_sessions(
        self,
        *,
        scenarios: set[str] | None,
        purposes: set[str] | None,
        roots_only: bool,
        limit: int | None,
    ) -> list[dict[str, Any]]: ...

    def load_conversations(
        self,
        session_ids: list[str],
        *,
        include_thinking: bool,
        include_system_prompt: bool,
    ) -> list[Conversation]: ...


# ---------------------------------------------------------------------------
# ClickHouse backend
# ---------------------------------------------------------------------------


def _system_prompt_entry(text: str) -> dict[str, Any]:
    """Synthetic SessionEntry carrying the system prompt.

    Shaped like ``clickhouse.messages()``'s system row so both backends and
    the trace CLI agree on one representation.
    """
    return {
        "type": "message", "id": "system-prompt",
        "parent_id": None, "timestamp": 0,
        "payload": {
            "role": "system",
            "content": [{"type": "text", "text": text}],
            "timestamp": 0,
        },
    }


class _ClickHouseBackend:
    def __init__(self, url: str) -> None:
        self._url = url

    def list_sessions(
        self,
        *,
        scenarios: set[str] | None = None,
        purposes: set[str] | None = None,
        roots_only: bool = False,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        from agentm.core.observability import clickhouse as ch
        return list(ch.index(
            self._url, purposes=purposes, scenarios=scenarios,
            roots_only=roots_only,
        ))[:limit]

    def load_conversations(
        self,
        session_ids: list[str],
        *,
        include_thinking: bool = False,
        include_system_prompt: bool = False,
    ) -> list[Conversation]:
        if not session_ids:
            return []
        from agentm.core.observability import clickhouse as ch

        messages_by_sid = ch.bulk_session_entries(self._url, session_ids)

        if include_system_prompt:
            for sid, text in ch.bulk_system_prompts(self._url, session_ids).items():
                if sid in messages_by_sid:
                    messages_by_sid[sid].insert(0, _system_prompt_entry(text))

        usage_by_sid = ch.bulk_turn_usage(self._url, session_ids)

        model_by_sid: dict[str, str] = {}
        try:
            model_by_sid = ch.bulk_models(self._url, session_ids)
        except Exception as exc:
            logger.debug("dataset_export: model query failed: {}", exc)

        result: list[Conversation] = []
        for sid in session_ids:
            entries = messages_by_sid.get(sid, [])
            chat_msgs = _entries_to_conversation(entries, include_thinking=include_thinking)
            if not chat_msgs:
                continue
            u = usage_by_sid.get(sid, {})
            result.append(Conversation(
                session_id=sid,
                messages=chat_msgs,
                model=model_by_sid.get(sid),
                turns=u.get("turns", 0),
                input_tokens=u.get("input_tokens", 0),
                output_tokens=u.get("output_tokens", 0),
                tool_call_count=sum(
                    len(m.get("tool_calls", []))
                    for m in chat_msgs if m.get("role") == "assistant"
                ),
            ))
        return result

    def export_raw_parquet(
        self,
        output: Path,
        *,
        scenarios: set[str] | None = None,
        purposes: set[str] | None = None,
        roots_only: bool = False,
    ) -> int:
        """ClickHouse → Parquet in one HTTP round-trip, no Python transform."""
        from agentm.core.observability import clickhouse as ch

        sessions = list(ch.index(
            self._url, purposes=purposes, scenarios=scenarios,
            roots_only=roots_only,
        ))
        sids = [s["session_id"] for s in sessions if s.get("session_id")]
        if not sids:
            return 0
        data = ch.raw_parquet_export(self._url, sids)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(data)
        return len(sids)


# ---------------------------------------------------------------------------
# Local JSONL backend
# ---------------------------------------------------------------------------


class _LocalBackend:
    def __init__(self, obs_dir: Path) -> None:
        self._obs_dir = obs_dir

    def _jsonl_files(self) -> list[Path]:
        if not self._obs_dir.is_dir():
            return []
        return sorted(self._obs_dir.glob("*.jsonl"))

    def list_sessions(
        self,
        *,
        scenarios: set[str] | None = None,
        purposes: set[str] | None = None,
        roots_only: bool = False,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        from agentm.core.lib.trace_reader import TraceReader

        files = self._jsonl_files()
        results: list[dict[str, Any]] = []

        def _scan(path: Path) -> dict[str, Any] | None:
            reader = TraceReader(path)
            identity, line_count = reader.scan_identity_and_line_count()
            if identity is None:
                return None
            if scenarios and identity.scenario not in scenarios:
                return None
            if purposes and identity.purpose not in purposes:
                return None
            if roots_only and identity.parent_session_id is not None:
                return None
            return {
                "session_id": identity.session_id,
                "trace_id": identity.trace_id,
                "parent_session_id": identity.parent_session_id,
                "purpose": identity.purpose,
                "scenario": identity.scenario,
                "records": line_count,
            }

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(_scan, f): f for f in files}
            for future in as_completed(futures):
                try:
                    row = future.result()
                except Exception as exc:
                    logger.debug("dataset_export: scan failed for {}: {}",
                                 futures[future], exc)
                    continue
                if row is not None:
                    results.append(row)
                    if limit is not None and len(results) >= limit:
                        break

        return results[:limit]

    def load_conversations(
        self,
        session_ids: list[str],
        *,
        include_thinking: bool = False,
        include_system_prompt: bool = False,
    ) -> list[Conversation]:
        if not session_ids:
            return []
        from agentm.core.lib.trace_reader import TraceReader

        sid_set = set(session_ids)
        files = [f for f in self._jsonl_files() if f.stem in sid_set]

        def _load_one(path: Path) -> Conversation | None:
            reader = TraceReader(path)
            identity = reader.first_session_identity()
            if identity is None or identity.session_id is None:
                return None

            entries = reader.load_messages()
            if include_system_prompt:
                for rec in reader.iter_log_records(name="agentm.llm.system_prompt"):
                    body = rec.body if isinstance(rec.body, dict) else {}
                    text = body.get("text", "")
                    if text:
                        entries.insert(0, _system_prompt_entry(text))
                    break

            chat_msgs = _entries_to_conversation(entries, include_thinking=include_thinking)
            if not chat_msgs:
                return None

            turn_summaries = reader.load_turn_summaries()
            model: str | None = None
            for span in reader.chat_calls():
                if span.name.startswith("chat "):
                    model = span.name.removeprefix("chat ").strip()
                break

            return Conversation(
                session_id=identity.session_id,
                messages=chat_msgs,
                scenario=identity.scenario,
                purpose=identity.purpose,
                model=model,
                turns=len(turn_summaries),
                input_tokens=sum(t.get("input_tokens", 0) for t in turn_summaries),
                output_tokens=sum(t.get("output_tokens", 0) for t in turn_summaries),
                tool_call_count=sum(
                    len(m.get("tool_calls", []))
                    for m in chat_msgs if m.get("role") == "assistant"
                ),
            )

        result: list[Conversation] = []
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(_load_one, f): f for f in files}
            for future in as_completed(futures):
                try:
                    conv = future.result()
                except Exception as exc:
                    logger.debug("dataset_export: load failed for {}: {}",
                                 futures[future], exc)
                    continue
                if conv is not None:
                    result.append(conv)
        return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


_BATCH_SIZE = 200

_CREATE_TABLE_SQL = """\
CREATE TABLE conversations (
    session_id      VARCHAR,
    messages        JSON,
    scenario        VARCHAR,
    purpose         VARCHAR,
    model           VARCHAR,
    turns           INTEGER,
    input_tokens    BIGINT,
    output_tokens   BIGINT,
    tool_call_count INTEGER
)"""

_INSERT_SQL = "INSERT INTO conversations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"

_PARQUET_COMPRESSIONS = frozenset(
    {"uncompressed", "snappy", "gzip", "zstd", "brotli", "lz4", "lz4_raw"}
)


def _conv_to_row(conv: Conversation) -> tuple[Any, ...]:
    return (
        conv.session_id,
        json.dumps(conv.messages, ensure_ascii=False),
        conv.scenario, conv.purpose, conv.model,
        conv.turns, conv.input_tokens, conv.output_tokens, conv.tool_call_count,
    )


class DatasetExporter:
    """Export AgentM traces to HuggingFace-compatible datasets."""

    def __init__(self, backend: _Backend) -> None:
        self._backend = backend

    @classmethod
    def from_clickhouse(cls, url: str | None = None) -> DatasetExporter:
        if url is None:
            from agentm.core.observability.clickhouse import get_url
            url = get_url()
        if url is None:
            raise RuntimeError(
                "No ClickHouse URL. Set AGENTM_CLICKHOUSE_URL or use from_local()."
            )
        return cls(_ClickHouseBackend(url))

    @classmethod
    def from_local(cls, obs_dir: Path | str | None = None) -> DatasetExporter:
        if obs_dir is None:
            from agentm.core.lib.observability_dir import resolve_observability_dir
            obs_dir = resolve_observability_dir()
        return cls(_LocalBackend(Path(obs_dir)))

    @classmethod
    def auto(cls) -> DatasetExporter:
        try:
            return cls.from_clickhouse()
        except RuntimeError:
            return cls.from_local()

    def list_sessions(
        self,
        *,
        scenarios: set[str] | None = None,
        purposes: set[str] | None = None,
        roots_only: bool = False,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        return self._backend.list_sessions(
            scenarios=scenarios, purposes=purposes,
            roots_only=roots_only, limit=limit,
        )

    def iter_conversations(
        self,
        *,
        session_ids: list[str] | None = None,
        scenarios: set[str] | None = None,
        purposes: set[str] | None = None,
        roots_only: bool = False,
        include_system_prompt: bool = False,
        include_thinking: bool = False,
        limit: int | None = None,
    ) -> Iterator[Conversation]:
        """Yield conversations. Loads in batches of 200 for bounded memory."""
        if session_ids is None:
            rows = self.list_sessions(
                scenarios=scenarios, purposes=purposes,
                roots_only=roots_only, limit=limit,
            )
            session_ids = [r["session_id"] for r in rows if r.get("session_id")]

        emitted = 0
        for i in range(0, len(session_ids), _BATCH_SIZE):
            batch = session_ids[i : i + _BATCH_SIZE]
            for conv in self._backend.load_conversations(
                batch,
                include_thinking=include_thinking,
                include_system_prompt=include_system_prompt,
            ):
                if scenarios and conv.scenario not in scenarios:
                    continue
                if purposes and conv.purpose not in purposes:
                    continue
                yield conv
                emitted += 1
                if limit is not None and emitted >= limit:
                    return

    def export_parquet(
        self,
        output: Path | str,
        *,
        session_ids: list[str] | None = None,
        scenarios: set[str] | None = None,
        purposes: set[str] | None = None,
        roots_only: bool = False,
        include_system_prompt: bool = False,
        include_thinking: bool = False,
        compression: str = "zstd",
        limit: int | None = None,
    ) -> int:
        """Export to Parquet with OpenAI-format messages. Returns row count."""
        import duckdb

        if compression.lower() not in _PARQUET_COMPRESSIONS:
            raise ValueError(
                f"unsupported parquet compression {compression!r}; "
                f"expected one of {sorted(_PARQUET_COMPRESSIONS)}"
            )
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        conn = duckdb.connect(":memory:")
        conn.execute(_CREATE_TABLE_SQL)

        count = 0
        batch: list[tuple[Any, ...]] = []
        for conv in self.iter_conversations(
            session_ids=session_ids, scenarios=scenarios, purposes=purposes,
            roots_only=roots_only, include_system_prompt=include_system_prompt,
            include_thinking=include_thinking, limit=limit,
        ):
            batch.append(_conv_to_row(conv))
            count += 1
            if len(batch) >= 1000:
                conn.executemany(_INSERT_SQL, batch)
                batch.clear()
        if batch:
            conn.executemany(_INSERT_SQL, batch)
        if count == 0:
            conn.close()
            return 0

        # DuckDB COPY does not support a bound parameter for the target path;
        # single-quote doubling is the documented literal escape.
        escaped_output = str(output).replace("'", "''")
        conn.execute(
            f"COPY conversations TO '{escaped_output}' "
            f"(FORMAT PARQUET, COMPRESSION '{compression.lower()}')"
        )
        conn.close()
        return count

    def export_raw_parquet(
        self,
        output: Path | str,
        *,
        scenarios: set[str] | None = None,
        purposes: set[str] | None = None,
        roots_only: bool = False,
    ) -> int:
        """ClickHouse → Parquet in one HTTP round-trip.

        Writes raw OTLP message bodies (no OpenAI conversion) grouped by
        session. Only available on the ClickHouse backend.
        """
        backend = self._backend
        if not isinstance(backend, _ClickHouseBackend):
            raise RuntimeError("export_raw_parquet requires ClickHouse backend")
        return backend.export_raw_parquet(
            Path(output), scenarios=scenarios, purposes=purposes,
            roots_only=roots_only,
        )

    def export_jsonl(
        self,
        output: Path | str,
        *,
        session_ids: list[str] | None = None,
        scenarios: set[str] | None = None,
        purposes: set[str] | None = None,
        roots_only: bool = False,
        include_system_prompt: bool = False,
        include_thinking: bool = False,
        limit: int | None = None,
    ) -> int:
        """Export to JSONL. Streaming — no full materialisation."""
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with output.open("w", encoding="utf-8") as fh:
            for conv in self.iter_conversations(
                session_ids=session_ids, scenarios=scenarios, purposes=purposes,
                roots_only=roots_only, include_system_prompt=include_system_prompt,
                include_thinking=include_thinking, limit=limit,
            ):
                fh.write(json.dumps({
                    "session_id": conv.session_id,
                    "messages": conv.messages,
                    "scenario": conv.scenario,
                    "purpose": conv.purpose,
                    "model": conv.model,
                    "turns": conv.turns,
                    "input_tokens": conv.input_tokens,
                    "output_tokens": conv.output_tokens,
                    "tool_call_count": conv.tool_call_count,
                }, ensure_ascii=False))
                fh.write("\n")
                count += 1
        return count
