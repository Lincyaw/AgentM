"""Workflow resume journal: persistence, invalidation, and the resume rule.

The journal is backed by the ``artifact_store`` service. Each ``agent()``
result is keyed by ``sha256(prompt, opts)`` and written as one artifact
(``kind="workflow_journal"``); an *invalidation* is a second, append-only
artifact kind flagging a key. The single resume rule lives here, in one
place: **an invalidation newer than the newest result is pending** — the key
is a forced miss until a fresh result (recorded under the same key)
supersedes the flag.

Entries store the script-authored prompt alongside the result in a versioned
JSON envelope so lineage derivation (which node's result flowed into which
node's prompt) needs no trace join. See reliability-substrate.md §4.3.

Addressing convention (single writer, two readers): journal and invalidation
artifacts carry the key as BOTH ``title`` and the sole ``tags`` element —
``_write_keyed`` is the only writer; per-key readers query by tags, bulk
readers group by title.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Final, Protocol

from loguru import logger

from agentm.core.abi import ToolResult


class _ArtifactStore(Protocol):
    """The ``artifact_store`` service surface the journal consumes."""

    async def list_artifacts(self, args: dict[str, Any]) -> ToolResult: ...
    async def read(self, args: dict[str, Any]) -> ToolResult: ...
    async def write_artifact(
        self,
        *,
        kind: str,
        title: str,
        body: str,
        tags: list[str] | None = None,
    ) -> dict[str, str]: ...


_JOURNAL_KIND: Final[str] = "workflow_journal"
_JOURNAL_INVALIDATION_KIND: Final[str] = "workflow_journal_invalidation"
_JOURNAL_ENVELOPE_MARKER: Final[str] = "__workflow_journal__"
_JOURNAL_ENVELOPE_VERSION: Final[int] = 2
# artifact_read without a range truncates bodies at inline_max_bytes; journal
# correctness needs the full body, so every journal read passes an explicit
# byte range covering any realistic size.
_FULL_BODY_RANGE: Final[dict[str, list[int]]] = {"bytes": [0, 1 << 31]}
# Upper bound on journal size for bulk loads / the prime-time invalidation
# index. Far above any realistic workflow (agent-count backstop is 1000).
_BULK_LIMIT: Final[int] = 2000


@dataclass(slots=True)
class JournalInvalidation:
    """Pending invalidation flag for one journal key (append-only record)."""

    reason: str
    feedback: str | None
    carry_previous: bool


@dataclass(slots=True)
class JournalEntry:
    """Decoded newest journal record for one key (lineage / tooling view)."""

    key: str
    result: str
    prompt: str | None
    timestamp: float
    invalidated: bool = False


@dataclass(slots=True)
class _JournalState:
    """Resume-relevant state of one key.

    ``cached`` is the newest recorded result (also present when a pending
    invalidation forces a miss — the re-run may carry it as reference).
    ``invalidation`` is set only when it is *pending*: newer than the newest
    result record, i.e. not yet superseded by a re-run."""

    cached: str | None = None
    invalidation: JournalInvalidation | None = None


def _is_pending(invalidation_ts: float, result_ts: float) -> bool:
    """THE resume rule: a flag newer than the newest result forces a miss."""
    return invalidation_ts > result_ts


def _encode_journal_body(prompt: str, result: str) -> str:
    return json.dumps(
        {
            _JOURNAL_ENVELOPE_MARKER: _JOURNAL_ENVELOPE_VERSION,
            "prompt": prompt,
            "result": result,
        },
        ensure_ascii=False,
    )


def _decode_journal_body(body: str) -> tuple[str | None, str]:
    """Return ``(prompt, result)``; legacy plain-string bodies → ``(None, body)``."""
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        return None, body
    if isinstance(payload, dict) and payload.get(_JOURNAL_ENVELOPE_MARKER) is not None:
        prompt = payload.get("prompt")
        return (
            prompt if isinstance(prompt, str) else None,
            str(payload.get("result", "")),
        )
    return None, body


def _listed_artifacts(listing: ToolResult) -> list[dict[str, Any]]:
    if listing.is_error or not listing.extras:
        return []
    artifacts = listing.extras.get("artifacts")
    return list(artifacts) if isinstance(artifacts, list) else []


def _artifact_timestamp(item: dict[str, Any]) -> float:
    created = item.get("created_by")
    ts = created.get("timestamp") if isinstance(created, dict) else None
    return float(ts) if isinstance(ts, (int, float)) else 0.0


async def _read_full_body(store: _ArtifactStore, artifact_id: str) -> str | None:
    read = await store.read({"artifact_id": artifact_id, "range": _FULL_BODY_RANGE})
    if read.is_error or not read.extras:
        return None
    body = read.extras.get("body")
    return body if isinstance(body, str) else None


async def _newest_for_key(
    store: _ArtifactStore, kind: str, key: str
) -> tuple[str, float] | None:
    """``(artifact_id, timestamp)`` of the newest ``kind`` artifact tagged ``key``."""
    listed = _listed_artifacts(
        await store.list_artifacts({"kind": kind, "tags": [key], "limit": 1})
    )
    if not listed:
        return None
    artifact_id = str(listed[0].get("id", ""))
    if not artifact_id:
        return None
    return artifact_id, _artifact_timestamp(listed[0])


async def _write_keyed(
    store: _ArtifactStore, *, kind: str, key: str, body: str
) -> None:
    """The single writer for keyed journal artifacts (title = tags[0] = key)."""
    await store.write_artifact(kind=kind, title=key, body=body, tags=[key])


def _decode_invalidation(body: str | None) -> JournalInvalidation:
    payload: dict[str, Any] = {}
    if body is not None:
        try:
            decoded = json.loads(body)
            if isinstance(decoded, dict):
                payload = decoded
        except (json.JSONDecodeError, ValueError):
            logger.warning(
                "workflow journal: unreadable invalidation record; "
                "treating as bare invalidation"
            )
    feedback = payload.get("feedback")
    return JournalInvalidation(
        reason=str(payload.get("reason", "")),
        feedback=feedback if isinstance(feedback, str) else None,
        carry_previous=bool(payload.get("carry_previous", False)),
    )


async def write_invalidation(
    store: _ArtifactStore,
    *,
    key: str,
    reason: str,
    feedback: str | None = None,
    carry_previous: bool = False,
) -> None:
    """Append an invalidation flag for ``key`` (never mutates the entry)."""
    await _write_keyed(
        store,
        kind=_JOURNAL_INVALIDATION_KIND,
        key=key,
        body=json.dumps(
            {
                "reason": reason,
                "feedback": feedback,
                "carry_previous": carry_previous,
            },
            ensure_ascii=False,
        ),
    )


async def journal_key_exists(store: _ArtifactStore, key: str) -> bool:
    """Existence check for one key — one indexed list call, no body reads."""
    return await _newest_for_key(store, _JOURNAL_KIND, key) is not None


async def _newest_invalidation_index(
    store: _ArtifactStore,
) -> dict[str, tuple[str, float]]:
    """key → ``(artifact_id, timestamp)`` of its newest invalidation record."""
    index: dict[str, tuple[str, float]] = {}
    for item in _listed_artifacts(
        await store.list_artifacts(
            {"kind": _JOURNAL_INVALIDATION_KIND, "limit": _BULK_LIMIT}
        )
    ):  # newest first — first occurrence per key wins
        key = str(item.get("title", ""))
        artifact_id = str(item.get("id", ""))
        if key and artifact_id and key not in index:
            index[key] = (artifact_id, _artifact_timestamp(item))
    return index


async def load_journal_entries(
    store: _ArtifactStore, *, limit: int = _BULK_LIMIT
) -> list[JournalEntry]:
    """Newest record per journal key, decoded, with pending-invalidation flags."""
    invalidation_index = await _newest_invalidation_index(store)
    newest: list[tuple[str, str, float]] = []  # (key, artifact_id, ts)
    seen: set[str] = set()
    for item in _listed_artifacts(
        await store.list_artifacts({"kind": _JOURNAL_KIND, "limit": limit})
    ):  # newest first
        key = str(item.get("title", ""))
        artifact_id = str(item.get("id", ""))
        if not key or not artifact_id or key in seen:
            continue
        seen.add(key)
        newest.append((key, artifact_id, _artifact_timestamp(item)))
    bodies = await asyncio.gather(
        *(_read_full_body(store, artifact_id) for _key, artifact_id, _ts in newest)
    )
    entries: list[JournalEntry] = []
    for (key, _artifact_id, ts), body in zip(newest, bodies):
        if body is None:
            continue
        prompt, result = _decode_journal_body(body)
        invalidation = invalidation_index.get(key)
        entries.append(
            JournalEntry(
                key=key,
                result=result,
                prompt=prompt,
                timestamp=ts,
                invalidated=invalidation is not None
                and _is_pending(invalidation[1], ts),
            )
        )
    return entries


def _invalidation_rerun_prompt(
    prompt: str,
    invalidation: JournalInvalidation,
    previous_result: str | None,
) -> str:
    """Compose the execution prompt for a forced re-run.

    Execution detail only: the journal key stays the hash of the original
    script-authored prompt, so downstream key-shift propagation and resume
    addressing are unaffected by this injection."""
    lines = [
        prompt,
        "",
        "---",
        "A previous result for this exact task was found to be wrong and has "
        "been invalidated. Redo the task from the original instructions above.",
    ]
    if invalidation.reason.strip():
        lines.append(
            f"Why the previous result was wrong: {invalidation.reason.strip()}"
        )
    if invalidation.feedback and invalidation.feedback.strip():
        lines.append(f"Guidance for this attempt: {invalidation.feedback.strip()}")
    if invalidation.carry_previous and previous_result:
        lines += [
            "The invalidated previous result follows for reference only — do "
            "not repeat its mistake:",
            "<previous_attempt>",
            previous_result,
            "</previous_attempt>",
        ]
    return "\n".join(lines)


@dataclass(slots=True)
class _Journal:
    """Workflow-local journal for one run.

    On resume the host looks each key up and returns the cached body without
    re-spawning — unless a pending invalidation forces a miss, in which case
    the node re-runs with the invalidation feedback injected and its fresh
    result, recorded under the same key, supersedes the flag. Upstream
    results are interpolated into dependent prompts, so a changed re-run
    result shifts every dependent key and re-runs exactly the affected
    subgraph (reliability-substrate.md §4.3).

    *Not* ``SessionStore.open`` — that is session-level transcript replay, the
    wrong granularity (design §3.3).
    """

    store: _ArtifactStore | None
    _cache: dict[str, str] = field(default_factory=dict)
    _store_has_entries: bool = True
    # key → (artifact_id, ts) of the newest invalidation record, snapshotted
    # once per run (eagerly at prime(), lazily on first lookup otherwise) —
    # invalidations are written between runs (via the workflow_invalidate
    # tool), so per-key store probes on every lookup would buy nothing: a
    # mid-run invalidation applies from the next run. ``None`` = not yet
    # snapshotted; correctness must not depend on prime() being called.
    _invalidation_index: dict[str, tuple[str, float]] | None = None

    async def prime(self) -> None:
        """Probe the store once before the run starts.

        A fresh (non-resume) run can never hit the store — its keys don't
        exist yet, and intra-run repeats of the same ``agent()`` call are
        served from the in-memory ``_cache`` that ``record`` populates. So if
        the store holds no ``workflow_journal`` artifacts, disable per-agent
        store lookups for the whole run. (No journal entries also means no
        invalidation can apply — there is nothing recorded to invalidate.)
        Otherwise also snapshot the invalidation index for the run."""
        if self.store is None:
            self._store_has_entries = False
            return
        listing = await self.store.list_artifacts({"kind": _JOURNAL_KIND, "limit": 1})
        self._store_has_entries = bool(_listed_artifacts(listing))
        if self._store_has_entries:
            self._invalidation_index = await _newest_invalidation_index(self.store)

    @staticmethod
    def key(prompt: str, opts: dict[str, Any]) -> str:
        payload = json.dumps(
            {"prompt": prompt, "opts": opts},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]

    async def lookup_state(self, key: str) -> _JournalState:
        """Resume state for ``key``: cached result and any pending invalidation.

        The in-run ``_cache`` wins unconditionally: within one run a key is
        resolved once."""
        if key in self._cache:
            return _JournalState(cached=self._cache[key])
        if self.store is None or not self._store_has_entries:
            return _JournalState()
        if self._invalidation_index is None:
            self._invalidation_index = await _newest_invalidation_index(self.store)
        newest = await _newest_for_key(self.store, _JOURNAL_KIND, key)
        result_ts = float("-inf")
        cached: str | None = None
        if newest is not None:
            artifact_id, result_ts = newest
            body = await _read_full_body(self.store, artifact_id)
            if body is not None:
                _prompt, cached = _decode_journal_body(body)
        invalidation: JournalInvalidation | None = None
        flagged = self._invalidation_index.get(key)
        if flagged is not None and _is_pending(flagged[1], result_ts):
            invalidation = _decode_invalidation(
                await _read_full_body(self.store, flagged[0])
            )
        if cached is not None and invalidation is None:
            self._cache[key] = cached
        return _JournalState(cached=cached, invalidation=invalidation)

    async def record(self, key: str, body: str, *, prompt: str) -> None:
        self._cache[key] = body
        if self.store is None:
            return
        await _write_keyed(
            self.store,
            kind=_JOURNAL_KIND,
            key=key,
            body=_encode_journal_body(prompt, body),
        )


__all__ = (
    "JournalEntry",
    "JournalInvalidation",
    "journal_key_exists",
    "load_journal_entries",
    "write_invalidation",
)
