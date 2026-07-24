# code-health: ignore-file[AM025] -- raw rows are untyped at the boundary
"""The data plane: one queryable fact model derived from raw trajectory data.

Checklist authors and the trigger DSL never touch raw trajectories; they
query this schema. Facts are DERIVED bottom-up from what the raw layer
actually records — nothing here is wished into existence:

    raw tool events ──parse──▶ runs / run_scopes / run_selectors /
                               run_failures / edits
    raw result text ─parse──▶ repo_files / imports  (files the agent read
                               or wrote; import lines visible in the clip)
    assistant msgs  ─ingest─▶ claims
    joins (views) ──────────▶ superseded / unresolved reds / independent
                               greens / uncovered scopes

Everything lives in the same per-session SQLite file as the raw events, so
`DataPlane.query()` is the unified entry for the live watcher, offline
replay, the critic's evidence digest, and ad-hoc calibration
(`python -m policy_engine query`). Rebuild is idempotent and cheap (a
session is a few hundred raw rows); live evaluation rebuilds at each
decision point so online and offline consumers share one code path.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .commands import (
    ENV_PROBE_EXITS,
    ExecRecord,
    is_validation,
    scope_refs,
    scope_tokens,
    selector_tokens,
    split_segments,
    supersedes,
    usage_error,
)

_PLANE_SCHEMA = """
CREATE TABLE IF NOT EXISTS plane_runs (
    id INTEGER PRIMARY KEY,
    turn INTEGER NOT NULL,
    head TEXT NOT NULL,
    raw TEXT NOT NULL,
    exit_code INTEGER,
    is_validation INTEGER NOT NULL,
    is_env_probe INTEGER NOT NULL,
    is_usage_error INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS plane_run_scopes (
    run_id INTEGER NOT NULL,
    scope TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS plane_run_selectors (
    run_id INTEGER NOT NULL,
    selector TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS plane_run_tokens (
    run_id INTEGER NOT NULL,
    token TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS plane_run_test_files (
    run_id INTEGER NOT NULL,
    stem TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS plane_run_failures (
    run_id INTEGER NOT NULL,
    test_name TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS plane_edits (
    id INTEGER PRIMARY KEY,
    turn INTEGER NOT NULL,
    path TEXT NOT NULL,
    stem TEXT NOT NULL,
    is_test INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS plane_claims (
    id INTEGER PRIMARY KEY,
    turn INTEGER NOT NULL,
    is_final INTEGER NOT NULL,
    text TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS plane_repo_files (
    path TEXT PRIMARY KEY,
    is_test INTEGER NOT NULL,
    source TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS plane_imports (
    src_path TEXT NOT NULL,
    dst_token TEXT NOT NULL,
    source TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS plane_superseded (
    red_run_id INTEGER NOT NULL,
    green_run_id INTEGER NOT NULL
);
"""

_VIEWS = """
CREATE TEMP VIEW IF NOT EXISTS v_validations AS
    SELECT * FROM plane_runs WHERE is_validation = 1;

CREATE TEMP VIEW IF NOT EXISTS v_reds AS
    SELECT * FROM v_validations
    WHERE exit_code IS NOT NULL AND exit_code != 0
      AND is_env_probe = 0 AND is_usage_error = 0;

CREATE TEMP VIEW IF NOT EXISTS v_greens AS
    SELECT * FROM v_validations WHERE exit_code = 0;

CREATE TEMP VIEW IF NOT EXISTS v_unresolved_reds AS
    SELECT r.* FROM v_reds r
    WHERE NOT EXISTS (
        SELECT 1 FROM plane_superseded s WHERE s.red_run_id = r.id
    );

-- greens that exercise something beyond the agent's own test files
CREATE TEMP VIEW IF NOT EXISTS v_independent_greens AS
    SELECT g.* FROM v_greens g
    WHERE EXISTS (
            SELECT 1 FROM plane_run_test_files tf
            WHERE tf.run_id = g.id
              AND tf.stem NOT IN (SELECT stem FROM plane_edits WHERE is_test = 1)
        )
       OR (
            NOT EXISTS (SELECT 1 FROM plane_run_test_files tf
                        WHERE tf.run_id = g.id)
            AND NOT EXISTS (SELECT 1 FROM plane_run_selectors sel
                            WHERE sel.run_id = g.id)
        );

-- mutated scopes never referenced by any validation run
CREATE TEMP VIEW IF NOT EXISTS v_uncovered_scopes AS
    SELECT DISTINCT e.stem AS scope FROM plane_edits e
    WHERE e.stem NOT IN (SELECT scope FROM plane_run_scopes);
"""

# Import-line shapes across the languages in the corpus (rust/go/ts/py/ex).
# Lexical by design; rows carry source='observed' provenance.
_IMPORT_LINE = re.compile(
    r"^\s*(?:import|use|from|require|alias|include)\b(.{1,120})",
    re.MULTILINE,
)

_FAIL_NAME = re.compile(
    r"(?:FAIL(?:ED)?:?\s+|test\s+)([A-Za-z_][\w:.\-/]{2,80})(?:\s+\.\.\.\s+FAILED)?"
)


@dataclass(slots=True)
class DataPlane:
    """Facts derived from one session's raw rows; query() is the entry."""

    conn: sqlite3.Connection
    raw_schema: str = "main"

    # -- construction ----------------------------------------------------------

    @classmethod
    def open(cls, db_path: Path) -> "DataPlane":
        """Live mode: plane tables live next to the raw rows."""

        conn = sqlite3.connect(str(db_path))
        conn.executescript(_PLANE_SCHEMA)
        conn.executescript(_VIEWS)
        return cls(conn=conn)

    @classmethod
    def snapshot(cls, db_path: Path) -> "DataPlane":
        """Analysis mode: raw file attached read-only, facts in memory —
        replay over archived corpora never mutates the archives."""

        conn = sqlite3.connect(":memory:", uri=True)
        conn.execute("ATTACH DATABASE ? AS raw", (f"file:{db_path}?mode=ro",))
        conn.executescript(_PLANE_SCHEMA)
        conn.executescript(_VIEWS)
        return cls(conn=conn, raw_schema="raw")

    def fetch_raw(self) -> list[tuple]:
        return self.conn.execute(
            f"SELECT id, turn, tool_name, args_json, result_json, exit_code "  # noqa: S608
            f"FROM {self.raw_schema}.policy_tool_events "
            "WHERE phase='post' ORDER BY id"
        ).fetchall()

    def rebuild(self, rows: list[tuple] | None = None) -> None:
        """Re-derive every fact table from raw rows (or a prefix of them
        for incremental replay). Idempotent."""

        raw = rows if rows is not None else self.fetch_raw()

        for table in (
            "plane_runs",
            "plane_run_scopes",
            "plane_run_selectors",
            "plane_run_tokens",
            "plane_run_test_files",
            "plane_run_failures",
            "plane_edits",
            "plane_repo_files",
            "plane_imports",
            "plane_superseded",
        ):
            self.conn.execute(f"DELETE FROM {table}")  # noqa: S608

        edits: list[tuple[int, int, str]] = []
        bashes: list[tuple[int, int, str, int | None, str]] = []
        for row_id, turn, tool, args_json, result_json, exit_code in raw:
            try:
                args = json.loads(args_json) if args_json else {}
            except json.JSONDecodeError:
                continue
            result_error, result_text = _decode_result(result_json)
            if tool == "bash":
                cmd = args.get("cmd")
                if isinstance(cmd, str) and cmd.strip():
                    bashes.append((row_id, turn, cmd, exit_code, result_text))
            elif tool in {"write", "edit"} and not result_error:
                path = args.get("path") or args.get("file_path")
                if isinstance(path, str) and path:
                    edits.append((row_id, turn, path))
                    self._observe_file(path, source="edit")
                    content = args.get("content") or args.get("new_text") or ""
                    if isinstance(content, str) and content:
                        self._observe_imports(path, content, source="edit")
            elif tool == "read" and not result_error:
                path = args.get("path") or args.get("file_path")
                if isinstance(path, str) and path and result_text:
                    self._observe_file(path, source="read")
                    self._observe_imports(path, result_text, source="read")

        for row_id, turn, path in edits:
            self.conn.execute(
                "INSERT INTO plane_edits (id, turn, path, stem, is_test) "
                "VALUES (?, ?, ?, ?, ?)",
                (row_id, turn, path, Path(path).stem, int(_is_test_path(path))),
            )
        mutated = [path for _i, _t, path in edits]
        scopes = scope_tokens(mutated)

        records: list[tuple[int, ExecRecord]] = []
        for row_id, turn, cmd, exit_code, result_text in bashes:
            record = ExecRecord(
                raw=cmd,
                segments=split_segments(cmd),
                exit_code=exit_code,
                turn=turn,
            )
            records.append((row_id, record))
            validation_segments = [
                segment for segment in record.segments if is_validation(segment)
            ]
            head_segment = (
                validation_segments[0]
                if validation_segments
                else (record.segments[0] if record.segments else None)
            )
            if head_segment is None:
                continue
            for segment in record.segments:
                for token in dict.fromkeys(segment.norm):
                    self.conn.execute(
                        "INSERT INTO plane_run_tokens (run_id, token) VALUES (?, ?)",
                        (row_id, token.lower()),
                    )
            self.conn.execute(
                "INSERT INTO plane_runs "
                "(id, turn, head, raw, exit_code, is_validation, "
                " is_env_probe, is_usage_error) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    row_id,
                    turn,
                    head_segment.head,
                    cmd,
                    exit_code,
                    int(bool(validation_segments)),
                    int(exit_code in ENV_PROBE_EXITS if exit_code is not None else 0),
                    int(usage_error(record)),
                ),
            )
            for segment in validation_segments:
                for scope in scope_refs(segment, scopes):
                    self.conn.execute(
                        "INSERT INTO plane_run_scopes (run_id, scope) VALUES (?, ?)",
                        (row_id, scope),
                    )
                    for selector in selector_tokens(segment, scope):
                        self.conn.execute(
                            "INSERT INTO plane_run_selectors (run_id, selector) "
                            "VALUES (?, ?)",
                            (row_id, selector),
                        )
                if not scope_refs(segment, scopes):
                    for selector in selector_tokens(segment, "\x00none"):
                        self.conn.execute(
                            "INSERT INTO plane_run_selectors (run_id, selector) "
                            "VALUES (?, ?)",
                            (row_id, selector),
                        )
                for token in segment.ordered:
                    if _is_test_path(token):
                        self.conn.execute(
                            "INSERT INTO plane_run_test_files (run_id, stem) "
                            "VALUES (?, ?)",
                            (row_id, Path(token).stem),
                        )
            if exit_code not in (None, 0) and result_text:
                for name in _FAIL_NAME.findall(result_text)[:20]:
                    self.conn.execute(
                        "INSERT INTO plane_run_failures (run_id, test_name) "
                        "VALUES (?, ?)",
                        (row_id, name),
                    )

        self._derive_supersession(records, scopes)
        self.conn.commit()

    def ingest_claims(self, claims: list[tuple[int, bool, str]]) -> None:
        """Assistant messages: (turn, is_final, text). Source: live decide
        events or the trajectory store; old session DBs have none."""

        self.conn.execute("DELETE FROM plane_claims")
        self.conn.executemany(
            "INSERT INTO plane_claims (turn, is_final, text) VALUES (?, ?, ?)",
            [(turn, int(final), text) for turn, final, text in claims],
        )
        self.conn.commit()

    def _derive_supersession(
        self, records: list[tuple[int, ExecRecord]], scopes: frozenset[str]
    ) -> None:
        reds = []
        greens = []
        for row_id, record in records:
            segments = [s for s in record.segments if is_validation(s)]
            if not segments:
                continue
            if record.exit_code == 0:
                greens.append((row_id, segments))
            elif (
                record.exit_code is not None
                and record.exit_code not in ENV_PROBE_EXITS
                and not usage_error(record)
            ):
                reds.append((row_id, segments))
        for red_id, red_segments in reds:
            for green_id, green_segments in greens:
                if green_id <= red_id:
                    continue
                if all(
                    any(supersedes(g, r, scopes) for g in green_segments)
                    for r in red_segments
                ):
                    self.conn.execute(
                        "INSERT INTO plane_superseded (red_run_id, green_run_id) "
                        "VALUES (?, ?)",
                        (red_id, green_id),
                    )
                    break

    def _observe_file(self, path: str, *, source: str) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO plane_repo_files (path, is_test, source) "
            "VALUES (?, ?, ?)",
            (path, int(_is_test_path(path)), source),
        )

    def _observe_imports(self, path: str, content: str, *, source: str) -> None:
        for match in _IMPORT_LINE.findall(content[:4000])[:30]:
            self.conn.execute(
                "INSERT INTO plane_imports (src_path, dst_token, source) "
                "VALUES (?, ?, ?)",
                (path, match.strip()[:120], source),
            )

    # -- unified query entry ---------------------------------------------------

    def query(self, sql: str, params: tuple[object, ...] = ()) -> list[tuple]:
        """The single query entry. Fact tables and v_* views only."""

        return self.conn.execute(sql, params).fetchall()

    def scalar(self, sql: str, params: tuple[object, ...] = ()) -> object:
        row = self.conn.execute(sql, params).fetchone()
        return row[0] if row else None

    def close(self) -> None:
        self.conn.close()


def _decode_result(result_json: str | None) -> tuple[bool, str]:
    if not result_json:
        return False, ""
    try:
        payload = json.loads(result_json)
    except json.JSONDecodeError:
        return False, ""
    texts = [
        block.get("text", "")
        for block in payload.get("content", [])
        if isinstance(block, dict)
    ]
    return bool(payload.get("is_error")), "\n".join(t for t in texts if t)


def _is_test_path(path: str) -> bool:
    stem = Path(path).stem.lower()
    return stem.startswith(("test_", "spec_")) or stem.endswith(
        ("_test", "_spec", ".test", ".spec", "-test", "-spec")
    )
