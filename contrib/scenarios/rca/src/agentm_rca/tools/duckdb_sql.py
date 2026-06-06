"""DuckDB SQL tool for the RCA scenario.

Exposes two atoms:

* ``list_tables`` — enumerate the parquet files under ``data_dir`` and
  return their schema + row count. Each file is registered as a DuckDB
  view named after its filename without the ``.parquet`` suffix, so the
  agent can write ``SELECT ... FROM abnormal_traces`` instead of
  ``read_parquet('/abs/path/abnormal_traces.parquet')``.
* ``query_sql`` — execute a single read-only SQL statement against the
  registered views and return rows as JSON, capped by a token budget.

The atom runs in one of two modes, selected at install time; the agent's
prompt and the tool schemas are **identical** in both:

* **Local mode** (default): reads parquet from a local ``data_dir`` with an
  in-process DuckDB. Requires no network and no ``pyarrow``.
* **Remote mode**: when an ``endpoint`` is configured, ``list_tables`` and
  ``query_sql`` become thin clients over the rcabench SDK's blob query
  endpoint (``BlobApi.blob_query_bucket`` →
  ``POST /api/v2/blob/buckets/{bucket}/query``; see
  ``docs/rfc/0001-remote-data-plane-duckdb-over-http.md``). The endpoint
  must be the gateway / edge-proxy base, never a direct upstream pod. The
  SDK returns already-decoded JSON rows, which run through the same
  serialise / id-compact / truncate pipeline, so the agent sees output
  structurally identical to local mode — same rows and values; column key
  order is alphabetical, as the JSON envelope carries no SQL select order.

Configuration (local)::

    - module: agentm_rca.tools.duckdb_sql
      config:
        data_dir: /path/to/converted     # required; or AGENTM_RCA_DATA_DIR
        exclude: [conclusion.parquet]    # optional, hides ground truth
        row_limit: 200                   # optional default LIMIT
        token_limit: 5000                # optional response token cap

Configuration (remote)::

    - module: agentm_rca.tools.duckdb_sql
      config:
        endpoint: https://aegis.example:8082   # gateway base; or AGENTM_DUCKDB_ENDPOINT
        bucket: my-bucket                       # or AGENTM_DUCKDB_BUCKET
        dataset: cases/batch-01KQ.../           # S3 key prefix; or AGENTM_DUCKDB_DATASET
        # keys: [a.parquet, b.parquet]          # explicit keys instead of a prefix
        # bearer token via AGENTM_DUCKDB_TOKEN env only (never in config/logs)
        row_limit: 200
        token_limit: 5000

Remote mode needs the optional ``rcabench`` SDK
(``uv sync --extra duckdb-remote``); local mode does not import it.
"""

from __future__ import annotations

import json
import math
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb  # type: ignore[import-not-found,import-untyped]

from agentm.core.abi.messages import TextContent
from agentm.core.abi import FunctionTool, ToolResult
from agentm.extensions import ExtensionManifest
from agentm.core.abi.extension import ExtensionAPI

MANIFEST = ExtensionManifest(
    name="duckdb_sql",
    description="DuckDB SQL access to parquet files for RCA investigation.",
    registers=("tool:list_tables", "tool:query_sql"),
    config_schema={
        "type": "object",
        "properties": {
            "data_dir": {"type": "string"},
            "endpoint": {"type": "string"},
            "bucket": {"type": "string"},
            "dataset": {"type": "string"},
            "prefix": {"type": "string"},
            "keys": {"type": "array", "items": {"type": "string"}},
            "exclude": {"type": "array", "items": {"type": "string"}},
            "row_limit": {"type": "integer", "minimum": 1},
            "token_limit": {"type": "integer", "minimum": 100},
        },
        "additionalProperties": False,
    },
)

_WRITE_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|ATTACH|COPY|"
    r"PRAGMA|EXPORT|IMPORT|INSTALL|LOAD|CALL|SET)\b",
    re.IGNORECASE,
)
_DEFAULT_TOKEN_LIMIT = 5000  # rationale: fits RCA result rows in a compact tool turn.
_DEFAULT_ROW_LIMIT = 200  # rationale: enough rows for patterns without flooding context.


_ID_COLUMNS = frozenset({
    "trace_id", "span_id", "parent_span_id",
    "attr.trace_id", "attr.span_id",
})
_ID_KEEP = 12


def _serialize(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        if abs(obj) < 1e12:
            return round(obj, 4)
        return obj
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_serialize(item) for item in obj]
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    return obj


def _compact_ids(rows: list[dict[str, Any]], cols: list[str]) -> list[dict[str, Any]]:
    """Truncate long hex ID columns to save tokens."""
    id_cols = [c for c in cols if c in _ID_COLUMNS]
    if not id_cols:
        return rows
    out = []
    for row in rows:
        r = dict(row)
        for c in id_cols:
            v = r.get(c)
            if isinstance(v, str) and len(v) > _ID_KEEP:
                r[c] = v[:_ID_KEEP] + "…"
        out.append(r)
    return out


def _estimate_tokens(text: str) -> int:
    # Cheap heuristic; close enough for budget enforcement.
    return (len(text) + 2) // 3


def _truncate(payload: str, *, token_limit: int, rows: list[dict[str, Any]]) -> str:
    if _estimate_tokens(payload) <= token_limit:
        return payload
    ratio = token_limit / max(1, _estimate_tokens(payload))
    keep = max(1, int(len(rows) * ratio * 0.8))
    out: dict[str, Any] = {
        "_truncated": True,
        "_total_rows": len(rows),
        "_rows_returned": keep,
        "_suggestion": "Add WHERE filters, narrower time range, or a smaller LIMIT.",
        "rows": rows[:keep],
    }
    return json.dumps(out, ensure_ascii=False, default=str)


def _ok(text: str, *, is_error: bool = False) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)], is_error=is_error)


def _err(msg: str, **extra: Any) -> ToolResult:
    return _ok(json.dumps({"error": msg, **extra}, ensure_ascii=False), is_error=True)


def _rows_to_result(
    rows: list[dict[str, Any]], cols: list[str], *, token_limit: int
) -> ToolResult:
    """Shape result rows into the agent-facing ``ToolResult``.

    The single pipeline shared by local and remote ``query_sql`` so both
    modes emit byte-identical output: id-compaction → JSON body →
    token-budget truncation.
    """
    rows = _compact_ids(rows, cols)
    body = json.dumps(
        {"row_count": len(rows), "rows": rows},
        ensure_ascii=False,
        default=str,
    )
    return _ok(_truncate(body, token_limit=token_limit, rows=rows))


def _table_name(filename: str) -> str:
    return Path(filename).stem


def _error_hint(msg: str) -> str | None:
    """Map a DuckDB error to a one-line, actionable recovery hint.

    The dominant failure modes when an LLM drives this tool are dotted
    OTLP column names used unquoted (``attr.status_code``), guessed
    table/column names, and unqualified columns in JOINs. DuckDB already
    appends "Candidate bindings:" to binder errors; this adds the fix.
    """
    low = msg.lower()
    if "referenced column" in low or "referenced table" in low:
        return (
            'Call list_tables for exact names. Dotted OTLP columns must be '
            'double-quoted, e.g. "attr.status_code", "attr.k8s.node.name".'
        )
    if "ambiguous reference" in low:
        return "Qualify the column with its table alias (e.g. child.service_name)."
    if "does not exist" in low and "function" in low:
        return (
            "No such function. Percentiles: use p50/p90/p95/p99(col) or "
            "quantile_cont(col, 0.99). Avoid backtick identifiers; use \"...\"."
        )
    return None


# Percentile aliases agents habitually reach for (``p99(duration)``) that
# DuckDB has no scalar/aggregate function for — without these every such
# call costs a wasted "function does not exist" round-trip. Each macro
# expands to ``quantile_cont`` and works in aggregate / GROUP BY context.
_HELPER_MACROS = (
    "CREATE OR REPLACE MACRO p50(x) AS quantile_cont(x, 0.5)",
    "CREATE OR REPLACE MACRO p90(x) AS quantile_cont(x, 0.9)",
    "CREATE OR REPLACE MACRO p95(x) AS quantile_cont(x, 0.95)",
    "CREATE OR REPLACE MACRO p99(x) AS quantile_cont(x, 0.99)",
)


def _install_helper_macros(conn: duckdb.DuckDBPyConnection) -> None:
    for stmt in _HELPER_MACROS:
        try:
            conn.execute(stmt)
        except duckdb.Error:  # pragma: no cover - macro already present / engine drift
            pass


def _cap_duckdb_threads(conn: duckdb.DuckDBPyConnection) -> None:
    """Bound this connection's DuckDB worker pool.

    Each ``duckdb.connect()`` defaults its task scheduler to the host core
    count. When many of these tools run as concurrent agent subprocesses
    (e.g. an eval fan-out spawning hundreds of ``agentm`` processes), every
    connection grabbing all cores oversubscribes the box badly — the data
    here is tiny (per-case parquet), so a low cap costs no real query speed.
    Opt-in via ``AGENTM_DUCKDB_THREADS``; unset preserves DuckDB's default.
    """
    raw = os.environ.get("AGENTM_DUCKDB_THREADS")
    if not raw:
        return
    try:
        n = max(1, int(raw))
    except ValueError:
        return
    try:
        conn.execute(f"SET threads={n}")
    except duckdb.Error:  # pragma: no cover - engine drift
        pass


class _DuckDBState:
    def __init__(
        self,
        *,
        data_dir: Path,
        exclude: set[str],
        row_limit: int,
        token_limit: int,
    ) -> None:
        self.data_dir = data_dir
        self.exclude = exclude
        self.row_limit = row_limit
        self.token_limit = token_limit
        self.conn: duckdb.DuckDBPyConnection | None = None
        self.tables: list[str] = []
        self.sql_cache: dict[str, ToolResult] = {}

    def connect(self) -> duckdb.DuckDBPyConnection:
        if self.conn is not None:
            return self.conn
        conn = duckdb.connect(":memory:")
        _cap_duckdb_threads(conn)
        _install_helper_macros(conn)
        files = sorted(
            f.name
            for f in self.data_dir.iterdir()
            if f.is_file() and f.suffix == ".parquet" and f.name not in self.exclude
        )
        for fname in files:
            view = _table_name(fname)
            path = (self.data_dir / fname).as_posix().replace("'", "''")
            conn.execute(
                f"CREATE OR REPLACE VIEW {view} AS "
                f"SELECT * FROM read_parquet('{path}')"
            )
            self.tables.append(view)
        self.conn = conn
        return conn

    def describe(self) -> list[dict[str, Any]]:
        conn = self.connect()
        result: list[dict[str, Any]] = []
        for view in self.tables:
            try:
                schema = conn.execute(f"DESCRIBE {view}").fetchall()
                count = conn.execute(f"SELECT count(*) FROM {view}").fetchone()
                result.append(
                    {
                        "table": view,
                        "row_count": int(count[0]) if count else 0,
                        "columns": [
                            {"name": col[0], "type": col[1]} for col in schema
                        ],
                    }
                )
            except duckdb.Error as exc:  # noqa: PERF203
                result.append({"table": view, "error": str(exc)})
        return result


def _resolve_data_dir(config: dict[str, Any]) -> Path:
    raw = config.get("data_dir") or os.environ.get("AGENTM_RCA_DATA_DIR")
    if not raw:
        raise ValueError(
            "duckdb_sql requires config.data_dir or AGENTM_RCA_DATA_DIR env var"
        )
    path = Path(str(raw)).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"data_dir is not a directory: {path}")
    return path


def _validate_sql(sql_raw: str) -> tuple[str, str] | ToolResult:
    """Apply the client-side read-only guards shared by both modes.

    Returns ``(normalised_sql, leading_keyword)`` on success or a
    ``ToolResult`` error to return verbatim. The keyword denylist/allowlist
    is UX-shaping, not the security boundary — the server enforces read-only
    independently (defense in depth).
    """
    sql_raw = sql_raw.strip()
    if not sql_raw:
        return _err("sql is required")
    sql = sql_raw.rstrip(";").strip()
    if ";" in sql:
        return _err("only one statement per call (no ';' inside SQL)")
    if _WRITE_KEYWORDS.search(sql):
        return _err("only read-only SELECT/WITH/EXPLAIN/DESCRIBE statements are allowed")
    head = sql.lstrip().split(None, 1)[0].upper() if sql.lstrip() else ""
    if head not in {"SELECT", "WITH", "EXPLAIN", "DESCRIBE", "SHOW", "SUMMARIZE"}:
        return _err(
            f"unsupported leading keyword: {head!r}; "
            "use SELECT/WITH/EXPLAIN/DESCRIBE/SHOW/SUMMARIZE"
        )
    return sql, head


def _duplicate_result(sql: str) -> ToolResult:
    return _ok(
        json.dumps(
            {
                "_duplicate": True,
                "_hint": "You already ran this exact query earlier in this "
                "session and got the same result. Re-read that result from "
                "your context instead of re-querying. Try a DIFFERENT query "
                "or move on to submit your report.",
                "sql": sql,
            },
            ensure_ascii=False,
        ),
        is_error=True,
    )


def _wrap_with_limit(sql: str, head: str, row_limit: int) -> str:
    if head in {"EXPLAIN", "DESCRIBE", "SHOW", "SUMMARIZE"}:
        return sql
    return f"SELECT * FROM ({sql}) LIMIT {row_limit}"


# ---------------------------------------------------------------------------
# Remote mode — HTTP client against the aegis blob query endpoint.
# ---------------------------------------------------------------------------

# A bearer token (``AGENTM_DUCKDB_TOKEN``) and any presigned URLs are
# secrets that must never reach the agent, tool results, error strings, or
# the observability trace. ``_REDACTED`` is what the agent sees where a
# remote URL would otherwise appear.
_REDACTED = "<remote>"


def _scrub(text: str, secrets: tuple[str, ...]) -> str:
    """Strip known secrets from any string headed for an agent/log."""
    out = text
    for s in secrets:
        if s:
            out = out.replace(s, _REDACTED)
    return out


class _RemoteState:
    """SDK client that targets the aegis blob query endpoint.

    Holds the wire secrets (endpoint, bucket auth context, bearer token)
    and the per-session dedup cache, and lazily builds the rcabench
    ``BlobApi``. The atom never surfaces the endpoint or token; ``handle``
    is the only locator the agent sees and it carries no auth material.
    """

    def __init__(
        self,
        *,
        endpoint: str,
        bucket: str,
        prefix: str | None,
        keys: list[str] | None,
        token: str | None,
        row_limit: int,
        token_limit: int,
    ) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._bucket = bucket
        self._prefix = prefix
        self._keys = keys
        self._token = token
        self.row_limit = row_limit
        self.token_limit = token_limit
        self.sql_cache: dict[str, ToolResult] = {}
        self._blob: Any = None

    @property
    def handle(self) -> str:
        """Agent-facing, secret-free locator for the dataset."""
        loc = self._prefix if self._prefix is not None else f"keys[{len(self._keys or [])}]"
        return f"blob://{self._bucket}/{loc}"

    @property
    def _secrets(self) -> tuple[str, ...]:
        return (self._endpoint, self._token or "")

    def _selector(self) -> dict[str, Any]:
        if self._keys is not None:
            return {"keys": self._keys}
        return {"prefix": self._prefix or ""}

    def blob(self) -> Any:
        """Lazily build the rcabench ``BlobApi`` client (imported only here).

        Keeps local mode free of the rcabench SDK: the import happens on the
        first remote query, never at module load. TLS verification follows
        ``AGENTM_DUCKDB_TLS_VERIFY`` (set ``0`` for a self-signed endpoint).
        """
        if self._blob is not None:
            return self._blob
        try:
            from rcabench.openapi.api.blob_api import (  # type: ignore[import-not-found,import-untyped]
                BlobApi,
            )
            from rcabench.openapi.api_client import (  # type: ignore[import-not-found,import-untyped]
                ApiClient,
            )
            from rcabench.openapi.configuration import (  # type: ignore[import-not-found,import-untyped]
                Configuration,
            )
        except ImportError as exc:  # pragma: no cover - exercised via message, not transport
            raise RuntimeError(
                "remote duckdb_sql needs the 'rcabench' SDK; "
                "install with: uv sync --extra duckdb-remote"
            ) from exc
        cfg = Configuration(host=self._endpoint)
        if self._token:
            cfg.api_key = {"BearerAuth": self._token}
            cfg.api_key_prefix = {"BearerAuth": "Bearer"}
        cfg.verify_ssl = os.environ.get("AGENTM_DUCKDB_TLS_VERIFY", "1") not in (
            "0",
            "false",
            "no",
        )
        self._blob = BlobApi(ApiClient(cfg))
        return self._blob

    def scrub(self, text: str) -> str:
        return _scrub(text, self._secrets)


def _resolve_remote(config: dict[str, Any]) -> _RemoteState | None:
    """Build a ``_RemoteState`` iff an endpoint is configured, else ``None``.

    ``None`` means local mode — the caller falls back to ``data_dir``
    resolution exactly as before.
    """
    endpoint = config.get("endpoint") or os.environ.get("AGENTM_DUCKDB_ENDPOINT")
    if not endpoint:
        return None
    bucket = config.get("bucket") or os.environ.get("AGENTM_DUCKDB_BUCKET")
    if not bucket:
        raise ValueError(
            "remote duckdb_sql requires config.bucket or AGENTM_DUCKDB_BUCKET"
        )
    keys_raw = config.get("keys")
    keys = [str(k) for k in keys_raw] if keys_raw else None
    prefix = (
        config.get("prefix")
        or config.get("dataset")
        or os.environ.get("AGENTM_DUCKDB_DATASET")
    )
    if keys is None and prefix is None:
        raise ValueError(
            "remote duckdb_sql requires a dataset/prefix (config.dataset, "
            "config.prefix, AGENTM_DUCKDB_DATASET) or explicit config.keys"
        )
    return _RemoteState(
        endpoint=str(endpoint),
        bucket=str(bucket),
        prefix=str(prefix) if prefix is not None else None,
        keys=keys,
        token=os.environ.get("AGENTM_DUCKDB_TOKEN"),
        row_limit=int(config.get("row_limit") or _DEFAULT_ROW_LIMIT),
        token_limit=int(config.get("token_limit") or _DEFAULT_TOKEN_LIMIT),
    )


def _build_local_handlers(
    config: dict[str, Any],
) -> tuple[Any, Any]:
    state = _DuckDBState(
        data_dir=_resolve_data_dir(config),
        exclude=set(config.get("exclude") or []),
        row_limit=int(config.get("row_limit") or _DEFAULT_ROW_LIMIT),
        token_limit=int(config.get("token_limit") or _DEFAULT_TOKEN_LIMIT),
    )

    async def _list(_: dict[str, Any]) -> ToolResult:
        try:
            tables = state.describe()
        except (duckdb.Error, OSError) as exc:
            return _err(f"list_tables failed: {exc}")
        payload = json.dumps(
            {"data_dir": str(state.data_dir), "tables": tables},
            ensure_ascii=False,
            default=str,
            indent=2,
        )
        return _ok(payload)

    async def _query(args: dict[str, Any]) -> ToolResult:
        validated = _validate_sql(str(args.get("sql", "")))
        if isinstance(validated, ToolResult):
            return validated
        sql, head = validated

        cache_key = " ".join(sql.split())
        cached = state.sql_cache.get(cache_key)
        if cached is not None:
            return _duplicate_result(sql)

        wrapped = _wrap_with_limit(sql, head, state.row_limit)
        try:
            conn = state.connect()
            cur = conn.execute(wrapped)
            cols = [d[0] for d in cur.description]
            rows = [
                dict(zip(cols, _serialize(list(r)), strict=False))
                for r in cur.fetchall()
            ]
        except duckdb.Error as exc:
            hint = _error_hint(str(exc))
            extra: dict[str, Any] = {"sql": sql}
            if hint:
                extra["hint"] = hint
            return _err(f"query failed: {exc}", **extra)

        result = _rows_to_result(rows, cols, token_limit=state.token_limit)
        state.sql_cache[cache_key] = result
        return result

    return _list, _query


# list_tables in remote mode is just a query: the schema IS reachable via
# SQL, so there is no separate /schema endpoint. list_tables runs this
# discovery query over the per-request views through /query and reshapes the
# flat rows into the local-mode {tables:[...]} payload.
_SCHEMA_DISCOVERY_SQL = (
    "SELECT table_name, column_name, data_type "
    "FROM information_schema.columns "
    "ORDER BY table_name, ordinal_position"
)


def _invoke_blob_query(state: _RemoteState, request_body: dict[str, Any]) -> Any:
    """The single SDK call seam (monkeypatched in tests).

    Returns the SDK's ``GenericResponseAny`` (``.data`` carries
    ``{row_count, rows}``); raises ``ApiException`` on a non-2xx response or
    a transport error on connection failure.
    """
    return state.blob().blob_query_bucket(bucket=state._bucket, request_body=request_body)


def _remote_error(exc: Exception) -> tuple[str, str | None]:
    """Extract an agent-facing message (+ optional hint) from a query failure.

    The rcabench SDK raises ``ApiException`` with ``.body`` carrying the
    server's ``{code, message}`` envelope (a DuckDB binder error for bad SQL)
    and ``.status`` the HTTP code; transport failures surface as plain
    exceptions. Duck-typed on ``.body`` / ``.status`` so this module never
    imports rcabench at scope.
    """
    body = getattr(exc, "body", None)
    status = getattr(exc, "status", None)
    if isinstance(body, bytes | bytearray):
        body = body.decode("utf-8", errors="replace")
    msg: str | None = None
    if isinstance(body, str) and body:
        try:
            msg = json.loads(body).get("message")
        except (ValueError, AttributeError):
            msg = body
    if not msg:
        msg = f"HTTP {status}: {exc}" if status else str(exc)
    msg = msg[:500]
    return msg, _error_hint(msg)


def _post_query(
    state: _RemoteState, sql: str, *, err_sql: str | None = None
) -> tuple[list[dict[str, Any]], list[str]] | ToolResult:
    """Run one SQL through the SDK and return ``(rows, columns)``.

    The single network seam shared by query_sql and list_tables — there is
    one server endpoint, /query. On any failure returns a scrubbed
    ``ToolResult`` error; for a SQL/binder error it also attaches the same
    recovery hint local mode surfaces.
    """
    request_body = {**state._selector(), "sql": sql}
    try:
        resp = _invoke_blob_query(state, request_body)
    except Exception as exc:  # noqa: BLE001 - server/transport error → tool error
        msg, hint = _remote_error(exc)
        extra: dict[str, Any] = {"sql": err_sql} if err_sql else {}
        if hint:
            extra["hint"] = hint
        return _err(f"query failed: {state.scrub(msg)}", **extra)
    data = getattr(resp, "data", None) or {}
    rows = data.get("rows") or []
    cols = list(rows[0].keys()) if rows else []
    return rows, cols


# Remote mode deliberately omits the local-mode hardening: the aegis
# ``/query`` server pre-registers the same p50/p90/p95/p99 percentile macros
# (so the tool-description promise that they are "predefined" holds in both
# modes) and bounds its own DuckDB worker pool centrally, so the per-process
# ``AGENTM_DUCKDB_THREADS`` cap is a no-op here — the agent carries zero
# DuckDB footprint.
def _build_remote_handlers(state: _RemoteState) -> tuple[Any, Any]:
    async def _list(_: dict[str, Any]) -> ToolResult:
        fetched = _post_query(state, _SCHEMA_DISCOVERY_SQL)
        if isinstance(fetched, ToolResult):
            return fetched
        rows, _cols = fetched
        # Flat (table_name, column_name, data_type) → grouped tables; order
        # preserved (server already ORDER BY table_name, ordinal_position).
        tables: dict[str, list[dict[str, Any]]] = {}
        order: list[str] = []
        for r in rows:
            name = r.get("table_name")
            if name is None:
                continue
            if name not in tables:
                tables[name] = []
                order.append(name)
            tables[name].append(
                {"name": r.get("column_name"), "type": r.get("data_type")}
            )
        payload = json.dumps(
            {
                "data_dir": state.handle,
                "tables": [{"table": t, "columns": tables[t]} for t in order],
            },
            ensure_ascii=False,
            default=str,
            indent=2,
        )
        return _ok(payload)

    async def _query(args: dict[str, Any]) -> ToolResult:
        validated = _validate_sql(str(args.get("sql", "")))
        if isinstance(validated, ToolResult):
            return validated
        sql, head = validated

        cache_key = " ".join(sql.split())
        cached = state.sql_cache.get(cache_key)
        if cached is not None:
            return _duplicate_result(sql)

        fetched = _post_query(
            state, _wrap_with_limit(sql, head, state.row_limit), err_sql=sql
        )
        if isinstance(fetched, ToolResult):
            return fetched
        rows, cols = fetched
        rows = [_serialize(r) for r in rows]
        result = _rows_to_result(rows, cols, token_limit=state.token_limit)
        state.sql_cache[cache_key] = result
        return result

    return _list, _query


def install(api: ExtensionAPI, config: dict[str, Any]) -> None:
    remote = _resolve_remote(config)
    if remote is not None:
        _list, _query = _build_remote_handlers(remote)
    else:
        _list, _query = _build_local_handlers(config)

    api.register_tool(
        FunctionTool(
            name="list_tables",
            description=(
                "List the parquet files registered as DuckDB views, with row "
                "counts and column schemas. Call this first to discover what "
                "data is available."
            ),
            parameters={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            fn=_list,
        )
    )
    api.register_tool(
        FunctionTool(
            name="query_sql",
            description=(
                "Run a single read-only DuckDB SQL statement (SELECT / WITH / "
                "EXPLAIN / DESCRIBE / SHOW / SUMMARIZE) against the parquet "
                "views. SELECTs are auto-wrapped with a LIMIT; results are "
                "JSON-serialised and capped by a token budget. "
                'Double-quote dotted OTLP columns ("attr.status_code", '
                '"attr.k8s.node.name"). Percentile helpers p50/p90/p95/p99(col) '
                "are predefined (aliases for quantile_cont)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "DuckDB SQL statement"},
                },
                "required": ["sql"],
                "additionalProperties": False,
            },
            fn=_query,
        )
    )


__all__ = ["MANIFEST", "install"]
