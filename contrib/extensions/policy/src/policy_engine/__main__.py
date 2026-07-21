# code-health: ignore-file[AM025] -- CLI decodes untyped policy files and rows
"""Policy engine CLI — query policy data, lint rules, view stats.

Run from anywhere with uv:
    uv run python -m policy_engine stats
    uv run python -m policy_engine log --rule stuck-loop
    uv run python -m policy_engine backfill --session abc123
    uv run python -m policy_engine view --session abc123
    uv run python -m policy_engine events --session abc123
    uv run python -m policy_engine state files --session abc123
    uv run python -m policy_engine rules [file.yaml]
    uv run python -m policy_engine lint <file.yaml>
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Protocol

import typer
from rich.console import Console
from rich.table import Table
from sqlalchemy.engine import Connection, Engine, RowMapping

from agentm.core.abi.query import SessionFilter, SessionIdentity
from agentm.core.abi.store import TrajectoryStore
from agentm.core.abi.trajectory import Turn, TurnCheckpoint
from agentm.storage.sql import create_sqlite_engine

from .paths import resolve_policy_db_path

app = typer.Typer(
    name="policy",
    add_completion=False,
    help="Policy engine data queries and rule management.",
)
ifg_app = typer.Typer(
    name="ifg",
    add_completion=False,
    help="Information Flow Graph extraction and queries.",
)
app.add_typer(ifg_app, name="ifg")
console = Console()


class _SessionQueryable(Protocol):
    def sessions(self) -> Iterable[SessionIdentity]: ...


class _TrajectoryStoreQueryAdapter:
    """CLI-local query view over a TrajectoryStore without runtime imports."""

    __slots__ = ("_store",)

    def __init__(self, store: TrajectoryStore) -> None:
        self._store = store

    def sessions(
        self,
        filter: SessionFilter | None = None,
    ) -> Iterable[SessionIdentity]:
        rows = [SessionIdentity.from_meta(meta) for meta in self._store.list_sessions()]
        if filter is None:
            return rows
        rows = [row for row in rows if _session_matches(row, filter)]
        if filter.limit is not None:
            rows = rows[: filter.limit]
        return rows

    def turns(self, session_id: str) -> Iterable[Turn]:
        _, turns = self._store.load(session_id)
        return turns

    def checkpoints(self, session_id: str) -> Iterable[TurnCheckpoint]:
        checkpoint = self._store.load_checkpoint(session_id)
        return () if checkpoint is None else (checkpoint,)


def _session_matches(row: SessionIdentity, filter: SessionFilter) -> bool:
    if filter.session_id is not None and row.id != filter.session_id:
        return False
    if (
        filter.parent_session_id is not None
        and row.parent_session_id != filter.parent_session_id
    ):
        return False
    if (
        filter.root_session_id is not None
        and (row.root_session_id or row.id) != filter.root_session_id
    ):
        return False
    if filter.purpose is not None and row.purpose != filter.purpose:
        return False
    if filter.since is not None and row.created_at < filter.since:
        return False
    if filter.until is not None and row.created_at > filter.until:
        return False
    return True


def _db_path(*, session_id: str | None = None, cwd: Path | None = None) -> Path:
    return resolve_policy_db_path(session_id=session_id, cwd=cwd)


def _connect(*, session_id: str | None = None) -> Engine | None:
    path = _db_path(session_id=session_id)
    if not path.exists():
        console.print(f"[yellow]No policy database at {path}[/yellow]")
        return None
    return create_sqlite_engine(path)


def _rows(
    conn: Connection,
    sql: str,
    params: Sequence[object] = (),
) -> list[RowMapping]:
    return list(conn.exec_driver_sql(sql, tuple(params)).mappings().all())


def _row(
    conn: Connection,
    sql: str,
    params: Sequence[object] = (),
) -> RowMapping | None:
    return conn.exec_driver_sql(sql, tuple(params)).mappings().fetchone()


def _table_exists(conn: Connection, table: str) -> bool:
    row = _row(
        conn,
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    )
    return row is not None


def _has_policy_tool_events(conn: Connection, session: str) -> bool:
    if not _table_exists(conn, "policy_tool_events"):
        return False
    row = _row(
        conn,
        "SELECT 1 FROM policy_tool_events WHERE session_id = ? LIMIT 1",
        (session,),
    )
    return row is not None


def _short_json(raw: str | None, *, max_chars: int = 80) -> str:
    if not raw:
        return ""
    try:
        value = json.loads(raw)
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, json.JSONDecodeError):
        text = str(raw)
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3]}..."


@app.command()
def stats(
    session: str | None = typer.Option(
        None, "--session", "-s", help="Filter by session ID"
    ),
    days: int = typer.Option(7, "--days", "-d", help="Look back N days"),
) -> None:
    """Show per-rule firing statistics."""
    engine = _connect(session_id=session)
    if not engine:
        return

    cutoff = time.time() - (days * 86400)
    conditions = ["ts > ?"]
    params: list[float | str] = [cutoff]
    if session:
        conditions.append("session_id = ?")
        params.append(session)

    sql = f"""
        SELECT rule_id, mode, effect, COUNT(*) as count,
               MIN(ts) as first_ts, MAX(ts) as last_ts
        FROM event_log
        WHERE {" AND ".join(conditions)}
        GROUP BY rule_id, mode, effect
        ORDER BY count DESC
    """
    try:
        with engine.connect() as conn:
            rows = _rows(conn, sql, params)
    finally:
        engine.dispose()

    if not rows:
        console.print("[dim]No rule firings in the specified window.[/dim]")
        return

    table = Table(title=f"Policy Stats (last {days}d)")
    table.add_column("Rule", style="cyan")
    table.add_column("Mode", style="dim")
    table.add_column("Effect", style="bold")
    table.add_column("Count", justify="right")
    table.add_column("Last Fired")

    for row in rows:
        last = time.strftime("%Y-%m-%d %H:%M", time.localtime(row["last_ts"]))
        table.add_row(
            row["rule_id"],
            row["mode"],
            row["effect"],
            str(row["count"]),
            last,
        )

    console.print(table)


@app.command()
def log(
    session: str | None = typer.Option(
        None, "--session", "-s", help="Filter by session ID"
    ),
    rule: str | None = typer.Option(None, "--rule", "-r", help="Filter by rule name"),
    limit: int = typer.Option(20, "--limit", "-n", help="Max rows"),
) -> None:
    """Show recent effect_log entries."""
    engine = _connect(session_id=session)
    if not engine:
        return

    conditions: list[str] = []
    params: list[str] = []
    if session:
        conditions.append("session_id = ?")
        params.append(session)
    if rule:
        conditions.append("rule_id = ?")
        params.append(rule)

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    sql = f"SELECT * FROM event_log {where} ORDER BY ts DESC LIMIT ?"
    try:
        with engine.connect() as conn:
            rows = _rows(conn, sql, [*params, limit])
    finally:
        engine.dispose()

    if not rows:
        console.print("[dim]No entries found.[/dim]")
        return

    table = Table(title="Effect Log")
    table.add_column("Time", style="dim")
    table.add_column("Session", style="dim", max_width=12)
    table.add_column("Rule", style="cyan")
    table.add_column("Mode")
    table.add_column("Effect", style="bold")
    table.add_column("Turn", justify="right")
    table.add_column("Reason", max_width=50)

    for row in rows:
        ts = time.strftime("%m-%d %H:%M:%S", time.localtime(row["ts"]))
        sid = row["session_id"][:12] if row["session_id"] else ""
        table.add_row(
            ts,
            sid,
            row["rule_id"] or "",
            row["mode"] or "",
            row["effect"] or "",
            str(row["turn"]) if row["turn"] is not None else "",
            (row["reason"] or "")[:50],
        )

    console.print(table)


@app.command()
def backfill(
    session: str = typer.Option(..., "--session", "-s", help="Session ID to project"),
    policy_file: list[str] = typer.Option(
        [],
        "--policy-file",
        "-p",
        help="Additional policy YAML file. May be passed multiple times.",
    ),
    cwd: Path | None = typer.Option(
        None,
        "--cwd",
        help="Project cwd for resolving trajectory store and relative policies.",
    ),
    db_path: Path | None = typer.Option(
        None,
        "--db-path",
        help="Policy SQLite db path. Defaults to $AGENTM_HOME/policy_state/policy.db.",
    ),
    replace: bool = typer.Option(
        True,
        "--replace/--append",
        help="Delete existing persisted policy rows for the session before projection.",
    ),
    include_checkpoint: bool = typer.Option(
        True,
        "--include-checkpoint/--committed-only",
        help="Also project the latest incomplete checkpoint when present.",
    ),
) -> None:
    """Recompute persisted policy state from an existing trajectory session."""
    from agentm.storage.trajectory.resolve import resolve_trajectory_store

    from .loader import load_policy_bundle
    from .persistence import PolicyPersistence
    from .projection import events_from_turns, project_events

    resolved = resolve_trajectory_store(str(cwd) if cwd else None)
    if resolved is None:
        console.print(
            "[red]No trajectory store found.[/red]\n"
            "[dim]Run from a project with agentm.toml, or set "
            "AGENTM_TRAJECTORY_DSN / AGENTM_TRAJECTORY_DIR.[/dim]"
        )
        raise typer.Exit(1)

    resolved_db_path = db_path or _db_path(session_id=session, cwd=cwd)
    persistence = PolicyPersistence(resolved_db_path)
    try:
        persistence.open()
        try:
            meta, turns = resolved.store.load(session)
        except KeyError:
            console.print(f"[red]Session not found: {session}[/red]")
            raise typer.Exit(1)
        checkpoint = (
            resolved.store.load_checkpoint(session) if include_checkpoint else None
        )
        trajectory_records: list[Turn | TurnCheckpoint] = [*turns]
        if checkpoint is not None:
            trajectory_records.append(checkpoint)

        policy_cwd = Path(meta.cwd) if meta.cwd else (cwd or Path.cwd())
        bundle = load_policy_bundle(policy_file, cwd=policy_cwd)

        deleted = persistence.delete_session(session) if replace else 0
        result = project_events(
            session_id=session,
            events=events_from_turns(trajectory_records),
            rules=bundle.rules,
            ifc=bundle.ifc,
            persistence=persistence,
        )
    finally:
        persistence.close()
        resolved.close()

    console.print(
        "[green]Backfilled policy projection[/green] "
        f"session={result.session_id} turns={result.turns} "
        f"committed={len(turns)} checkpoint={1 if checkpoint is not None else 0} "
        f"tool_calls={result.tool_calls} effects={result.effects} "
        f"eval_errors={result.eval_errors} deleted={deleted} db={resolved_db_path}"
    )


@app.command()
def view(
    session: str | None = typer.Option(
        None, "--session", "-s", help="Session ID to view"
    ),
    cwd: Path | None = typer.Option(
        None,
        "--cwd",
        help="Project cwd for resolving the trajectory store.",
    ),
    db_path: Path | None = typer.Option(
        None,
        "--db-path",
        help="Policy SQLite db path. Defaults to $AGENTM_HOME/policy_state/policy.db.",
    ),
    follow: bool = typer.Option(False, "--follow", "-f", help="Refresh live data"),
) -> None:
    """Open AgentM's shared trace TUI with policy projection tabs."""
    if not sys.stdout.isatty():
        console.print("[red]error: interactive viewer requires a terminal[/red]")
        raise typer.Exit(2)

    from agentm.storage.trajectory.resolve import resolve_trajectory_store

    from .trace_view import run_policy_trace_viewer

    resolved = resolve_trajectory_store(str(cwd) if cwd else None)
    if resolved is None:
        console.print(
            "[red]No trajectory store found.[/red]\n"
            "[dim]Run from a project with agentm.toml, or set "
            "AGENTM_TRAJECTORY_DSN / AGENTM_TRAJECTORY_DIR.[/dim]"
        )
        raise typer.Exit(1)

    try:
        query = _TrajectoryStoreQueryAdapter(resolved.store)
        sid = _resolve_view_session_id(query, session)
        run_policy_trace_viewer(
            query,
            sid,
            db_path=db_path,
            cwd=cwd or Path.cwd(),
            follow=follow,
        )
    finally:
        resolved.close()


def _resolve_view_session_id(query: _SessionQueryable, session: str | None) -> str:
    if session:
        return session
    sessions = list(query.sessions())
    if not sessions:
        console.print("[red]No sessions in trajectory store.[/red]")
        raise typer.Exit(1)
    return max(sessions, key=lambda item: item.created_at).id


@app.command()
def events(
    session: str | None = typer.Option(
        None, "--session", "-s", help="Filter by session ID"
    ),
    tool: str | None = typer.Option(None, "--tool", "-t", help="Filter by tool name"),
    limit: int = typer.Option(20, "--limit", "-n", help="Max rows"),
) -> None:
    """Show persisted policy-observed tool events."""
    engine = _connect(session_id=session)
    if not engine:
        return
    try:
        with engine.connect() as conn:
            if not _table_exists(conn, "policy_tool_events"):
                console.print("[dim]No policy_tool_events table found.[/dim]")
                return

            conditions: list[str] = []
            params: list[str] = []
            if session:
                conditions.append("session_id = ?")
                params.append(session)
            if tool:
                conditions.append("tool_name = ?")
                params.append(tool)

            where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
            sql = f"SELECT * FROM policy_tool_events {where} ORDER BY ts DESC LIMIT ?"
            rows = _rows(conn, sql, [*params, limit])
    finally:
        engine.dispose()

    if not rows:
        console.print("[dim]No tool events found.[/dim]")
        return

    table = Table(title="Policy Tool Events")
    table.add_column("Time", style="dim")
    table.add_column("Session", style="dim", max_width=12)
    table.add_column("Turn", justify="right")
    table.add_column("Phase")
    table.add_column("Tool", style="cyan")
    table.add_column("Call ID", style="dim", max_width=12)
    table.add_column("Args", max_width=40)
    table.add_column("Result", max_width=40)
    table.add_column("State", max_width=40)

    for row in rows:
        ts = time.strftime("%m-%d %H:%M:%S", time.localtime(row["ts"]))
        table.add_row(
            ts,
            row["session_id"][:12] if row["session_id"] else "",
            str(row["turn"]),
            row["phase"],
            row["tool_name"],
            (row["tool_call_id"] or "")[:12],
            _short_json(row["args_json"], max_chars=60),
            _short_json(row["result_json"], max_chars=60),
            _short_json(row["state_json"], max_chars=60),
        )

    console.print(table)


@app.command()
def state(
    kind: str = typer.Argument(
        "files",
        help="State table to show: files, entities, context, turns, errors",
    ),
    session: str | None = typer.Option(
        None, "--session", "-s", help="Filter by session ID"
    ),
    limit: int = typer.Option(20, "--limit", "-n", help="Max rows"),
) -> None:
    """Show persisted policy intermediate state snapshots."""
    engine = _connect(session_id=session)
    if not engine:
        return

    try:
        with engine.connect() as conn:
            normalized = kind.lower()
            if normalized == "files":
                _show_file_state(conn, session, limit)
            elif normalized == "entities":
                _show_entity_state(conn, session, limit)
            elif normalized == "context":
                _show_json_state(
                    conn,
                    table_name="policy_context_state",
                    title="Policy Context State",
                    json_column="context_json",
                    session=session,
                    limit=limit,
                )
            elif normalized in ("turns", "turn_summary"):
                _show_json_state(
                    conn,
                    table_name="policy_turn_summary",
                    title="Policy Turn Summary",
                    json_column="summary_json",
                    session=session,
                    limit=limit,
                )
            elif normalized in ("errors", "eval_errors"):
                _show_eval_errors(conn, session, limit)
            else:
                console.print(
                    "[red]Unknown state kind. Use one of: files, entities, context, turns, errors[/red]"
                )
                raise typer.Exit(1)
    finally:
        engine.dispose()


def _session_where(
    session: str | None,
    *,
    order_by: str,
    limit: int,
) -> tuple[str, list[str | int]]:
    conditions: list[str] = []
    params: list[str | int] = []
    if session:
        conditions.append("session_id = ?")
        params.append(session)
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    return f"{where} ORDER BY {order_by} DESC LIMIT ?", [*params, limit]


def _show_file_state(
    conn: Connection,
    session: str | None,
    limit: int,
) -> None:
    if not _table_exists(conn, "policy_file_state"):
        console.print("[dim]No policy_file_state table found.[/dim]")
        return

    where, params = _session_where(session, order_by="updated_at", limit=limit)
    rows = _rows(conn, f"SELECT * FROM policy_file_state {where}", params)
    if not rows:
        console.print("[dim]No file state found.[/dim]")
        return

    table = Table(title="Policy File State")
    table.add_column("Session", style="dim", max_width=12)
    table.add_column("Path", style="cyan", max_width=48)
    table.add_column("Reads", justify="right")
    table.add_column("Writes", justify="right")
    table.add_column("First Read", justify="right")
    table.add_column("Last Read", justify="right")
    table.add_column("Last Write", justify="right")
    table.add_column("Hash", style="dim", max_width=12)

    for row in rows:
        table.add_row(
            row["session_id"][:12],
            row["path"],
            str(row["read_count"]),
            str(row["write_count"]),
            "" if row["first_read_turn"] is None else str(row["first_read_turn"]),
            "" if row["last_read_turn"] is None else str(row["last_read_turn"]),
            "" if row["last_write_turn"] is None else str(row["last_write_turn"]),
            row["content_hash"] or "",
        )
    console.print(table)


def _show_entity_state(
    conn: Connection,
    session: str | None,
    limit: int,
) -> None:
    if not _table_exists(conn, "policy_entity_state"):
        console.print("[dim]No policy_entity_state table found.[/dim]")
        return

    where, params = _session_where(session, order_by="updated_at", limit=limit)
    rows = _rows(conn, f"SELECT * FROM policy_entity_state {where}", params)
    if not rows:
        console.print("[dim]No entity state found.[/dim]")
        return

    table = Table(title="Policy Entity State")
    table.add_column("Session", style="dim", max_width=12)
    table.add_column("Entity", style="cyan", max_width=42)
    table.add_column("Type")
    table.add_column("First", justify="right")
    table.add_column("Last", justify="right")
    table.add_column("Count", justify="right")
    table.add_column("Evidence", max_width=60)

    for row in rows:
        table.add_row(
            row["session_id"][:12],
            row["entity"],
            row["entity_type"],
            str(row["first_seen_turn"]),
            str(row["last_seen_turn"]),
            str(row["occurrence_count"]),
            _short_json(row["evidence_json"], max_chars=80),
        )
    console.print(table)


def _show_json_state(
    conn: Connection,
    *,
    table_name: str,
    title: str,
    json_column: str,
    session: str | None,
    limit: int,
) -> None:
    if not _table_exists(conn, table_name):
        console.print(f"[dim]No {table_name} table found.[/dim]")
        return

    where, params = _session_where(session, order_by="updated_at", limit=limit)
    rows = _rows(conn, f"SELECT * FROM {table_name} {where}", params)
    if not rows:
        console.print("[dim]No state rows found.[/dim]")
        return

    table = Table(title=title)
    table.add_column("Session", style="dim", max_width=12)
    table.add_column("Turn", justify="right")
    table.add_column("State", max_width=100)

    for row in rows:
        table.add_row(
            row["session_id"][:12],
            str(row["turn_index"]),
            _short_json(row[json_column], max_chars=120),
        )
    console.print(table)


def _show_eval_errors(
    conn: Connection,
    session: str | None,
    limit: int,
) -> None:
    if not _table_exists(conn, "policy_eval_error"):
        console.print("[dim]No policy_eval_error table found.[/dim]")
        return

    where, params = _session_where(session, order_by="ts", limit=limit)
    rows = _rows(conn, f"SELECT * FROM policy_eval_error {where}", params)
    if not rows:
        console.print("[dim]No eval errors found.[/dim]")
        return

    table = Table(title="Policy Eval Errors")
    table.add_column("Time", style="dim")
    table.add_column("Session", style="dim", max_width=12)
    table.add_column("Turn", justify="right")
    table.add_column("Rule", style="cyan")
    table.add_column("Channel")
    table.add_column("Tool")
    table.add_column("Error", max_width=80)

    for row in rows:
        ts = time.strftime("%m-%d %H:%M:%S", time.localtime(row["ts"]))
        table.add_row(
            ts,
            row["session_id"][:12],
            str(row["turn"]),
            row["rule_id"],
            row["channel"],
            row["tool_name"] or "",
            row["error"],
        )
    console.print(table)


@app.command()
def rules(
    file: Path = typer.Argument(
        None, help="Policy YAML file to inspect (default: base_policy.yaml)"
    ),
) -> None:
    """List compiled rules from a policy file."""
    from .compiler import compile_policy_file

    if file is None:
        file = Path(__file__).parent / "base_policy.yaml"

    if not file.exists():
        console.print(f"[red]File not found: {file}[/red]")
        raise typer.Exit(1)

    content = file.read_text(encoding="utf-8")
    compiled, disabled = compile_policy_file(content)

    table = Table(title=f"Rules: {file.name}")
    table.add_column("Name", style="cyan")
    table.add_column("Layer", justify="center")
    table.add_column("Channel")
    table.add_column("Mode")
    table.add_column("Effect", style="bold")
    table.add_column("Cooldown", justify="right")
    table.add_column("Guard")

    for r in compiled:
        guard_str = ""
        if r.guard.tool_names:
            guard_str = "|".join(sorted(r.guard.tool_names))
        table.add_row(
            r.rule_id,
            f"L{r.layer}",
            r.channel,
            r.mode,
            r.effect.effect,
            str(r.cooldown_turns) if r.cooldown_turns else "",
            guard_str,
        )

    console.print(table)

    if disabled:
        console.print(f"\n[dim]Disabled: {', '.join(disabled)}[/dim]")


@app.command()
def lint(
    file: Path = typer.Argument(..., help="Policy YAML file to validate"),
) -> None:
    """Validate a policy YAML file — report compilation errors."""
    from .compiler import CompileError

    if not file.exists():
        console.print(f"[red]File not found: {file}[/red]")
        raise typer.Exit(1)

    content = file.read_text(encoding="utf-8")

    import yaml

    try:
        data = yaml.safe_load(content)
    except yaml.YAMLError as e:
        console.print(f"[red]YAML parse error: {e}[/red]")
        raise typer.Exit(1)

    if not isinstance(data, dict):
        console.print("[red]Policy file must be a YAML mapping[/red]")
        raise typer.Exit(1)

    rules_data = data.get("rules", [])
    if not isinstance(rules_data, list):
        console.print("[red]'rules' must be a list[/red]")
        raise typer.Exit(1)

    errors: list[str] = []
    ok_count = 0
    from .compiler import compile_rule

    for rule_dict in rules_data:
        if not isinstance(rule_dict, dict):
            continue
        try:
            compile_rule(rule_dict)
            ok_count += 1
        except CompileError as e:
            errors.append(str(e))

    if errors:
        console.print(f"[green]{ok_count} rules OK[/green]")
        for err in errors:
            console.print(f"[red]  ERROR: {err}[/red]")
        raise typer.Exit(1)
    else:
        console.print(f"[green]All {ok_count} rules compiled successfully.[/green]")


@ifg_app.command("backfill")
def ifg_backfill(
    session: str = typer.Option(..., "--session", "-s", help="Session ID to extract"),
    db_path: Path | None = typer.Option(
        None,
        "--db-path",
        help="Policy SQLite db path. Defaults to $AGENTM_HOME/policy_state/policy.db.",
    ),
    source: str = typer.Option(
        "policy",
        "--source",
        help="Input source: policy, trajectory, or auto.",
    ),
    cwd: Path | None = typer.Option(
        None,
        "--cwd",
        help="Project cwd for resolving trajectory source.",
    ),
    extractor_version: str | None = typer.Option(
        None,
        "--extractor-version",
        help="Override IFG extractor version for controlled re-runs.",
    ),
    replace: bool = typer.Option(
        True,
        "--replace/--append",
        help="Delete existing IFG rows for this session/version before extraction.",
    ),
    include_checkpoint: bool = typer.Option(
        True,
        "--include-checkpoint/--committed-only",
        help="With trajectory source, include the latest incomplete checkpoint.",
    ),
) -> None:
    """Build IFG facts from policy events or the selected trajectory store."""
    from .ifg import (
        IFG_EXTRACTOR_VERSION,
        backfill_ifg_from_policy_events,
        backfill_ifg_from_trajectory_turns,
    )

    path = db_path or _db_path(session_id=session, cwd=cwd)
    normalized_source = source.lower()
    if normalized_source not in {"policy", "trajectory", "auto"}:
        console.print("[red]Invalid --source. Use policy, trajectory, or auto.[/red]")
        raise typer.Exit(2)
    if normalized_source == "policy" and not path.exists():
        console.print(f"[red]No policy database at {path}[/red]")
        raise typer.Exit(1)
    if normalized_source in {"trajectory", "auto"}:
        path.parent.mkdir(parents=True, exist_ok=True)

    version = extractor_version or IFG_EXTRACTOR_VERSION
    used_source = normalized_source
    committed_turns: int | None = None
    checkpoint_count: int | None = None
    engine = create_sqlite_engine(path)
    try:
        with engine.begin() as conn:
            if normalized_source in {"policy", "auto"} and _has_policy_tool_events(
                conn,
                session,
            ):
                result = backfill_ifg_from_policy_events(
                    conn,
                    session,
                    replace=replace,
                    extractor_version=version,
                )
                used_source = "policy"
            elif normalized_source == "policy":
                console.print(
                    "[yellow]No policy_tool_events for session.[/yellow] "
                    "Run `policy_engine backfill --session ...` first, or use "
                    "`policy_engine ifg backfill --source trajectory`."
                )
                return
            else:
                from agentm.storage.trajectory.resolve import resolve_trajectory_store

                resolved = resolve_trajectory_store(str(cwd) if cwd else None)
                if resolved is None:
                    console.print(
                        "[red]No trajectory store found.[/red]\n"
                        "[dim]Run from a project with agentm.toml, or set "
                        "AGENTM_TRAJECTORY_DSN / AGENTM_TRAJECTORY_DIR.[/dim]"
                    )
                    raise typer.Exit(1)
                try:
                    try:
                        meta, turns = resolved.store.load(session)
                    except KeyError:
                        console.print(f"[red]Session not found: {session}[/red]")
                        raise typer.Exit(1)
                    checkpoint = (
                        resolved.store.load_checkpoint(session)
                        if include_checkpoint
                        else None
                    )
                    trajectory_records = list(turns)
                    if checkpoint is not None:
                        trajectory_records.append(checkpoint)
                    committed_turns = len(turns)
                    checkpoint_count = 1 if checkpoint is not None else 0
                    trajectory_cwd = (
                        str(Path(meta.cwd)) if meta.cwd else str(cwd or Path.cwd())
                    )
                    result = backfill_ifg_from_trajectory_turns(
                        conn,
                        session,
                        trajectory_records,
                        cwd=trajectory_cwd,
                        replace=replace,
                        extractor_version=version,
                    )
                finally:
                    resolved.close()
                used_source = "trajectory"
    finally:
        engine.dispose()

    if result.source_events == 0:
        console.print(
            f"[yellow]No IFG source tool events for session via {used_source}.[/yellow]"
        )
        return

    console.print(
        "[green]Backfilled IFG[/green] "
        f"session={result.session_id} source={used_source} "
        f"version={result.extractor_version} "
        f"source_events={result.source_events} actions={result.actions} "
        f"nodes={result.graph_nodes} edges={result.graph_edges} "
        f"files={result.files} file_edges={result.file_edges} "
        f"source_units={result.source_units} path_candidates={result.path_candidates} "
        f"unresolved_path_candidates={result.unresolved_path_candidates} "
        f"symbols={result.symbols} action_symbol_edges={result.action_symbol_edges} "
        f"file_symbol_edges={result.file_symbol_edges} "
        f"symbol_symbol_edges={result.symbol_symbol_edges} "
        f"symbol_mentions={result.symbol_mentions} "
        f"unresolved_symbol_mentions={result.unresolved_symbol_mentions} "
        f"errors={result.errors} "
        f"deleted={result.deleted} db={path}"
    )
    if result.action_kinds:
        console.print(
            "[dim]action_kinds="
            + ", ".join(
                f"{kind}:{count}" for kind, count in sorted(result.action_kinds.items())
            )
            + "[/dim]"
        )
    if used_source == "trajectory":
        console.print(
            f"[dim]trajectory_turns={committed_turns} "
            f"checkpoint={checkpoint_count}[/dim]"
        )


@ifg_app.command("status")
def ifg_status(
    session: str = typer.Option(..., "--session", "-s", help="Session ID to inspect"),
    db_path: Path | None = typer.Option(
        None,
        "--db-path",
        help="Policy SQLite db path. Defaults to $AGENTM_HOME/policy_state/policy.db.",
    ),
    extractor_version: str | None = typer.Option(
        None,
        "--extractor-version",
        help="Filter by extractor version.",
    ),
) -> None:
    """Show persisted IFG extraction counts for a session."""
    path = db_path or _db_path(session_id=session)
    if not path.exists():
        console.print(f"[red]No policy database at {path}[/red]")
        raise typer.Exit(1)

    engine = create_sqlite_engine(path)
    try:
        with engine.connect() as conn:
            if not _table_exists(conn, "ifg_actions"):
                console.print(
                    "[dim]No IFG tables found. Run `policy_engine ifg backfill`.[/dim]"
                )
                return
            counts = _ifg_counts(conn, session, extractor_version)
            action_kinds = _ifg_action_kind_counts(conn, session, extractor_version)
            summary = _ifg_latest_summary(conn, session, extractor_version)
    finally:
        engine.dispose()

    table = Table(title="IFG Status")
    table.add_column("Table", style="cyan")
    table.add_column("Rows", justify="right")
    for name, count in counts.items():
        table.add_row(name, str(count))
    console.print(table)

    if action_kinds:
        kinds = Table(title="IFG Action Kinds")
        kinds.add_column("Kind", style="cyan")
        kinds.add_column("Count", justify="right")
        for kind, count in action_kinds:
            kinds.add_row(kind, str(count))
        console.print(kinds)
    if summary:
        console.print(
            f"[dim]latest_summary={_short_json(summary, max_chars=220)}[/dim]"
        )


@ifg_app.command("clear")
def ifg_clear(
    session: str = typer.Option(..., "--session", "-s", help="Session ID to clear"),
    db_path: Path | None = typer.Option(
        None,
        "--db-path",
        help="Policy SQLite db path. Defaults to $AGENTM_HOME/policy_state/policy.db.",
    ),
    extractor_version: str | None = typer.Option(
        None,
        "--extractor-version",
        help="Only clear rows for this extractor version.",
    ),
) -> None:
    """Delete persisted IFG rows for a session."""
    from .ifg import delete_ifg_session

    path = db_path or _db_path(session_id=session)
    if not path.exists():
        console.print(f"[red]No policy database at {path}[/red]")
        raise typer.Exit(1)
    engine = create_sqlite_engine(path)
    try:
        with engine.begin() as conn:
            deleted = delete_ifg_session(
                conn,
                session,
                extractor_version=extractor_version,
            )
    finally:
        engine.dispose()
    scope = extractor_version or "all versions"
    console.print(f"Deleted {deleted} IFG rows for session={session} scope={scope}.")


@ifg_app.command("serve")
def ifg_serve(
    session: str = typer.Option(..., "--session", "-s", help="Session ID to view"),
    db_path: Path | None = typer.Option(
        None,
        "--db-path",
        help="Policy SQLite DB path. Auto-detected from the session by default.",
    ),
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        help="HTTP bind host.",
    ),
    port: int = typer.Option(
        8765,
        "--port",
        min=0,
        max=65535,
        help="HTTP port. Use 0 to select an available port.",
    ),
    refresh_ms: int = typer.Option(
        1500,
        "--refresh-ms",
        min=250,
        max=60_000,
        help="Browser refresh interval in milliseconds.",
    ),
    open_browser: bool = typer.Option(
        False,
        "--open/--no-open",
        help="Open the viewer in the default browser.",
    ),
) -> None:
    """Serve a live, component-aware IFG graph in the browser."""
    from .ifg.web import create_ifg_web_server

    path = db_path or _db_path(session_id=session)
    if not path.exists():
        console.print(f"[red]No policy database at {path}[/red]")
        raise typer.Exit(1)

    server = create_ifg_web_server(
        path,
        session,
        host=host,
        port=port,
        refresh_ms=refresh_ms,
    )
    bound_host, bound_port = server.server_address[:2]
    visible_host = "127.0.0.1" if bound_host in {"0.0.0.0", "::"} else bound_host
    url = f"http://{visible_host}:{bound_port}/"
    console.print(f"[green]IFG viewer[/green] {url}")
    console.print(
        f"[dim]session={session} db={path} refresh={refresh_ms}ms (Ctrl+C to stop)[/dim]"
    )
    if open_browser:
        import webbrowser

        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        console.print("\n[dim]IFG viewer stopped.[/dim]")
    finally:
        server.server_close()


def _ifg_counts(
    conn: Connection,
    session: str,
    extractor_version: str | None,
) -> dict[str, int]:
    tables = (
        "ifg_nodes",
        "ifg_edges",
        "ifg_normalized_tool_events",
        "ifg_actions",
        "ifg_files",
        "ifg_action_file_edges",
        "ifg_source_units",
        "ifg_path_candidates",
        "ifg_symbols",
        "ifg_action_symbol_edges",
        "ifg_file_symbol_edges",
        "ifg_symbol_symbol_edges",
        "ifg_symbol_mentions",
        "ifg_extraction_error",
    )
    counts: dict[str, int] = {}
    for table_name in tables:
        if not _table_exists(conn, table_name):
            counts[table_name] = 0
            continue
        where, params = _ifg_session_filter(session, extractor_version)
        row = conn.exec_driver_sql(
            f"SELECT COUNT(*) FROM {table_name} {where}",  # noqa: S608
            tuple(params),
        ).fetchone()
        counts[table_name] = int(row[0]) if row else 0
    return counts


def _ifg_action_kind_counts(
    conn: Connection,
    session: str,
    extractor_version: str | None,
) -> list[tuple[str, int]]:
    where, params = _ifg_session_filter(session, extractor_version)
    rows = conn.exec_driver_sql(
        f"""
        SELECT action_kind, COUNT(*) AS count
        FROM ifg_actions
        {where}
        GROUP BY action_kind
        ORDER BY count DESC, action_kind ASC
        """,  # noqa: S608
        tuple(params),
    ).fetchall()
    return [(str(row[0]), int(row[1])) for row in rows]


def _ifg_latest_summary(
    conn: Connection,
    session: str,
    extractor_version: str | None,
) -> str | None:
    if not _table_exists(conn, "ifg_session_summary"):
        return None
    where, params = _ifg_session_filter(session, extractor_version)
    row = conn.exec_driver_sql(
        f"""
        SELECT summary_json
        FROM ifg_session_summary
        {where}
        ORDER BY updated_at DESC
        LIMIT 1
        """,  # noqa: S608
        tuple(params),
    ).fetchone()
    return str(row[0]) if row else None


def _ifg_session_filter(
    session: str,
    extractor_version: str | None,
) -> tuple[str, list[str]]:
    conditions = ["session_id = ?"]
    params = [session]
    if extractor_version:
        conditions.append("extractor_version = ?")
        params.append(extractor_version)
    return f"WHERE {' AND '.join(conditions)}", params


@app.command()
def prune(
    days: int = typer.Option(
        30, "--days", "-d", help="Remove records older than N days"
    ),
) -> None:
    """Remove old entries from the policy database."""
    engine = _connect()
    if not engine:
        return

    cutoff = time.time() - (days * 86400)
    deleted = 0
    try:
        with engine.begin() as conn:
            for table_name in ("event_log", "policy_tool_events", "policy_eval_error"):
                if _table_exists(conn, table_name):
                    cursor = conn.exec_driver_sql(
                        f"DELETE FROM {table_name} WHERE ts < ?",
                        (cutoff,),
                    )
                    deleted += cursor.rowcount
            for table_name in (
                "policy_file_state",
                "policy_entity_state",
                "policy_context_state",
                "policy_turn_summary",
            ):
                if _table_exists(conn, table_name):
                    cursor = conn.exec_driver_sql(
                        f"DELETE FROM {table_name} WHERE updated_at < ?",
                        (cutoff,),
                    )
                    deleted += cursor.rowcount
    finally:
        engine.dispose()
    console.print(f"Pruned {deleted} records older than {days} days.")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
