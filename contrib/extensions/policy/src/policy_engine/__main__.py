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

app = typer.Typer(
    name="policy",
    add_completion=False,
    help="Policy engine data queries and rule management.",
)
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


def _db_path() -> Path:
    import os

    agentm_home = os.environ.get("AGENTM_HOME", str(Path.home() / ".agentm"))
    return Path(agentm_home) / "policy_state" / "policy.db"


def _connect() -> Engine | None:
    path = _db_path()
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
    engine = _connect()
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
    engine = _connect()
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

    persistence = PolicyPersistence(db_path or _db_path())
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
        f"eval_errors={result.eval_errors} deleted={deleted} db={db_path or _db_path()}"
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
            db_path=db_path or _db_path(),
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
    engine = _connect()
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
    engine = _connect()
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
