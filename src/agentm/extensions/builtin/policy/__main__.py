"""Policy engine CLI — query policy data, lint rules, view stats.

Run from anywhere with uv:
    uv run python -m agentm.extensions.builtin.policy stats
    uv run python -m agentm.extensions.builtin.policy log --rule stuck-loop
    uv run python -m agentm.extensions.builtin.policy rules [file.yaml]
    uv run python -m agentm.extensions.builtin.policy lint <file.yaml>
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="policy",
    add_completion=False,
    help="Policy engine data queries and rule management.",
)
console = Console()


def _db_path() -> Path:
    import os
    agentm_home = os.environ.get("AGENTM_HOME", str(Path.home() / ".agentm"))
    return Path(agentm_home) / "policy_state" / "policy.db"


def _connect() -> sqlite3.Connection | None:
    path = _db_path()
    if not path.exists():
        console.print(f"[yellow]No policy database at {path}[/yellow]")
        return None
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


@app.command()
def stats(
    session: str | None = typer.Option(None, "--session", "-s", help="Filter by session ID"),
    days: int = typer.Option(7, "--days", "-d", help="Look back N days"),
) -> None:
    """Show per-rule firing statistics."""
    conn = _connect()
    if not conn:
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
        WHERE {' AND '.join(conditions)}
        GROUP BY rule_id, mode, effect
        ORDER BY count DESC
    """
    rows = conn.execute(sql, params).fetchall()
    conn.close()

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
    session: str | None = typer.Option(None, "--session", "-s", help="Filter by session ID"),
    rule: str | None = typer.Option(None, "--rule", "-r", help="Filter by rule name"),
    limit: int = typer.Option(20, "--limit", "-n", help="Max rows"),
) -> None:
    """Show recent effect_log entries."""
    conn = _connect()
    if not conn:
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
    rows = conn.execute(sql, [*params, limit]).fetchall()
    conn.close()

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
    days: int = typer.Option(30, "--days", "-d", help="Remove records older than N days"),
) -> None:
    """Remove old entries from the policy database."""
    conn = _connect()
    if not conn:
        return

    cutoff = time.time() - (days * 86400)
    cursor = conn.execute("DELETE FROM event_log WHERE ts < ?", (cutoff,))
    conn.commit()
    console.print(f"Pruned {cursor.rowcount} records older than {days} days.")
    conn.close()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
