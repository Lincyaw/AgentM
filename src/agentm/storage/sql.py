# code-health: ignore-file[AM025] -- SQL boundary normalizes backend rows and URLs
"""Shared SQLAlchemy entry points for AgentM storage adapters."""

from __future__ import annotations

from pathlib import Path
from typing import NoReturn

from sqlalchemy import create_engine
from sqlalchemy.engine import Connection, Engine, URL, make_url
from sqlalchemy.exc import NoSuchModuleError


def create_sql_engine(
    database_url: str | URL,
    *,
    pool_pre_ping: bool = True,
) -> Engine:
    """Create a SQLAlchemy engine with AgentM's supported driver defaults."""

    url = normalize_sql_url(database_url)
    try:
        return create_engine(
            url,
            future=True,
            pool_pre_ping=pool_pre_ping,
        )
    except NoSuchModuleError as exc:
        _raise_driver_hint(url, exc)


def create_sqlite_engine(path: str | Path) -> Engine:
    """Create an engine for a local SQLite database file."""

    db_path = Path(path).expanduser()
    url = sqlite_url(db_path)
    return create_engine(url, future=True)


def sqlite_url(path: str | Path) -> URL:
    """Return a SQLAlchemy URL for a local SQLite database file."""

    db_path = Path(path).expanduser()
    return URL.create("sqlite", database=str(db_path))


def normalize_sql_url(database_url: str | URL) -> URL:
    """Normalize shorthand URLs onto the SQLAlchemy dialects used by AgentM."""

    if isinstance(database_url, URL):
        return _normalize_url(database_url)
    url = make_url(database_url)
    return _normalize_url(url)


def execute_script(connection: Connection, sql: str) -> None:
    """Execute a semicolon-delimited SQL script on one SQLAlchemy connection."""

    for statement in _split_sql_script(sql):
        connection.exec_driver_sql(statement)


def _normalize_url(url: URL) -> URL:
    if url.drivername == "postgres":
        return url.set(drivername="postgresql+psycopg")
    if url.drivername == "postgresql":
        return url.set(drivername="postgresql+psycopg")
    if url.drivername == "clickhouse":
        _register_clickhouse_dialect()
        return url.set(drivername="clickhouse+connect")
    if url.drivername == "clickhouse+connect":
        _register_clickhouse_dialect()
        return url
    if url.drivername in {"http", "https"}:
        _register_clickhouse_dialect()
        database = url.database or url.normalized_query.get("database", ("otel",))[0]
        query = dict(url.query)
        if url.drivername == "https":
            query.setdefault("secure", "true")
        return URL.create(
            "clickhouse+connect",
            username=url.username,
            password=url.password,
            host=url.host,
            port=url.port,
            database=database,
            query=query,
        )
    return url


def _register_clickhouse_dialect() -> None:
    try:
        import clickhouse_connect.cc_sqlalchemy  # noqa: F401
    except ModuleNotFoundError as exc:
        if exc.name != "clickhouse_connect":
            raise
        raise RuntimeError(
            "ClickHouse SQL storage requires the 'agentm[storage-clickhouse]' "
            "extra (clickhouse-connect)."
        ) from exc


def _raise_driver_hint(database_url: URL, exc: NoSuchModuleError) -> NoReturn:
    driver = database_url.drivername
    if driver.startswith("postgres"):
        raise RuntimeError(
            "Postgres SQL storage requires the 'agentm[storage-postgres]' "
            "extra (psycopg >= 3.2)."
        ) from exc
    if driver.startswith("clickhouse"):
        raise RuntimeError(
            "ClickHouse SQL storage requires the 'agentm[storage-clickhouse]' "
            "extra (clickhouse-connect)."
        ) from exc
    raise exc


def _split_sql_script(sql: str) -> tuple[str, ...]:
    statements: list[str] = []
    current: list[str] = []
    quote: str | None = None
    escaped = False
    for char in sql:
        if escaped:
            current.append(char)
            escaped = False
            continue
        if quote is not None:
            current.append(char)
            if char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            current.append(char)
            quote = char
            continue
        if char == ";":
            statement = "".join(current).strip()
            if statement:
                statements.append(statement)
            current = []
            continue
        current.append(char)
    tail = "".join(current).strip()
    if tail:
        statements.append(tail)
    return tuple(statements)


__all__ = [
    "create_sql_engine",
    "create_sqlite_engine",
    "execute_script",
    "normalize_sql_url",
    "sqlite_url",
]
