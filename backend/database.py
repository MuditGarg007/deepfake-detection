"""Raw-SQL database access: sqlite3 locally, psycopg2 against Neon/Postgres.

There is no ORM here. Statements are written by hand with ``?`` placeholders and
translated to ``%s`` for psycopg2 by :func:`sql`; both drivers bind parameters
themselves, so values are never formatted into the statement text.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from typing import Any, Iterator, Sequence

from .config import settings

logger = logging.getLogger(__name__)

_url = settings.database_url
IS_SQLITE = _url.startswith("sqlite")

# ``sqlite:///path`` for the fallback; a libpq URI for Postgres. The
# ``+psycopg2`` dialect suffix is SQLAlchemy notation that libpq does not know.
_SQLITE_PATH = _url.split("sqlite:///", 1)[1] if IS_SQLITE else ""
_DSN = "" if IS_SQLITE else _url.replace("+psycopg2", "", 1)

# psycopg2 pools eagerly, so the pool is built on first use rather than at
# import time — importing the app must not require a reachable database.
_pool: Any = None
_pool_lock = threading.Lock()

CREATE_TABLE_SQLITE = """
CREATE TABLE IF NOT EXISTS analyses (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    filename          TEXT NOT NULL,
    storage_path      TEXT,
    fake_probability  FLOAT NOT NULL,
    status            TEXT NOT NULL,
    suspicious_start  FLOAT,
    suspicious_end    FLOAT,
    frame_scores      JSON NOT NULL,
    created_at        TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT ck_analyses_status CHECK (status IN ('REAL','SUSPICIOUS','HIGH_RISK'))
)
"""

CREATE_TABLE_POSTGRES = """
CREATE TABLE IF NOT EXISTS analyses (
    id                SERIAL PRIMARY KEY,
    filename          TEXT NOT NULL,
    storage_path      TEXT,
    fake_probability  DOUBLE PRECISION NOT NULL,
    status            TEXT NOT NULL,
    suspicious_start  DOUBLE PRECISION,
    suspicious_end    DOUBLE PRECISION,
    frame_scores      JSONB NOT NULL,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT ck_analyses_status CHECK (status IN ('REAL','SUSPICIOUS','HIGH_RISK'))
)
"""

# Columns added to the table after it first shipped. A database written before
# the column existed keeps the old shape, so they are added on startup if
# missing. Every one must be nullable; anything else (backfills, type changes,
# drops) is a real migration.
ADDED_COLUMNS: dict[str, dict[str, str]] = {
    "analyses": {"storage_path": "TEXT"},
}


def sql(statement: str) -> str:
    """Translate ``?`` placeholders to the driver's paramstyle.

    sqlite3 is qmark, psycopg2 is pyformat. No statement in this project
    contains a literal question mark, so the substitution is unambiguous.
    """
    return statement if IS_SQLITE else statement.replace("?", "%s")


def _get_pool():
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                from psycopg2.pool import ThreadedConnectionPool

                _pool = ThreadedConnectionPool(1, 10, _DSN)
    return _pool


def _connect_sqlite() -> sqlite3.Connection:
    # SQLite opens the connection on one thread while FastAPI's threadpool
    # serves the request on another.
    connection = sqlite3.connect(_SQLITE_PATH, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    return connection


@contextmanager
def connect() -> Iterator[Any]:
    """A connection, committed on success and rolled back on any exception."""
    if IS_SQLITE:
        connection = _connect_sqlite()
        try:
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()
    else:
        pool = _get_pool()
        connection = pool.getconn()
        try:
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            pool.putconn(connection)


class Db:
    """Thin cursor helper over a live connection — execute and read rows.

    Rows come back as plain dicts from both drivers so callers do not have to
    care which one is underneath.
    """

    def __init__(self, connection: Any) -> None:
        self.connection = connection

    def _cursor(self):
        if IS_SQLITE:
            return self.connection.cursor()
        from psycopg2.extras import RealDictCursor

        return self.connection.cursor(cursor_factory=RealDictCursor)

    def execute(self, statement: str, params: Sequence[Any] = ()) -> Any:
        cursor = self._cursor()
        cursor.execute(sql(statement), tuple(params))
        return cursor

    def fetch_one(self, statement: str, params: Sequence[Any] = ()) -> dict | None:
        cursor = self.execute(statement, params)
        try:
            row = cursor.fetchone()
        finally:
            cursor.close()
        return dict(row) if row is not None else None

    def fetch_all(self, statement: str, params: Sequence[Any] = ()) -> list[dict]:
        cursor = self.execute(statement, params)
        try:
            rows = cursor.fetchall()
        finally:
            cursor.close()
        return [dict(row) for row in rows]


def get_db() -> Iterator[Db]:
    """FastAPI dependency: one connection per request."""
    with connect() as connection:
        yield Db(connection)


def to_json(value: Any) -> Any:
    """Encode a JSON column for the driver.

    psycopg2 adapts a Python object to JSONB only when it is wrapped; sqlite3
    stores the serialised text.
    """
    if IS_SQLITE:
        return json.dumps(value)
    from psycopg2.extras import Json

    return Json(value)


def from_json(value: Any) -> Any:
    """Decode a JSON column. psycopg2 already returns JSONB as Python."""
    if isinstance(value, (str, bytes)):
        return json.loads(value)
    return value


def _existing_columns(db: Db, table: str) -> set[str]:
    if IS_SQLITE:
        return {row["name"] for row in db.fetch_all(f"PRAGMA table_info({table})")}
    rows = db.fetch_all(
        "SELECT column_name FROM information_schema.columns WHERE table_name = ?",
        (table,),
    )
    return {row["column_name"] for row in rows}


def init_schema() -> None:
    """Create the table if it is missing and add any columns added since."""
    with connect() as connection:
        db = Db(connection)
        db.execute(CREATE_TABLE_SQLITE if IS_SQLITE else CREATE_TABLE_POSTGRES).close()
        for table, columns in ADDED_COLUMNS.items():
            existing = _existing_columns(db, table)
            for name, type_sql in columns.items():
                if name in existing:
                    continue
                db.execute(f'ALTER TABLE {table} ADD COLUMN "{name}" {type_sql}').close()
                logger.info("added column %s.%s", table, name)
