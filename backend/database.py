import psycopg2
from psycopg2 import sql
from psycopg2.extras import Json, RealDictCursor

from .config import DATABASE_URL

CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS analyses (
    id                SERIAL PRIMARY KEY,
    filename          TEXT NOT NULL,
    storage_path      TEXT,
    fake_probability  DOUBLE PRECISION NOT NULL,
    status            TEXT NOT NULL,
    suspicious_start  DOUBLE PRECISION,
    suspicious_end    DOUBLE PRECISION,
    frame_scores      JSONB NOT NULL,
    user_feedback     TEXT,
    user_id           TEXT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""

CREATE_LIVE_TABLE = """
CREATE TABLE IF NOT EXISTS live_sessions (
    id                SERIAL PRIMARY KEY,
    started_at        TIMESTAMPTZ NOT NULL,
    ended_at          TIMESTAMPTZ NOT NULL,
    duration_seconds  DOUBLE PRECISION NOT NULL,
    frames_scored     INTEGER NOT NULL,
    mean_probability  DOUBLE PRECISION NOT NULL,
    peak_probability  DOUBLE PRECISION NOT NULL,
    status            TEXT NOT NULL,
    timeline          JSONB NOT NULL,
    user_feedback     TEXT,
    user_id           TEXT
)
"""

CREATE_USERS_TABLE = """
CREATE TABLE IF NOT EXISTS users (
    id             SERIAL PRIMARY KEY,
    email          TEXT NOT NULL UNIQUE,
    password_hash  TEXT NOT NULL,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""

OLD_COLUMNS = [
    "ALTER TABLE analyses ADD COLUMN IF NOT EXISTS storage_path TEXT",
    "ALTER TABLE analyses ADD COLUMN IF NOT EXISTS user_feedback TEXT",
    # Rows from before sign-in existed keep a NULL owner and are visible to
    # no signed-in user.
    "ALTER TABLE analyses ADD COLUMN IF NOT EXISTS user_id TEXT",
    "ALTER TABLE live_sessions ADD COLUMN IF NOT EXISTS user_id TEXT",
]


def connect():
    return psycopg2.connect(DATABASE_URL)


def execute(query, params=()):
    conn = connect()
    try:
        cur = conn.cursor()
        cur.execute(query, params)
        cur.close()
        conn.commit()
    finally:
        conn.close()


def fetch_one(query, params=()):
    conn = connect()
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(query, params)
        row = cur.fetchone()
        cur.close()
        conn.commit()
    finally:
        conn.close()
    if row is None:
        return None
    return dict(row)


def fetch_all(query, params=()):
    conn = connect()
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(query, params)
        rows = cur.fetchall()
        cur.close()
        conn.commit()
    finally:
        conn.close()
    return [dict(row) for row in rows]


def table_names():
    rows = fetch_all(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'public' AND table_type = 'BASE TABLE' ORDER BY table_name"
    )
    return [row["table_name"] for row in rows]


def fetch_table(name, limit):
    """Column names and rows of one table, newest rows first when it has an id.

    `name` must come from table_names(); it is quoted as an identifier, never
    pasted into the query as text.
    """
    conn = connect()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_schema = 'public' AND table_name = %s AND column_name = 'id'",
            (name,),
        )
        order = sql.SQL("ORDER BY id DESC") if cur.fetchone() else sql.SQL("")
        cur.execute(
            sql.SQL("SELECT * FROM {} {} LIMIT %s").format(sql.Identifier(name), order),
            (limit,),
        )
        columns = [column.name for column in cur.description]
        rows = cur.fetchall()
        cur.close()
        conn.commit()
    finally:
        conn.close()
    return columns, rows


def to_json(value):
    return Json(value)


def init_schema():
    execute(CREATE_TABLE)
    execute(CREATE_LIVE_TABLE)
    execute(CREATE_USERS_TABLE)
    for query in OLD_COLUMNS:
        execute(query)
