import psycopg2
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
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""

OLD_COLUMNS = [
    "ALTER TABLE analyses ADD COLUMN IF NOT EXISTS storage_path TEXT",
    "ALTER TABLE analyses ADD COLUMN IF NOT EXISTS user_feedback TEXT",
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


def to_json(value):
    return Json(value)


def init_schema():
    execute(CREATE_TABLE)
    for query in OLD_COLUMNS:
        execute(query)
