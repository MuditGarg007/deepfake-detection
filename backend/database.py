"""SQLAlchemy engine, session, and Base for the analyses database."""

import logging

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from .config import settings

logger = logging.getLogger(__name__)

_url = settings.database_url
# SQLite (the no-credentials fallback) opens the connection on one thread while
# FastAPI's threadpool serves the request on another.
_connect_args = {"check_same_thread": False} if _url.startswith("sqlite") else {}

engine = create_engine(_url, connect_args=_connect_args, pool_pre_ping=True)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def ensure_schema() -> None:
    """Add columns that exist on the models but not yet in the database.

    ``create_all`` only ever creates whole tables, so a database written before a
    column was added keeps the old shape. Every column added this way must be
    nullable, which is all this project has needed; anything else (backfills,
    type changes, drops) is a real migration and belongs in Alembic.
    """
    inspector = inspect(engine)
    for table in Base.metadata.sorted_tables:
        if not inspector.has_table(table.name):
            continue
        existing = {column["name"] for column in inspector.get_columns(table.name)}
        for column in table.columns:
            if column.name in existing:
                continue
            if not column.nullable:
                raise RuntimeError(
                    f"{table.name}.{column.name} is missing and is NOT NULL — "
                    "this needs a real migration"
                )
            type_sql = column.type.compile(engine.dialect)
            with engine.begin() as connection:
                connection.execute(
                    text(f'ALTER TABLE {table.name} ADD COLUMN "{column.name}" {type_sql}')
                )
            logger.info("added column %s.%s", table.name, column.name)
