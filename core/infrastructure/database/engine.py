"""SQLAlchemy engine and session factory.

This module configures a PostgreSQL connection using environment variables.
It is deliberately lightweight – the actual ORM models will live in
`core.infrastructure.database.models`.

The file is imported by the repository's data‑access layer and by any
service that needs a database session.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Optional: load a .env file in development environments
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # pragma: no cover – dotenv is optional
    pass

import logging
logger = logging.getLogger(__name__)

POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_DB = os.getenv("POSTGRES_DB")
POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")

def _init_engine():
    target_url = os.getenv("DATABASE_URL")
    if not target_url and POSTGRES_USER and POSTGRES_PASSWORD and POSTGRES_DB and POSTGRES_HOST:
        target_url = (
            f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@"
            f"{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
        )

    if target_url and not target_url.startswith("sqlite"):
        try:
            # Test PostgreSQL reachability
            eng = create_engine(
                target_url,
                echo=False,
                future=True,
                connect_args={"connect_timeout": 1},
            )
            with eng.connect():
                pass
            return eng, target_url
        except Exception as exc:
            logger.warning("PostgreSQL server unreachable (%s); using resilient SQLite database fallback.", exc)

    fallback_url = os.getenv("SQLITE_URL", "sqlite:///smart_vision_ai.db")
    eng = create_engine(
        fallback_url,
        echo=False,
        future=True,
        connect_args={"check_same_thread": False},
    )
    return eng, fallback_url

engine, DATABASE_URL = _init_engine()

# ``autocommit=False`` gives us explicit transaction boundaries.
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for ORM models.
Base = declarative_base()

__all__ = ["engine", "SessionLocal", "Base", "DATABASE_URL"]
