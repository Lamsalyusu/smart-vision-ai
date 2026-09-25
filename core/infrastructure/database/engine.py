"""SQLAlchemy engine and session factory.

This module configures a PostgreSQL connection using environment variables.
It is deliberately lightweight – the actual ORM models will live in
`core.infrastructure.database.models`.

The file is imported by the repository's data‑access layer and by any
service that needs a database session.
"""

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# Optional: load a .env file in development environments
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # pragma: no cover – dotenv is optional
    pass

import logging

logger = logging.getLogger(__name__)

def _init_engine():
    environment = os.getenv("APP_ENV", os.getenv("NODE_ENV", "development")).lower()
    production = environment in {"production", "prod"}
    target_url = os.getenv("DATABASE_URL")
    postgres_user = os.getenv("POSTGRES_USER")
    postgres_password = os.getenv("POSTGRES_PASSWORD")
    postgres_db = os.getenv("POSTGRES_DB")
    postgres_host = os.getenv("POSTGRES_HOST")
    postgres_port = os.getenv("POSTGRES_PORT", "5432")
    if not target_url and postgres_user and postgres_password and postgres_db and postgres_host:
        target_url = (
            f"postgresql+psycopg2://{postgres_user}:{postgres_password}@"
            f"{postgres_host}:{postgres_port}/{postgres_db}"
        )

    if production and not target_url:
        raise RuntimeError("A production DATABASE_URL or complete PostgreSQL configuration is required")

    if target_url and target_url.startswith("sqlite") and production:
        raise RuntimeError("SQLite fallback is disabled in production")

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
            if production:
                raise RuntimeError("The configured production database is unavailable") from exc
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

__all__ = ["DATABASE_URL", "Base", "SessionLocal", "engine"]
