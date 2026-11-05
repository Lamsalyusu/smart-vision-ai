"""Run the service's Alembic migrations.

The service applies migrations at startup. They can also be run by hand:

    python -m core.infrastructure.database.migrate            # upgrade to latest
    python -m core.infrastructure.database.migrate revision "add column x"   # new migration
"""

from __future__ import annotations

import sys
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy.engine import Engine

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


def alembic_config(connection=None) -> Config:
    config = Config()
    config.set_main_option("script_location", str(MIGRATIONS_DIR))
    if connection is not None:
        config.attributes["connection"] = connection
    return config


def upgrade(engine: Engine) -> None:
    with engine.begin() as connection:
        command.upgrade(alembic_config(connection), "head")


def main(argv: list[str]) -> int:
    from .engine import engine

    if argv[:1] == ["revision"]:
        message = argv[1] if len(argv) > 1 else "schema change"
        with engine.begin() as connection:
            command.revision(alembic_config(connection), message=message, autogenerate=True)
        return 0
    upgrade(engine)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
