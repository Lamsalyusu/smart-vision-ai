"""Alembic environment for the FastAPI service's database."""

from alembic import context

from core.infrastructure.database import models  # noqa: F401  (registers tables on Base)
from core.infrastructure.database.engine import Base, engine

target_metadata = Base.metadata


def run_migrations_online() -> None:
    connection = context.config.attributes.get("connection")
    if connection is not None:
        _run(connection)
        return
    with engine.connect() as conn:
        _run(conn)


def _run(connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        render_as_batch=connection.dialect.name == "sqlite",
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


if context.is_offline_mode():
    raise RuntimeError("Offline migrations are not supported; run against a database.")
run_migrations_online()
