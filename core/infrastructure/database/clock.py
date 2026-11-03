from datetime import UTC, datetime


def utcnow_naive() -> datetime:
    """Return UTC now as a naive datetime for the existing database schema."""
    return datetime.now(UTC).replace(tzinfo=None)
