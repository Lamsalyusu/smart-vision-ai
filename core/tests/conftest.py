"""Keep the test suite on a throwaway SQLite database.

core.infrastructure.database.engine loads .env at import time; without this a
developer's real PostgreSQL credentials would receive test users and sessions.
load_dotenv() does not override variables that are already set.
"""

import os
import tempfile

_db_dir = tempfile.mkdtemp(prefix="smart-vision-tests-")
os.environ["DATABASE_URL"] = f"sqlite:///{os.path.join(_db_dir, 'test.db')}"
for _name in ("POSTGRES_HOST", "POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB"):
    os.environ.pop(_name, None)

# Never reach real external services from tests. Setting these to empty strings
# (rather than removing them) stops load_dotenv() from filling them in from .env.
# Tests that exercise email or alerts patch their own fake values and mock the sender.
for _name in (
    "EMAIL_HOST_USER", "EMAIL_HOST_PASSWORD", "EMAIL_FROM",
    "ALERT_WEBHOOK_URL", "GROQ_API_KEY",
    "SUPABASE_URL", "SUPABASE_ANON_KEY", "SUPABASE_SERVICE_ROLE_KEY",
):
    os.environ[_name] = ""
os.environ["REQUIRE_EMAIL_VERIFICATION"] = "false"


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_rate_limits():
    """Each test starts with empty rate-limit counters (all tests share one client address)."""
    from core.infrastructure.security.rate_limit import limiter

    limiter.reset()
    yield
    limiter.reset()
