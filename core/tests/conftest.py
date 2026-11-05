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
