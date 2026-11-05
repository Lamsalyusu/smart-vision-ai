import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from uuid import uuid4

from core.infrastructure.database.data_manager import data_manager
from core.infrastructure.manage import main


class ManageCommandTests(unittest.TestCase):
    def setUp(self):
        self.username = f"cli_{uuid4().hex[:8]}"
        data_manager.create_user("CLI User", f"{self.username}@example.com", self.username, "x")

    def run_cli(self, *args: str) -> tuple[int, str]:
        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(out):
            code = main(list(args))
        return code, out.getvalue()

    def test_set_role_promotes_user_and_is_audited(self):
        code, output = self.run_cli("set-role", self.username.upper(), "admin")
        self.assertEqual(code, 0, output)
        user = data_manager.find_user_by_identifier(self.username)
        self.assertEqual(user.role, "admin")
        events = [e for e in data_manager.get_audit_logs(50) if e["user_id"] == user.id]
        self.assertEqual(events[0]["event_type"], "ROLE_CHANGED")
        self.assertEqual(events[0]["details"]["new"], "admin")

    def test_unknown_user_fails(self):
        code, output = self.run_cli("set-role", f"missing_{uuid4().hex}", "admin")
        self.assertEqual(code, 1)
        self.assertIn("No user matches", output)

    def test_invalid_role_is_rejected(self):
        with self.assertRaises(SystemExit):
            self.run_cli("set-role", self.username, "superuser")

    def test_list_users_includes_role(self):
        code, output = self.run_cli("list-users")
        self.assertEqual(code, 0)
        self.assertIn(f"{self.username}\t{self.username}@example.com\toperator", output)


if __name__ == "__main__":
    unittest.main()
