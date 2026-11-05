"""Administrative commands for the FastAPI service.

Usage:
    python -m core.infrastructure.manage set-role <email-or-username> <role>
    python -m core.infrastructure.manage list-users
"""

from __future__ import annotations

import argparse
import sys

from .database.data_manager import data_manager
from .database.engine import SessionLocal
from .database.models import User
from .security.rbac import AppRole

ROLES = [role.value for role in AppRole]


def set_role(identifier: str, role: str) -> int:
    if role not in ROLES:
        print(f"Unknown role {role!r}; choose one of {', '.join(ROLES)}", file=sys.stderr)
        return 2
    user = data_manager.find_user_by_identifier(identifier)
    if user is None:
        print(f"No user matches {identifier!r}", file=sys.stderr)
        return 1
    try:
        previous, _ = data_manager.set_user_role(user.id, role)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    # Existing sessions keep working; the new role applies on the next request.
    data_manager.record_audit("ROLE_CHANGED", user.id, previous=previous, new=role, via="cli")
    print(f"{user.username}: {previous} -> {role}")
    return 0


def list_users() -> int:
    with SessionLocal() as db:
        for user in db.query(User).order_by(User.created_at).all():
            print(f"{user.username}\t{user.email}\t{user.role}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m core.infrastructure.manage")
    commands = parser.add_subparsers(dest="command", required=True)
    set_role_parser = commands.add_parser("set-role", help="change a user's role")
    set_role_parser.add_argument("identifier", help="email or username")
    set_role_parser.add_argument("role", choices=ROLES)
    commands.add_parser("list-users", help="list users and their roles")
    args = parser.parse_args(argv)

    if args.command == "set-role":
        return set_role(args.identifier, args.role)
    return list_users()


if __name__ == "__main__":
    sys.exit(main())
