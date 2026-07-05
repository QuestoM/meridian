#!/usr/bin/env python3
"""Seed the first admin account for the Kairos login system.

Usage:
    python scripts/init_auth.py

Behaviour:
- Creates KAIROS_AUTH_DIR (default data/auth) with users.json and seeds the
  "admin" account. From that moment every /api request requires a signed-in
  session; a running server picks the change up immediately.
- If KAIROS_ADMIN_PASSWORD is set, that password is used and never printed
  or written anywhere.
- Otherwise a one-time password is generated, printed once below and written
  once to data/auth/initial_admin_password.txt (mode 600). Sign in as admin
  and change it right away; the dashboard forces the change.
- Refuses to touch a store that already has accounts.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kairos_api import auth_store  # noqa: E402


def main() -> int:
    if auth_store.store_initialized() and auth_store.load_users():
        print(f"The login store at {auth_store.users_path()} already has accounts. No changes made.")
        print("To start over, stop the server, delete that file and rerun this script.")
        return 1
    try:
        username, generated = auth_store.seed_initial_admin()
    except (RuntimeError, ValueError) as exc:
        print(f"Could not seed the login store: {exc}")
        return 1
    print(f"Login store created at {auth_store.users_path()}")
    print(f"Admin account seeded: {username}")
    if generated:
        print(f"One-time admin password: {generated}")
        print(f"Also written once to {auth_store.auth_dir() / 'initial_admin_password.txt'} (mode 600).")
        print("Sign in as admin and change this password now; the dashboard will require it.")
    else:
        print("Password taken from KAIROS_ADMIN_PASSWORD (not shown).")
    print("The API now requires sign-in; a running server enforces this immediately.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
