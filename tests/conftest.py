"""Shared pytest configuration.

The API suites exercise endpoints with TestClient and no session. Once an
operator seeds the real account store (data/auth/users.json exists), the auth
middleware would wall every request with 401 and the suites would fail for a
reason unrelated to what they test. So the test session disables enforcement by
default; the dedicated auth suite (tests/test_auth.py) re-enables it per test by
deleting this variable and pointing KAIROS_AUTH_DIR at a tmp store, so the auth
behavior itself is still fully tested.
"""

import os

os.environ.setdefault("KAIROS_AUTH_DISABLED", "1")
# Ambient Claude Code Keychain OAuth on a developer machine must not make
# assistant suites report available=true without an explicit test key.
os.environ.setdefault("KAIROS_ASSISTANT_USE_CLAUDE_CODE_OAUTH", "0")
# A product recompute writes the version-controlled plan of record by design.
# Tests and browser-driven agent checks must never do that accidentally: suites
# that genuinely exercise a successful write redirect it to ``tmp_path`` first.
# Individual guard tests may temporarily override it only after relocating the
# shipped path; ambient shell state must not weaken the suite's safety boundary.
os.environ["KAIROS_PLAN_READONLY"] = "1"
