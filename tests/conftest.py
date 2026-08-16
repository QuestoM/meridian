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
import tempfile
from pathlib import Path

import pytest

os.environ.setdefault("KAIROS_AUTH_DISABLED", "1")
# TestClient requests are real audit events unless the store is relocated.
# Give every pytest process its own disposable ledger so a verification run can
# never append synthetic users/actions to the operator's local activity record.
os.environ.setdefault("KAIROS_AUDIT_DIR", tempfile.mkdtemp(prefix="kairos-pytest-audit-"))
# Settings writes create immutable history points. Keep those points disposable
# too, so a passing TestClient suite cannot append synthetic versions to the
# operator's real timeline.
os.environ.setdefault("KAIROS_VERSIONS_DIR", tempfile.mkdtemp(prefix="kairos-pytest-versions-"))
# Ambient Claude Code Keychain OAuth on a developer machine must not make
# assistant suites report available=true without an explicit test key.
os.environ.setdefault("KAIROS_ASSISTANT_USE_CLAUDE_CODE_OAUTH", "0")
# A product recompute writes the version-controlled plan of record by design.
# Tests and browser-driven agent checks must never do that accidentally: suites
# that genuinely exercise a successful write redirect it to ``tmp_path`` first.
# Individual guard tests may temporarily override it only after relocating the
# shipped path; ambient shell state must not weaken the suite's safety boundary.
os.environ["KAIROS_PLAN_READONLY"] = "1"


# Unrelated optimizer tests exercise an explicit neutral-inventory baseline.
# The repository's current real inventory source is intentionally present but
# all-invalid, and production API boundaries now refuse it. Letting that ambient
# operator file decide every other unit/integration test would turn thousands of
# contracts into repetitions of the same refusal. The dedicated inventory suites
# point this constant at present-invalid fixtures and prove every authoritative
# boundary; all other tests get the documented "not uploaded" identity signal.
_NEUTRAL_INVENTORY = Path(tempfile.mkdtemp(prefix="kairos-pytest-inventory-")) / "not-uploaded.csv"


@pytest.fixture(autouse=True)
def _neutral_optional_inventory(monkeypatch):
    from kairos.optimize import inventory

    monkeypatch.setattr(inventory, "DEFAULT_INVENTORY_PATH", _NEUTRAL_INVENTORY)
