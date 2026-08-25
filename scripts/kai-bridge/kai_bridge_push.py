#!/usr/bin/env python3
"""Push the live Claude Code access token to the cloud bridge secret.

Run from the repo root (the launchd agent does):

    ~/.venvs/meridian/bin/python scripts/kai-bridge/kai_bridge_push.py

The token travels keychain -> pipe -> AWS CLI stdin and is never printed or
passed as an argument. The refresh token never leaves this machine.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from kairos_api import assistant_bridge  # noqa: E402

SECRET_ID = os.environ.get("KAIROS_BRIDGE_SECRET_ID", "kairos/assistant-oauth-token")
PROFILE = os.environ.get("KAIROS_BRIDGE_AWS_PROFILE", "kairos")
REGION = os.environ.get("KAIROS_BRIDGE_AWS_REGION", "il-central-1")

if __name__ == "__main__":
    sys.exit(assistant_bridge.push(SECRET_ID, PROFILE, REGION))
