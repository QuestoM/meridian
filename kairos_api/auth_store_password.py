"""Password hashing for the Kairos login store: stdlib scrypt, constant time.

Split out of auth_store.py, verbatim, to keep that module under the file-size
cap when the job field landed on the account record. The parameters, the record
shape and the timing defence are unchanged, so every password already on disk
still verifies and every caller still reaches these through
``kairos_api.auth_store``, which re-exports them.

Stdlib only, like its parent, so the seed script can import the store without
pulling the engine or FastAPI.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from typing import Any

SCRYPT_N = 2**14
SCRYPT_R = 8
SCRYPT_P = 1
SCRYPT_DKLEN = 32
SCRYPT_MAXMEM = 64 * 1024 * 1024

_DUMMY_RECORD: dict[str, Any] | None = None


def hash_password(password: str) -> dict[str, Any]:
    salt = secrets.token_bytes(32)
    digest = hashlib.scrypt(
        password.encode("utf-8"), salt=salt,
        n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P, maxmem=SCRYPT_MAXMEM, dklen=SCRYPT_DKLEN,
    )
    return {
        "salt_hex": salt.hex(),
        "hash_hex": digest.hex(),
        "n": SCRYPT_N,
        "r": SCRYPT_R,
        "p": SCRYPT_P,
    }


def verify_password(password: str, record: dict[str, Any]) -> bool:
    try:
        salt = bytes.fromhex(str(record["salt_hex"]))
        expected = bytes.fromhex(str(record["hash_hex"]))
        digest = hashlib.scrypt(
            password.encode("utf-8"), salt=salt,
            n=int(record["n"]), r=int(record["r"]), p=int(record["p"]),
            maxmem=SCRYPT_MAXMEM, dklen=len(expected),
        )
    except (KeyError, TypeError, ValueError):
        return False
    return hmac.compare_digest(digest, expected)


def burn_password_check(password: str) -> None:
    """Verify against a throwaway record so unknown usernames cost the same
    time as a real password check (no account enumeration via timing)."""
    global _DUMMY_RECORD
    if _DUMMY_RECORD is None:
        _DUMMY_RECORD = hash_password(secrets.token_urlsafe(16))
    verify_password(password, _DUMMY_RECORD)
