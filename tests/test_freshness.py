"""Tests for the coefficient freshness guard.

The freshness checker re-hashes the source files a coefficients JSON was computed
from and reports whether the stored deltas still match the data on disk. These
tests prove the three honest states on a temporary filesystem (no real data, no
clock), so they stay deterministic and run in the fast gate:

  - fresh   : every fingerprinted file's current sha256 matches the stored one.
  - stale   : a fingerprinted file changed on disk; the changed path is named.
  - unknown : no fingerprints, or a fingerprinted file is missing -> cannot verify.
"""

from __future__ import annotations

from pathlib import Path

from kairos.model.freshness import (
    COMPUTED_AT_KEY,
    FINGERPRINTS_KEY,
    coefficient_freshness,
)
from kairos.observability.run_log import checksum_file


def _write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _metadata_for(root: Path, rel_files: dict[str, bytes], *, computed_at: str) -> dict:
    """Write the source files and return metadata with matching fingerprints."""
    fingerprints: dict[str, str] = {}
    for rel, content in rel_files.items():
        target = root / rel
        _write(target, content)
        fingerprints[rel] = checksum_file(target)
    return {COMPUTED_AT_KEY: computed_at, FINGERPRINTS_KEY: fingerprints}


def test_fresh_when_all_files_match(tmp_path: Path) -> None:
    metadata = _metadata_for(
        tmp_path,
        {"data/reference/Spots.xlsx": b"spots-bytes", "data/reference/Programmes.xlsx": b"prog-bytes"},
        computed_at="2026-06-17T00:00:00+00:00",
    )
    result = coefficient_freshness(metadata, root=tmp_path)
    assert result["status"] == "fresh"
    assert result["changed_files"] == []
    assert result["computed_at"] == "2026-06-17T00:00:00+00:00"


def test_stale_when_a_file_changes(tmp_path: Path) -> None:
    metadata = _metadata_for(
        tmp_path,
        {"data/reference/Spots.xlsx": b"spots-bytes", "data/reference/Programmes.xlsx": b"prog-bytes"},
        computed_at="2026-06-17T00:00:00+00:00",
    )
    # Mutate one source file: its sha256 now differs from the stored fingerprint.
    _write(tmp_path / "data/reference/Spots.xlsx", b"spots-bytes-CHANGED")
    result = coefficient_freshness(metadata, root=tmp_path)
    assert result["status"] == "stale"
    assert result["changed_files"] == ["data/reference/Spots.xlsx"]
    assert "data/reference/Spots.xlsx" in result["reason"]


def test_unknown_when_no_fingerprints(tmp_path: Path) -> None:
    # An old JSON written before the freshness feature carries no fingerprints.
    result = coefficient_freshness({"detrended": True}, root=tmp_path)
    assert result["status"] == "unknown"
    assert result["changed_files"] == []
    assert result["computed_at"] is None


def test_unknown_when_fingerprinted_file_missing(tmp_path: Path) -> None:
    metadata = _metadata_for(
        tmp_path,
        {"data/reference/Spots.xlsx": b"spots-bytes"},
        computed_at="2026-06-17T00:00:00+00:00",
    )
    # Delete a fingerprinted file: freshness cannot be verified, so "unknown",
    # never a false "fresh".
    (tmp_path / "data/reference/Spots.xlsx").unlink()
    result = coefficient_freshness(metadata, root=tmp_path)
    assert result["status"] == "unknown"
    assert "Spots.xlsx" in result["reason"]


def test_empty_fingerprints_is_unknown(tmp_path: Path) -> None:
    # A present-but-empty fingerprint map cannot prove anything: honest unknown.
    metadata = {COMPUTED_AT_KEY: "2026-06-17T00:00:00+00:00", FINGERPRINTS_KEY: {}}
    result = coefficient_freshness(metadata, root=tmp_path)
    assert result["status"] == "unknown"


def test_deterministic_no_clock(tmp_path: Path) -> None:
    # Same filesystem yields the same verdict every call (no datetime.now inside).
    metadata = _metadata_for(
        tmp_path,
        {"data/reference/Dayparts.xlsx": b"daypart-bytes"},
        computed_at="2026-06-17T00:00:00+00:00",
    )
    first = coefficient_freshness(metadata, root=tmp_path)
    second = coefficient_freshness(metadata, root=tmp_path)
    assert first == second
