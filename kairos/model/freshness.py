"""Coefficient freshness: tell honestly whether the measured deltas are stale.

The optimizer's retention cost comes from per-cell coefficients measured on the
reference data and stored in ``models/tv_break_coefficients.json``. When that
source data changes (the current one-month Nov-2024 sample is to be replaced by
~2 years of production data), the stored coefficients silently stop matching the
data they claim to describe. A number presented as current that is actually
stale is a dishonesty risk, so this module detects staleness and surfaces it.

The check is a pure, read-only comparison: the JSON's ``metadata`` block carries
``source_fingerprints`` (each source file's relative POSIX path -> the sha256 it
had when the coefficients were computed, see
:mod:`scripts.compute_measured_coefficients`). :func:`coefficient_freshness`
re-hashes those files on disk now and compares. It returns one of three honest
states and never invents a "fresh":

  * ``fresh``   every fingerprinted file still exists and its current sha256
                matches the stored one.
  * ``stale``   at least one fingerprinted file's current sha256 differs; the
                changed relative paths are listed so the operator knows what to
                recompute.
  * ``unknown`` the fingerprints are absent (an old JSON written before this
                feature), or a fingerprinted file is no longer on disk, so
                staleness cannot be verified. This is "cannot verify", never a
                false "fresh".

No clock is read here (``computed_at`` is only echoed from the metadata) and no
network is touched, so the result is deterministic given the filesystem.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

from kairos.observability.run_log import checksum_file

# The metadata key that holds {relative posix path: sha256} for the source files
# the coefficients were computed from. Written by the compute script.
FINGERPRINTS_KEY = "source_fingerprints"
# The metadata key that holds the UTC ISO-8601 compute timestamp.
COMPUTED_AT_KEY = "computed_at"


def coefficient_freshness(metadata: Mapping[str, Any], *, root: Path) -> dict[str, Any]:
    """Compare the stored source fingerprints against the files on disk now.

    ``metadata`` is the coefficients JSON's ``metadata`` block.  ``root`` is the
    repo root the relative fingerprint paths are resolved against (each path is a
    stable POSIX-style string such as ``data/reference/Spots.xlsx``).

    Returns a dict with:

      * ``status``        ``"fresh"`` / ``"stale"`` / ``"unknown"``.
      * ``computed_at``   the metadata timestamp, or ``None`` when absent.
      * ``changed_files`` the relative paths whose current sha256 differs from
                          the stored one (only populated for ``stale``).
      * ``reason``        a one-line, operator-readable explanation.

    The function is pure and read-only: it hashes files but writes nothing and
    reads no clock, so the same filesystem always yields the same verdict.
    """
    root = Path(root)
    computed_at: Optional[str] = _coerce_str(metadata.get(COMPUTED_AT_KEY))
    fingerprints = metadata.get(FINGERPRINTS_KEY)

    if not isinstance(fingerprints, Mapping) or not fingerprints:
        # No fingerprints to check against. This is an older JSON (or one written
        # without the freshness feature): we honestly cannot verify freshness.
        return {
            "status": "unknown",
            "computed_at": computed_at,
            "changed_files": [],
            "reason": (
                "No source_fingerprints in the coefficients metadata; "
                "cannot verify whether the measured deltas match the data on disk."
            ),
        }

    missing: list[str] = []
    changed: list[str] = []
    for rel_path, stored_digest in fingerprints.items():
        rel = str(rel_path)
        current_digest = checksum_file(root / rel)
        if current_digest is None:
            # A fingerprinted source file is gone. We cannot prove freshness, so
            # we say "unknown" rather than guess "fresh" or "stale".
            missing.append(rel)
            continue
        if current_digest != str(stored_digest):
            changed.append(rel)

    if missing:
        missing_sorted = sorted(missing)
        return {
            "status": "unknown",
            "computed_at": computed_at,
            "changed_files": [],
            "reason": (
                "Cannot verify freshness; fingerprinted source file(s) missing on disk: "
                + ", ".join(missing_sorted)
            ),
        }

    if changed:
        changed_sorted = sorted(changed)
        return {
            "status": "stale",
            "computed_at": computed_at,
            "changed_files": changed_sorted,
            "reason": (
                "Source data changed since the coefficients were computed; "
                "recompute to refresh. Changed file(s): " + ", ".join(changed_sorted)
            ),
        }

    return {
        "status": "fresh",
        "computed_at": computed_at,
        "changed_files": [],
        "reason": (
            "All fingerprinted source files match the coefficients; "
            "the measured retention deltas are current."
        ),
    }


def _coerce_str(value: Any) -> Optional[str]:
    """Return ``value`` as a non-empty string, else ``None`` (no fabrication)."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None
