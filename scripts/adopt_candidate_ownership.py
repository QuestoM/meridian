"""Where this piece may write, enforced at the one line that writes.

Section 8.2 of the specification gives this piece three paths:
``scripts/adopt_candidate.py``, ``models/candidates/**`` and
``models/releases/**`` by handover from P7. A path absent from that row is
frozen by absence, and ``models/tv_break_coefficients.json`` is absent from it.

That is a real gap and it is not a hypothetical one. The last step of an
adoption copies the chosen candidate over the shipped artifact, so the act this
piece owns ends with a write to a file this piece does not own. Nothing has
landed on this tree, because every candidate fails at least one adoption check,
so the violation has been latent rather than committed. Latent is not closed.

**What this module does about it.** Every write the adoption act makes goes
through one guard, and the guard refuses a target outside the row. The shipped
artifact is named here as the single pending path rather than special-cased into
silence, and it is released by a ruling recorded on disk, under
``models/releases/ownership/``, which is inside the row. So the lead records the
ruling in a file rather than a builder granting itself a path in code, the
release is auditable, and until it exists the act stops with a check that names
the exact line of the specification it is waiting on.

The same shape as the owner approval one step earlier in this act, deliberately:
a write this terminal may not authorise on its own is released by an artifact
naming who authorised it, never by a flag on a command line.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

# The write surface, relative to the repository root, from the row itself. The
# third entry is the frontend column of the same row.
WRITE_SURFACE = ("models/candidates", "models/releases",
                 "tv-break-dashboard/src/model/candidates")

# The one path the act writes that the row does not carry.
PENDING_PATH = "models/tv_break_coefficients.json"

# Where the ruling that puts it on the row is recorded. Inside the write
# surface, because the lead records it and this piece only reads it.
RULING_FILE = "ownership/shipped_artifact.json"

SPEC_ROW = "docs/ux-gauntlet/spec.md section 8.2, wave-two table, row P12"


class WriteOutsideTheRow(Exception):
    """Raised when a write would land on a path this piece does not own."""


def relative(root: Path, target: Path) -> str:
    """The target as the ownership row spells it, or its absolute path."""
    try:
        return Path(target).resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return Path(target).as_posix()


def ruling_path(releases_dir: Path) -> Path:
    return Path(releases_dir) / RULING_FILE


def ruling(releases_dir: Path) -> Optional[dict[str, Any]]:
    """The recorded ruling, or nothing. A ruling naming another path is nothing.

    The path is matched rather than assumed, so a ruling written about some
    other file cannot release this one by sitting in the right directory.
    """
    path = ruling_path(releases_dir)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or str(payload.get("path") or "") != PENDING_PATH:
        return None
    return payload


def may_write(root: Path, target: Path, releases_dir: Path) -> bool:
    """Whether one write is inside the row, counting a recorded ruling."""
    name = relative(root, target)
    if any(name == root_name or name.startswith(root_name + "/")
           for root_name in WRITE_SURFACE):
        return True
    return name == PENDING_PATH and ruling(releases_dir) is not None


def guard(root: Path, target: Path, releases_dir: Path) -> None:
    """Refuse a write outside the row, naming the row and what would release it."""
    if may_write(root, target, releases_dir):
        return
    name = relative(root, target)
    raise WriteOutsideTheRow(
        f"{name} is not on this piece's ownership row in {SPEC_ROW}, "
        f"which lists {', '.join(WRITE_SURFACE)}. Nothing was written.")


def state(root: Path, releases_dir: Path) -> dict[str, Any]:
    """Whether the last step of an adoption may write, and on whose authority."""
    recorded = ruling(releases_dir)
    return {
        "path": PENDING_PATH,
        "spec_row": SPEC_ROW,
        "write_surface": list(WRITE_SURFACE),
        "ruled": recorded is not None,
        "ruling": recorded,
        "ruling_file": relative(root, ruling_path(releases_dir)),
    }
