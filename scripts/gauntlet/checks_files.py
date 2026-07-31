"""What moved that should not have: data, model artifacts, settings, config.

A refactor is allowed to move code. It is not allowed to move the numbers the
code reads, and a changed data file is the quietest way for an engine figure to
drift without a single test noticing.

Every difference is cross-referenced against the paths the build order declares
each piece will create, read from the workbench state when it is present. That
turns a wall of expected additions into a short list of genuinely unexplained
ones, which is the only part a human needs to look at.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from result import Result

WATCHED = ("data", "models", "config")
IGNORE_SUFFIX = (".pyc",)
IGNORE_PARTS = {"__pycache__", "_backups", "versions"}

# Stores the running app writes by itself. Their contents are a record of somebody
# using the product, not evidence that a builder changed anything, and every one of
# them has an environment knob that redirects it. Reported, never counted against
# the wave. data/kairos_settings.json is deliberately NOT here: it feeds the engine,
# so a change to it has to be explained even though the app can write it.
RUNTIME_PREFIXES = ("data/assistant/", "data/audit/", "data/auth/", "data/versions/",
                    "data/uploads/", "data/jobs/")


def _hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _inventory(root: Path) -> dict[str, str]:
    found: dict[str, str] = {}
    for area in WATCHED:
        base = root / area
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or path.is_symlink():
                continue
            if path.suffix in IGNORE_SUFFIX or IGNORE_PARTS & set(path.parts):
                continue
            found[str(path.relative_to(root))] = _hash(path)
    return found


def _describe_change(before: Path, after: Path) -> str:
    """Say what moved inside the file, not merely that it moved.

    "changed" on a data file is an alarm nobody can act on. The added key or the
    added column is the thing a reader needs, and it is usually enough to tell a
    declared migration apart from an accident.
    """
    try:
        if after.suffix == ".json":
            a = json.loads(before.read_text(encoding="utf-8"))
            b = json.loads(after.read_text(encoding="utf-8"))
            if isinstance(a, dict) and isinstance(b, dict):
                added = sorted(set(b) - set(a))
                removed = sorted(set(a) - set(b))
                moved = sorted(k for k in set(a) & set(b) if a[k] != b[k])
                bits = []
                if added:
                    bits.append("keys added: %s" % ", ".join("%s=%r" % (k, b[k]) for k in added[:4]))
                if removed:
                    bits.append("keys removed: %s" % ", ".join(removed[:4]))
                if moved:
                    bits.append("values changed: %s" % ", ".join(
                        "%s %r to %r" % (k, a[k], b[k]) for k in moved[:4]))
                return "; ".join(bits) or "same keys, different bytes"
        if after.suffix == ".csv":
            head_a = before.read_text(encoding="utf-8", errors="replace").splitlines()
            head_b = after.read_text(encoding="utf-8", errors="replace").splitlines()
            cols_a = head_a[0].split(",") if head_a else []
            cols_b = head_b[0].split(",") if head_b else []
            new_cols = [c for c in cols_b if c not in cols_a]
            lost_cols = [c for c in cols_a if c not in cols_b]
            bits = ["rows %d to %d" % (len(head_a), len(head_b))]
            if new_cols:
                bits.append("columns added: %s" % ", ".join(new_cols))
            if lost_cols:
                bits.append("columns removed: %s" % ", ".join(lost_cols))
            return "; ".join(bits)
    except (ValueError, OSError, UnicodeDecodeError):
        pass
    return "contents differ"


def _declared_paths(repo: Path) -> dict[str, str]:
    """Paths the build order says a piece creates, mapped to the piece that owns them."""
    state = repo / "docs" / "ux-gauntlet" / "state.json"
    owners: dict[str, str] = {}
    if not state.is_file():
        return owners
    try:
        data = json.loads(state.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return owners
    for piece in data.get("pieces", []):
        files = piece.get("owner_files") or {}
        for entry in files.get("creates") or []:
            path = entry.get("path") if isinstance(entry, dict) else entry
            if path:
                owners[path.rstrip("*/")] = piece.get("id", "?")
        for path in (files.get("backend") or []) + (files.get("frontend") or []):
            if isinstance(path, str) and not path.startswith("kairos_api/assistant*"):
                owners.setdefault(path.rstrip("*/"), piece.get("id", "?"))
    return owners


def _owner_of(path: str, owners: dict[str, str]) -> str | None:
    if path in owners:
        return owners[path]
    for declared, piece in owners.items():
        if declared and path.startswith(declared):
            return piece
    return None


def check_moved_files(repo: Path, ref: Path, work_tree: Path) -> Result:
    """work_tree is the live tree, read only, because that is what actually ships."""
    r = Result("moved", "Data, models and config that moved")
    started = time.time()

    before, after = _inventory(ref), _inventory(work_tree)
    owners = _declared_paths(repo)

    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    changed = sorted(p for p in set(before) & set(after) if before[p] != after[p])
    r.seconds = time.time() - started

    unexplained: list[str] = []
    runtime = 0
    for label, paths in (("added", added), ("removed", removed), ("changed", changed)):
        for path in paths:
            if path.startswith(RUNTIME_PREFIXES):
                runtime += 1
                continue
            piece = _owner_of(path, owners)
            what = ""
            if label == "changed":
                what = ", %s" % _describe_change(ref / path, work_tree / path)
            if piece:
                r.note("%s: %s (declared by %s)%s" % (label, path, piece, what))
            else:
                r.note("%s: %s  UNEXPLAINED%s" % (label, path, what))
                unexplained.append("%s %s" % (label, path))
    if runtime:
        r.note("plus %d file(s) under the app's own runtime stores, which are a record of somebody "
               "using the product rather than a change to it" % runtime)

    measurements = {"added": len(added), "removed": len(removed), "changed": len(changed),
                    "runtime_state": runtime, "unexplained": len(unexplained),
                    "files_compared": len(before)}

    if not (added or removed or changed):
        return r.passed("nothing moved under %s" % ", ".join(WATCHED), **measurements)
    if unexplained:
        return r.failed(
            "%d file(s) moved that no build piece declares: %s"
            % (len(unexplained), "; ".join(unexplained[:6]) + (" ..." if len(unexplained) > 6 else "")),
            **measurements)
    declared = len(added) + len(removed) + len(changed) - runtime
    return r.passed(
        "%d declared change(s) from the build order, %d runtime state file(s), nothing unexplained"
        % (declared, runtime), **measurements)
