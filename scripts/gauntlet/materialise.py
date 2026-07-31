"""Put both sides of the comparison on disk without touching the shared tree.

Two rules drive every choice here.

The reference is materialised with `git archive`, not with `git worktree` and
not with a checkout. `git archive` writes nothing under `.git`, takes no lock a
builder could block on, and cannot leave administrative state behind if this
process dies. With several builders live in the shared tree that margin is
worth more than the convenience.

The working tree is copied rather than used in place, because the suite writes
into `data/` while it runs. Running it where the builders are working would be
the exact mutation this harness exists to rule out.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

# Copied by reference, never duplicated: heavy, derived, or irrelevant to behaviour.
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".pytest_cache", ".venv", "venv",
             ".mypy_cache", ".ruff_cache", "dist", ".DS_Store"}


class Materialised:
    """Both sides on disk, plus the scratch the checks are allowed to dirty."""

    def __init__(self, root: Path, reference: str, keep: bool):
        self.root = root
        self.reference = reference
        self.keep = keep
        self.ref = root / "ref"
        self.work = root / "work"
        self.scratch = root / "scratch"

    def cleanup(self) -> None:
        if self.keep:
            return
        shutil.rmtree(self.root, ignore_errors=True)


def _run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def export_reference(repo: Path, reference: str, dest: Path) -> None:
    """Reference tree from the object database. Read-only with respect to .git."""
    dest.mkdir(parents=True, exist_ok=True)
    tar_path = dest.parent / "ref.tar"
    with open(tar_path, "wb") as fh:
        proc = subprocess.run(["git", "archive", reference], cwd=repo, stdout=fh,
                              stderr=subprocess.PIPE, text=False)
    if proc.returncode != 0:
        raise RuntimeError("git archive %s failed: %s" % (reference, proc.stderr.decode()))
    with tarfile.open(tar_path) as tf:
        tf.extractall(dest)
    tar_path.unlink()


def copy_working_tree(repo: Path, dest: Path) -> dict[str, int]:
    """Everything that decides behaviour, including uncommitted and untracked work."""
    counts = {"files": 0, "dirs": 0}

    def ignore(directory: str, names: list[str]) -> set[str]:
        drop = {n for n in names if n in SKIP_DIRS}
        return drop

    shutil.copytree(repo, dest, ignore=ignore, symlinks=True, dirs_exist_ok=True)
    for path in dest.rglob("*"):
        if path.is_dir():
            counts["dirs"] += 1
        else:
            counts["files"] += 1
    return counts


def link_node_modules(repo: Path, side: Path) -> bool:
    """Share the installed dependency tree read-only rather than reinstalling it.

    Only safe when the dependency set is identical on both sides; the caller
    checks that and skips the link when it is not.
    """
    src = repo / "tv-break-dashboard" / "node_modules"
    if not src.is_dir():
        return False
    target = side / "tv-break-dashboard" / "node_modules"
    if target.exists() or not target.parent.is_dir():
        return target.exists()
    target.symlink_to(src, target_is_directory=True)
    return True


def dependency_sets_match(repo: Path, ref_dir: Path) -> bool:
    a = repo / "tv-break-dashboard" / "package.json"
    b = ref_dir / "tv-break-dashboard" / "package.json"
    if not (a.is_file() and b.is_file()):
        return False
    return a.read_bytes() == b.read_bytes()


def materialise(repo: Path, reference: str, keep: bool, need_work_copy: bool) -> Materialised:
    root = Path(tempfile.mkdtemp(prefix="gauntlet-verify-"))
    m = Materialised(root, reference, keep)
    m.scratch.mkdir(parents=True, exist_ok=True)
    export_reference(repo, reference, m.ref)
    if need_work_copy:
        copy_working_tree(repo, m.work)
    return m


def isolated_env(scratch: Path, extra: dict[str, str] | None = None) -> dict[str, str]:
    """Point every writable store the app knows about at throwaway space.

    This does not make a run read-only, it makes its writes land somewhere that
    does not matter. Anything the app writes by a hard-coded path still lands
    inside the copied tree, which is why the copy exists.
    """
    env = dict(os.environ)
    env.update({
        "KAIROS_AUTH_DISABLED": "1",
        "KAIROS_AUTH_DIR": str(scratch / "auth"),
        "KAIROS_VERSIONS_DIR": str(scratch / "versions"),
        "KAIROS_AUDIT_DIR": str(scratch / "audit"),
        "KAIROS_ASSISTANT_DATA_DIR": str(scratch / "assistant"),
        "KAIROS_ASSISTANT_USE_CLAUDE_CODE_OAUTH": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    for key in ("auth", "versions", "audit", "assistant"):
        (scratch / key).mkdir(parents=True, exist_ok=True)
    if extra:
        env.update(extra)
    return env
