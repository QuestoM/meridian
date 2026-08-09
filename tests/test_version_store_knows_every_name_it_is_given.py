"""A version call that names a file the store does not know versioned nothing.

``snapshot`` filtered its argument against ``_LOGICAL_ORDER`` and returned None
for anything left over. At every call site that read exactly like "nothing had
changed", so a caller passing a name the store never knew was indistinguishable
from a caller whose file was untouched.

MEASURED 2026-08-09 by reading every ``snapshot_manual_edit`` call in
``kairos_api``: thirteen call sites, ELEVEN passing a known name and TWO passing
a name the store had never heard of.

    campaigns_api_store.py:161   "campaigns"      -> versioned nothing
    target_store.py:266          "plan_targets"   -> versioned nothing

Both files are operator-editable through the dashboard, and the first one's own
docstring says "Version the campaigns store before a manual edit writes it". It
did not. A manual edit to either had no history to restore from, and the
restore screen would have shown the operator a timeline with their edit missing
rather than an error.

THE FIX IS IN TWO HALVES, because either alone is silent again.

The store now knows both names, so the two live sites work. And ``snapshot``
RAISES on a name it does not know instead of dropping it, so the next one is
loud. The manual-edit hook still swallows that raise deliberately, since a
history hiccup must never fail an operator's edit; the raise is for tests,
scripts and the assistant, which is why the guard below reads the tree rather
than trusting the runtime.

That last part is the point. A test listing today's thirteen call sites would go
quiet the day a fourteenth appeared, which is the exact failure this file
exists to close.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from kairos_api import version_store

ROOT = Path(__file__).resolve().parents[1]
API = ROOT / "kairos_api"

# Every literal name handed to the version store anywhere in the API layer.
_CALL = re.compile(r"snapshot_manual_edit\(\s*[^,)]+,\s*\"([^\"]+)\"\s*\)")
_SNAPSHOT_FILES = re.compile(r"snapshot\([^)]*files=\[([^\]]*)\]", re.S)


def _named_in_tree() -> dict[str, list[str]]:
    """Logical name -> the call sites that pass it, found by reading."""
    found: dict[str, list[str]] = {}
    for module in sorted(API.glob("*.py")):
        source = module.read_text(encoding="utf-8")
        for line_no, line in enumerate(source.splitlines(), start=1):
            for name in _CALL.findall(line):
                found.setdefault(name, []).append(f"{module.name}:{line_no}")
        for block in _SNAPSHOT_FILES.findall(source):
            for name in re.findall(r'"([^"]+)"', block):
                found.setdefault(name, []).append(module.name)
    return found


def test_every_logical_name_the_tree_passes_is_one_the_store_knows():
    """The class guard. Discovered by reading, so a new call site is covered."""
    named = _named_in_tree()
    assert named, "found no version-store call sites, so this guard checks nothing"
    unknown = {
        name: sites for name, sites in named.items()
        if name not in version_store._KNOWN_LOGICAL
    }
    assert not unknown, (
        "a caller names a logical file the version store does not know, so the edit "
        f"it guards is versioned nowhere and the call returns None silently: {unknown}"
    )


def test_the_restore_set_and_the_vocabulary_are_not_the_same_register():
    """The two names are capturable but are NOT part of a full restore, on purpose.

    Putting them in the restore set was the obvious move and is the wrong one: an
    operator restoring a settings version would then also revert campaign
    bookings, which is worse than the bug being fixed. A manual edit records a
    version holding only the file it touched.
    """
    assert set(version_store._LOGICAL_ORDER) < set(version_store._KNOWN_LOGICAL)
    added = set(version_store._KNOWN_LOGICAL) - set(version_store._LOGICAL_ORDER)
    # make_goods joined the vocabulary when the assistant gained a propose tool
    # over the pacing board, and it stays out of the restore set for exactly the
    # reason the other two do, only more so: the decision ledger records what a
    # channel owes a client, and rolling that back because somebody restored a
    # settings version would erase a debt nobody decided to erase.
    assert added == {"campaigns", "plan_targets", "make_goods"}
    for name in added:
        assert name not in version_store._LOGICAL_ORDER, (
            f"{name} joined the full restore set, so restoring any version now "
            "also rolls this file back"
        )


def test_the_two_that_were_broken_resolve_to_the_files_they_claim():
    """Named explicitly, because knowing a name is not the same as reading a file.

    Adding a string to a tuple satisfies the test above while ``_logical_path``
    still raises. These two assert the path resolves and points where the store
    that passes the name actually writes.
    """
    from kairos_api import campaigns_api_store, target_store

    assert version_store._logical_path("campaigns") == Path(campaigns_api_store.CAMPAIGNS_PATH)
    assert version_store._logical_path("plan_targets") == Path(target_store.TARGETS_PATH)
    # And the snapshot name a restore reads back carries the real suffix rather
    # than the .dat fallback, which is what an unresolvable path would produce.
    assert version_store._snapshot_name("campaigns") == "campaigns.csv"
    assert version_store._snapshot_name("plan_targets") == "plan_targets.csv"
    # And the third, for the same reason: the assistant's pacing proposals name
    # it, so a string in the tuple with no path behind it would raise on the
    # snapshot taken just before an approved decision is written.
    from kairos_api import makegood_store

    assert version_store._logical_path("make_goods") == Path(makegood_store.MAKE_GOODS_PATH)
    assert version_store._snapshot_name("make_goods") == "make_goods.csv"


def test_an_unknown_name_is_refused_rather_than_dropped():
    """The half that makes the next one loud instead of silent.

    The example name is deliberately fictional. This test used to name
    ``make_goods``, which stopped being unknown the day the assistant's pacing
    proposals needed it versioned, and a guard whose example quietly becomes
    valid stops testing the refusal it was written for. The assertion below
    keeps it fictional by construction.
    """
    fictional = "not_a_real_store_and_never_will_be"
    assert fictional not in version_store._KNOWN_LOGICAL
    with pytest.raises(ValueError) as raised:
        version_store.snapshot(source="test", actor="test", files=[fictional])
    assert "does not know" in str(raised.value)
    # Naming nothing at all is still a no-op rather than an error: a caller with
    # an empty list has asked for nothing, which is different from asking for
    # something that does not exist.
    assert version_store.snapshot(source="test", actor="test", files=[]) is None


def test_the_manual_edit_hook_still_never_fails_the_operators_edit(monkeypatch):
    """The raise must not reach the request, and this proves it does not.

    A history hiccup failing a save would be a worse defect than the one being
    fixed, so the hook keeps its swallow. This drives an unknown name straight
    through the hook and requires it to return quietly.
    """
    class _Request:
        cookies: dict[str, str] = {}
        headers: dict[str, str] = {}

    monkeypatch.setattr(version_store, "_actor", lambda request: "test")
    assert version_store.snapshot_manual_edit(_Request(), "not_a_logical_file") is None
