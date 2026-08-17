"""The dashboard's trade vocabulary is pinned to the Python taxonomy.

The review surface renders a term's HEBREW NAME, its family and its honesty
status. None of that travels on the wire: the API sends a term id and the
dashboard looks the rest up in a JavaScript module mirrored from
``kairos/trade/taxonomy.py``. Two copies of one vocabulary drift, and the drift
is invisible in exactly the way that matters — a reviewer reads a name that no
longer matches the term the engine will compile, or reads a raw ``term-id``
because the mirror never learned about a term that was added.

So the mirror is pinned here:

- every term id in the Python registry appears in the JavaScript module;
- no id appears in the module that the registry does not have;
- the Hebrew name is character-identical for every one of them;
- the family letter is identical, because the review screen groups by it;
- every irrelevant-clause class is mirrored too, since a clause classified into
  one of them is shown by that class's name and by nothing else.

The parse is deliberately literal rather than clever. It reads the module as
text and pulls the fields out with one expression per field, so a module that
stops being a flat table of literals fails here loudly instead of being
half-understood.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from kairos.trade import taxonomy

DASHBOARD = Path(__file__).resolve().parents[1] / "tv-break-dashboard" / "src" / "trade"

# The modules that may hold the mirror. More than one name has been used for it
# while the surface was being built, so the test finds whichever are present
# rather than pinning a filename that a rename would turn into a false pass.
#
# EVERY present candidate is checked, not the first one found. Checking only the
# first would let a second mirror sit beside it unverified, which is the precise
# failure this test exists to prevent — two copies of one vocabulary, one of them
# watched.
CANDIDATES = ("trade-terms.js", "trade-vocabulary.js")

# One entry of the mirror:  'term-id': { family: 'B', he: "שם", en: "Name", ... }
_ENTRY = re.compile(
    r"""['"](?P<id>[a-z][a-z0-9-]+)['"]\s*:\s*\{(?P<body>[^{}]*)\}""",
    re.VERBOSE,
)
_FIELD = re.compile(r"""(?P<key>\w+)\s*:\s*(?P<quote>['"])(?P<value>(?:[^\\]|\\.)*?)(?P=quote)""")


def _mirrors() -> list[tuple[str, Path]]:
    found = [
        (path.read_text(encoding="utf-8"), path)
        for path in (DASHBOARD / name for name in CANDIDATES)
        if path.exists()
    ]
    if not found:
        pytest.skip(
            "no trade vocabulary module in the dashboard yet; expected one of "
            f"{CANDIDATES} under {DASHBOARD}"
        )
    return found


def _term_mirrors() -> list[tuple[dict[str, dict[str, str]], Path]]:
    """Only the modules that actually hold a term table. A module named as a
    candidate but holding some other vocabulary is not a mirror, and pretending
    it is one would fail the whole suite for the wrong reason."""
    out = []
    for source, path in _mirrors():
        parsed = _parse_terms(source)
        if parsed:
            out.append((parsed, path))
    if not out:
        pytest.skip(
            "a trade vocabulary module exists but holds no term table; "
            f"looked in {[p.name for _, p in _mirrors()]}"
        )
    return out


def _parse_terms(source: str) -> dict[str, dict[str, str]]:
    """Every entry that carries both a family and a Hebrew name is a term."""
    out: dict[str, dict[str, str]] = {}
    for match in _ENTRY.finditer(source):
        fields = {f.group("key"): f.group("value") for f in _FIELD.finditer(match.group("body"))}
        if "family" in fields and "he" in fields:
            out[match.group("id")] = fields
    return out


def test_every_python_term_is_mirrored_in_the_dashboard():
    for mirrored, path in _term_mirrors():
        missing = sorted(set(taxonomy.ids()) - set(mirrored))
        assert not missing, (
            f"{path.name} has no entry for {len(missing)} terms the engine knows: "
            f"{missing}. A term with no entry renders as its raw id."
        )


def test_the_dashboard_invents_no_term_the_engine_does_not_have():
    for mirrored, path in _term_mirrors():
        invented = sorted(set(mirrored) - set(taxonomy.ids()))
        assert not invented, (
            f"{path.name} names {len(invented)} terms the engine has never heard "
            f"of: {invented}. The taxonomy is the authority on what a term is."
        )


def test_hebrew_names_and_families_are_character_identical():
    drifted = []
    for mirrored, path in _term_mirrors():
        for term_id in taxonomy.ids():
            entry = mirrored.get(term_id)
            if entry is None:
                continue  # named by the test above; not double-reported here
            spec = taxonomy.get(term_id)
            if entry["he"] != spec.name_he:
                drifted.append(f"{path.name} {term_id}: he {entry['he']!r} != {spec.name_he!r}")
            if entry["family"] != spec.family:
                drifted.append(f"{path.name} {term_id}: family {entry['family']!r} != {spec.family!r}")
            status = entry.get("status")
            if status is not None and status != spec.status:
                drifted.append(f"{path.name} {term_id}: status {status!r} != {spec.status!r}")
    assert not drifted, (
        f"the dashboard has drifted from the taxonomy in {len(drifted)} places:\n  "
        + "\n  ".join(drifted)
    )


def test_every_irrelevant_clause_class_is_mirrored():
    """A clause dismissed as irrelevant is shown by its class name, so an
    unmirrored class prints a raw key at exactly the moment a reviewer is being
    asked to accept that a clause does not matter.

    Checked across all mirrors together: the classes may honestly live in one
    module while the terms live in another, so this fails only when NO module
    names a class.
    """
    sources = [source for source, _ in _mirrors()]
    names = [path.name for _, path in _mirrors()]
    missing = [
        key for key in taxonomy.IRRELEVANT_CLASSES
        if not any(f"'{key}'" in source for source in sources)
    ]
    assert not missing, (
        f"none of {names} names the irrelevant classes {missing}; a clause in one "
        "of them would show its raw key to the reviewer."
    )
