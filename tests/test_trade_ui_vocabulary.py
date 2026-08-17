"""The dashboard's term vocabulary is pinned to the Python registry.

``tv-break-dashboard/src/trade/trade-terms.js`` is the interface's twin of
``kairos.trade.taxonomy``: for every term it carries the Hebrew name a reviewer
reads, the family it is grouped under, the honesty status that decides whether
approving it will change behaviour, and the provenance rank that decides whether
the interface may present it as local market fact.

That file's own header claims this test pins the two together, and a claim in a
comment is not a rule. This makes it one.

WHY EACH FIELD IS PINNED, and not just the id list:

- **The name.** A reviewer signs an agreement by reading these words. A Hebrew
  name that drifts from the registry means the screen and the engine disagree
  about what a clause IS, silently.
- **The status.** BINDS versus REPRESENTABLE is the entire product claim on this
  surface: whether approval wires a clause into live machinery or holds it and
  does nothing. A stale status is the interface promising an effect the engine
  will not deliver, which is the one lie this surface exists to prevent.
- **The rank.** IL, TRADE and STD say whether a term is attested by an Israeli
  primary source, by the owner's trade transcript, or is standard practice not
  verified in this market. Interface copy must never assert an STD term as local
  fact, and it cannot honour that if the rank has drifted.
- **The family.** The review groups terms by family, so a drifted family puts a
  clause under the wrong heading in front of a commercial director.

The irrelevant-class list is pinned too: it is the closed set of clause kinds
the pipeline is allowed to call commercially irrelevant, and a class present on
one side only would let "irrelevant" quietly become a dumping ground.

The parser here is deliberately literal rather than a JavaScript engine: it
reads the object literals with a regular expression, and it FAILS if it cannot
find them, so a refactor that moves the tables cannot make this test pass by
matching nothing.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from kairos.trade import taxonomy

TERMS_JS = (
    Path(__file__).resolve().parents[1]
    / "tv-break-dashboard" / "src" / "trade" / "trade-terms.js"
)

# One entry of the TERMS table, e.g.
#   'agency-commission': { family: 'C', he: "עמלת סוכנות", en: "Agency commission",
#                          status: 'BINDS', rank: 'TRADE' },
TERM_ENTRY = re.compile(
    r"""^\s*'(?P<id>[a-z0-9-]+)':\s*\{\s*
        family:\s*'(?P<family>[A-Z]+)',\s*
        he:\s*"(?P<he>[^"]*)",\s*
        en:\s*"(?P<en>[^"]*)",\s*
        status:\s*'(?P<status>[A-Z_]+)',\s*
        rank:\s*'(?P<rank>[A-Z]+)'""",
    re.VERBOSE,
)

IRRELEVANT_ENTRY = re.compile(r"^\s*'(?P<key>[a-z0-9-]+)':\s*\"(?P<he>[^\"]*)\",")


def _block(source: str, opener: str) -> str:
    """The body of a top-level ``export const NAME = {...};`` block."""
    start = source.index(opener) + len(opener)
    depth = 1
    for offset, char in enumerate(source[start:], start=start):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[start:offset]
    raise AssertionError(f"{opener!r} is not closed in {TERMS_JS.name}")


@pytest.fixture(scope="module")
def js_source() -> str:
    assert TERMS_JS.exists(), f"the interface twin is missing: {TERMS_JS}"
    return TERMS_JS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def js_terms(js_source: str) -> dict[str, dict[str, str]]:
    body = _block(js_source, "export const TERMS = {")
    found = {
        match.group("id"): {
            "family": match.group("family"),
            "he": match.group("he"),
            "en": match.group("en"),
            "status": match.group("status"),
            "rank": match.group("rank"),
        }
        for match in (TERM_ENTRY.match(line) for line in body.splitlines())
        if match
    }
    # A parser that silently matches nothing would make every assertion below
    # vacuously true, so the parse itself is asserted first.
    assert len(found) > 50, (
        f"only {len(found)} terms parsed out of {TERMS_JS.name}; the table's shape "
        "changed and this pin stopped reading it"
    )
    return found


def test_every_registry_term_is_present_in_the_interface(js_terms):
    missing = sorted(set(taxonomy.ids()) - set(js_terms))
    assert not missing, (
        "the engine knows terms the review screen cannot name, so they would render "
        f"as raw ids to a reviewer: {missing}"
    )


def test_the_interface_invents_no_term(js_terms):
    extra = sorted(set(js_terms) - set(taxonomy.ids()))
    assert not extra, (
        "the review screen offers terms the engine does not hold; adding one of "
        f"these would be refused on save: {extra}"
    )


def test_names_families_statuses_and_ranks_all_match(js_terms):
    drift: list[str] = []
    for term_id in sorted(taxonomy.ids()):
        spec = taxonomy.get(term_id)
        entry = js_terms.get(term_id)
        if entry is None:
            continue  # reported by the presence test above
        for field, engine, screen in (
            ("name_he", spec.name_he, entry["he"]),
            ("name_en", spec.name_en, entry["en"]),
            ("family", spec.family, entry["family"]),
            ("status", spec.status, entry["status"]),
            ("rank", spec.rank, entry["rank"]),
        ):
            if str(engine) != str(screen):
                drift.append(f"{term_id}.{field}: engine {engine!r} != screen {screen!r}")
    assert not drift, "the interface twin has drifted from the registry:\n" + "\n".join(drift)


def test_the_status_and_rank_vocabularies_are_the_registry_s_own(js_terms):
    statuses = {entry["status"] for entry in js_terms.values()}
    ranks = {entry["rank"] for entry in js_terms.values()}
    assert statuses <= set(taxonomy.STATUSES), (
        f"the screen uses statuses the registry does not define: {sorted(statuses - set(taxonomy.STATUSES))}"
    )
    assert ranks <= set(taxonomy.RANKS), (
        f"the screen uses ranks the registry does not define: {sorted(ranks - set(taxonomy.RANKS))}"
    )


def test_every_status_has_copy_for_a_reviewer(js_source, js_terms):
    """A status with no sentence would render a term with no honesty note."""
    copy_block = _block(js_source, "const STATUS_COPY = {")
    for status in sorted({entry["status"] for entry in js_terms.values()}):
        assert f"{status}: {{" in copy_block, (
            f"status {status} is used by a term but has no reviewer copy, so that "
            "term would appear without the sentence saying what approving it does"
        )


def test_the_family_table_covers_every_family_in_use(js_source, js_terms):
    families_block = _block(js_source, "export const TERM_FAMILIES = {")
    for family in sorted({entry["family"] for entry in js_terms.values()}):
        assert re.search(rf"^\s*{family}:\s*\"", families_block, re.M), (
            f"family {family} groups terms on the review screen but has no heading"
        )


def test_the_irrelevant_class_list_matches_the_registry(js_source):
    body = _block(js_source, "export const IRRELEVANT_CLASSES = {")
    screen = {
        match.group("key")
        for match in (IRRELEVANT_ENTRY.match(line) for line in body.splitlines())
        if match
    }
    assert screen, "the irrelevant-class table could not be parsed"
    engine = set(taxonomy.IRRELEVANT_CLASSES)
    assert screen == engine, (
        "the closed set of commercially irrelevant clause kinds disagrees. Only these "
        "may be called irrelevant, so a mismatch either hides a class from the "
        f"reviewer or invents one: engine-only {sorted(engine - screen)}, "
        f"screen-only {sorted(screen - engine)}"
    )
