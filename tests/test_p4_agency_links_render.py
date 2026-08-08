"""P4: the agency record renders the advertiser links the endpoint answers.

The measured defect. ``GET /api/agencies/AGY_01/advertisers`` answers HTTP 200
with six Hebrew advertiser names under the keys ``observed``, ``manual`` and
``effective``, and the drawer on the same screen printed "אין עדיין מפרסמים
המקושרים לסוכנות זו". Not a load failure: the response was 200 and the section
has its own error branch, so the ready-and-empty branch fired on a full payload.
The cause was one function, ``normalizeLinks``, which read only ``payload.links``
or ``payload.advertisers``, neither of which the shipped endpoint sends. Blast
radius, measured agency by agency: 6, 19, 3, 6, 1, 2, 2, 1 and 1, which is the
41 observed links the regression row in section 8.4 names. The client tree
rendered those same 41 correctly, so the destination contradicted itself between
two of its own tabs.

A helper-level assertion could not have caught it and cannot guard it, because
the number a person sees is produced by the component. So this file bundles the
shipped section with the bundler the product builds with, renders it with React,
and reads the names, the count and the provenance chips a person would see, from
the real payload the real store produces.

The last test restores the two-key reader into the shipped helper and asserts
the names vanish and the empty sentence returns, so a pass here can never be
vacuous.
"""

from __future__ import annotations

import html
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "tv-break-dashboard"
CLIENTS = APP / "src" / "clients"
HELPERS = CLIENTS / "agencies-helpers.js"
SECTION = CLIENTS / "AgencyLinkedAdvertisers.jsx"

# The number the regression row names, and the file the observations come from.
OBSERVED_LINKS = 41
AGY_01 = [
    "בנק הפועלים",
    "המרכז למימוש זכויות רפואיות",
    "כלמוביל",
    "מגדל",
    "פריסבי",
    "קרסו מוטורס",
]

# Isolation has one home, tv-break-dashboard/src/shell/bidi.jsx, and the shapes
# below are what it paints. Both are corrections rather than renames, so do not
# put the old ones back.
#
# A dir attribute on an inline run IS the defect. It fixes the run's internal
# order, which is wanted, and it also re-anchors that element's own alignment,
# which is a bug inside a right-to-left surface: a figure carrying dir="ltr"
# aligns left while its neighbours align right and the column stops lining up.
# The primitives isolate through a CSS class and never touch alignment, so a run
# they paint carries a class and no dir at all.
#
# U+2068 is the FIRST STRONG isolate: the run takes its direction from its own
# first strong character, so one call is right for a Hebrew name and a Latin
# one. U+2066 is the left-to-right isolate and would force a Hebrew name to read
# left to right, so its presence anywhere in this markup is the old defect.
#
# Written as escapes on purpose. The characters render as nothing, so a literal
# pair in this file would be invisible to review and to any editor that trims it.
FIRST_STRONG_ISOLATE = "\u2068"
POP_DIRECTIONAL_ISOLATE = "\u2069"
LEFT_TO_RIGHT_ISOLATE = "\u2066"
RIGHT_TO_LEFT_ISOLATE = "\u2067"

FIGURE_CLASS = "bidi-figure"
NAME_CLASS = "bidi-name"

OBSERVED_CHIP = "נצפה בנתונים"
MANUAL_CHIP = "קישור ידני"
LOAD_FAILED = "קישורי המפרסמים לא נטענו. זהו כשל טעינה, לא רשימה ריקה."
NO_FILE = "לא טעון קובץ ספוטים יומי, ולכן לא ניתן לקרוא קישורים נצפים. אפשר לטעון אותו בעמוד הנתונים."

# The precedence the fix introduced, and the reader that shipped before it. The
# mutation restores the old one exactly.
SHIPPED_CHOICE = """  let raw;
  if (Array.isArray(envelope.effective)) {
    raw = envelope.effective;
  } else if (Array.isArray(envelope.observed) || manualNames.size > 0) {
    raw = [...(envelope.observed || []), ...(envelope.manual || [])];
  } else if (Array.isArray(payload)) {
"""
TWO_KEY_READER = """  let raw;
  if (Array.isArray(payload)) {
"""

ENTRY = """
export {{ LinkedAdvertisers }} from '{section}';
export * as helpers from '{helpers}';
"""

# One node run: bundle the shipped section where it lives so its own imports
# resolve as they do in the browser bundle, then render each case and report the
# markup. React and the icon set come from the application's own install.
RENDER = """
import { createRequire, registerHooks } from 'node:module';
import { pathToFileURL } from 'node:url';
import fs from 'node:fs';

const [entry, outDir, casesFile, outFile, helpersSource] = process.argv.slice(2);
const require_ = createRequire('APP_PACKAGE');
const MAP = {};
for (const bare of ['react', 'react/jsx-runtime', 'react-dom/server', 'lucide-react', 'rolldown']) {
  MAP[bare] = pathToFileURL(require_.resolve(bare)).href;
}
registerHooks({
  resolve(specifier, context, nextResolve) {
    if (MAP[specifier]) {
      return { url: MAP[specifier], shortCircuit: true };
    }
    return nextResolve(specifier, context);
  },
});

const { build } = await import('rolldown');
await build({
  input: entry,
  external: ['react', 'react-dom', 'react/jsx-runtime', 'lucide-react'],
  output: { dir: outDir, format: 'esm', entryFileNames: 'surface.mjs' },
  resolve: { extensions: ['.js', '.jsx'] },
  logLevel: 'silent',
  plugins: [{
    name: 'agency-helpers-under-test',
    load(id) {
      return id === 'HELPERS_PATH' ? fs.readFileSync(helpersSource, 'utf8') : null;
    },
  }],
});

const React = (await import('react')).default;
const { renderToStaticMarkup } = await import('react-dom/server');
const surface = await import(pathToFileURL(`${outDir}/surface.mjs`).href);
const cases = JSON.parse(fs.readFileSync(casesFile, 'utf8'));

const rendered = {};
for (const [name, item] of Object.entries(cases)) {
  // Exactly what the drawer does with a response body.
  const links = item.status === 'ready' ? surface.helpers.normalizeLinks(item.payload) : [];
  const sourceFile = item.status === 'ready' ? surface.helpers.linksSourceFile(item.payload) : null;
  const state = { status: item.status, links, sourceFile };
  rendered[name] = {
    html: renderToStaticMarkup(React.createElement(surface.LinkedAdvertisers, { state, locale: 'he' })),
    names: links.map((link) => link.advertiser),
    sources: links.map((link) => link.source),
  };
}
fs.writeFileSync(outFile, JSON.stringify(rendered), 'utf8');
"""


def _as_text(value: str) -> str:
    """A name as React writes it into markup.

    Five of the 41 carry a quote or an ampersand (סיימן בע"מ, צ'מפיון מוטורס,
    עמותת מל"י, פרוקטר & גמבל, קופ"ח מאוחדת), which React escapes in a text
    node and the browser shows as typed. Comparing raw would fail on the escape
    and hide whether the name rendered at all.
    """
    return html.escape(value)


def _runs(markup: str, class_name: str) -> list[str]:
    """Every opening tag in the painted markup that the primitive produced."""
    return re.findall(rf'<[a-z]+ class="{class_name}[^"]*"[^>]*>', markup)


def _assert_isolated_without_a_dir(markup: str, class_name: str, count: int) -> None:
    """The painted shape of an isolated run: the class, and no dir attribute.

    Both halves matter. Without the class the run is not isolated at all and its
    digits merge with the Hebrew around them. With a dir attribute it is
    isolated and also re-anchored, which is the column-misalignment defect the
    primitive was written to remove.
    """
    tags = _runs(markup, class_name)
    assert len(tags) == count, f"{len(tags)} runs carry {class_name}, expected {count}"
    for tag in tags:
        assert "dir=" not in tag, f"{tag} re-anchors its own alignment inside a Hebrew surface"


def _node() -> str:
    found = shutil.which("node")
    if not found:
        pytest.skip("node is not on PATH, so the shipped section cannot be rendered here")
    probe = subprocess.run(
        [found, "-e", "const m = require('node:module'); process.stdout.write(typeof m.registerHooks)"],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.stdout.strip() != "function":
        pytest.skip("this node has no module.registerHooks, so react cannot be resolved from the app")
    if not (APP / "node_modules" / "react-dom").is_dir():
        pytest.skip("the dashboard's node_modules is not installed, so nothing can be rendered")
    if not (APP / "node_modules" / "rolldown").is_dir():
        pytest.skip("the bundler the product builds with is not installed")
    return found


@pytest.fixture(scope="module")
def payloads() -> dict:
    """The real response body of every agency, from the shipped store."""
    from kairos_api.agencies import _load_frame
    from kairos_api.agency_conditions import links_for

    ids = [str(value) for value in _load_frame()["agency_id"].tolist()]
    return {agency_id: links_for(agency_id) for agency_id in ids}


def _cases(payloads: dict) -> dict:
    """One render case per agency, plus the three states that are not a list."""
    cases = {agency_id: {"status": "ready", "payload": payload} for agency_id, payload in payloads.items()}
    first = payloads["AGY_01"]
    cases["error"] = {"status": "error", "payload": None}
    cases["empty_with_file"] = {
        "status": "ready",
        "payload": {"observed": [], "manual": [], "effective": [],
                    "observed_source_file": first["observed_source_file"]},
    }
    cases["empty_no_file"] = {
        "status": "ready",
        "payload": {"observed": [], "manual": [], "effective": [], "observed_source_file": None},
    }
    cases["manual_link"] = {
        "status": "ready",
        "payload": {"observed": [AGY_01[0]], "manual": ["מפרסם ידני"],
                    "effective": [AGY_01[0], "מפרסם ידני"],
                    "observed_source_file": first["observed_source_file"]},
    }
    return cases


def _render(tmp_path: Path, payloads: dict, helpers_source: str) -> dict:
    """Bundle and render the shipped section against one version of the helper."""
    node = _node()
    work = tmp_path / "surface"
    work.mkdir(parents=True, exist_ok=True)
    source = work / "helpers-under-test.js"
    source.write_text(helpers_source, encoding="utf-8")
    entry = work / "entry.mjs"
    entry.write_text(ENTRY.format(section=SECTION.as_posix(), helpers=HELPERS.as_posix()), encoding="utf-8")
    script = work / "render.mjs"
    script.write_text(
        RENDER.replace("APP_PACKAGE", (APP / "package.json").as_posix()).replace("HELPERS_PATH", HELPERS.as_posix()),
        encoding="utf-8",
    )
    cases = work / "cases.json"
    cases.write_text(json.dumps(_cases(payloads), ensure_ascii=False), encoding="utf-8")
    out = work / "out.json"
    result = subprocess.run(
        [node, str(script), str(entry), str(work / "bundle"), str(cases), str(out), str(source)],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(work),
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return json.loads(out.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def shipped() -> str:
    source = HELPERS.read_text(encoding="utf-8")
    assert SHIPPED_CHOICE in source, "the reader under test is not in the shipped helper any more"
    return source


@pytest.fixture(scope="module")
def rendered(tmp_path_factory, payloads, shipped) -> dict:
    return _render(tmp_path_factory.mktemp("agency-links"), payloads, shipped)


def test_the_endpoint_really_answers_forty_one_links_under_those_keys(payloads):
    """The state under test exists, or everything below would pass vacuously.

    The regression row counts the links the daily file evidences, so the count
    is taken over ``observed``. A link an operator adds by hand is a real link
    and rides in ``effective``, and counting those here would make onboarding a
    client, which this destination exists to do, read as a broken assertion.
    """
    assert set(payloads["AGY_01"]) == {"observed", "manual", "effective", "observed_source_file"}
    assert payloads["AGY_01"]["effective"] == AGY_01
    assert sum(len(payload["observed"]) for payload in payloads.values()) == OBSERVED_LINKS
    for agency_id, payload in payloads.items():
        invented = set(payload["effective"]) - set(payload["observed"]) - set(payload["manual"])
        assert not invented, f"{agency_id} answers a link that is neither observed nor manual"
    assert "links" not in payloads["AGY_01"] and "advertisers" not in payloads["AGY_01"]


def test_every_link_the_endpoint_answers_is_a_row_on_screen(payloads, rendered):
    """The regression row: the 41 observed advertiser links still render."""
    total = 0
    observed_on_screen = 0
    for agency_id, payload in payloads.items():
        names = rendered[agency_id]["names"]
        assert names == payload["effective"], f"{agency_id} renders a different set than it answers"
        for name in payload["effective"]:
            assert _as_text(name) in rendered[agency_id]["html"], f"{name} is not on screen under {agency_id}"
        observed_on_screen += len([name for name in names if name in set(payload["observed"])])
        total += len(names)
    assert observed_on_screen == OBSERVED_LINKS
    assert total == sum(len(payload["effective"]) for payload in payloads.values())


def test_the_six_names_the_critic_measured_are_on_the_first_agency(rendered):
    """The exact response the critic read, rendered."""
    html = rendered["AGY_01"]["html"]
    for name in AGY_01:
        # Isolation moved to the shell primitive, so a name paints as a class and
        # no dir. dir="auto" here would re-anchor the row's alignment.
        assert f'<span class="{NAME_CLASS} agz-link-name">{_as_text(name)}</span>' in html
    assert html.count(OBSERVED_CHIP) == len(AGY_01), "every observed link states its provenance"
    assert MANUAL_CHIP not in html, "no link here is manual, so no link may claim to be"


def test_a_full_payload_never_reaches_the_empty_state(payloads, rendered):
    """The defect itself: the ready and empty branch fired on six names."""
    for agency_id, payload in payloads.items():
        if not payload["effective"]:
            continue
        html = rendered[agency_id]["html"]
        assert "אין עדיין מפרסמים המקושרים לסוכנות זו" not in html
        assert "אף מפרסם בקובץ הספוטים היומי" not in html
        assert LOAD_FAILED not in html


def test_hebrew_advertiser_names_are_not_forced_left_to_right(rendered):
    """A Hebrew trade name forced left to right reads with its punctuation flipped.

    Read off the painted markup, not off the component source. Isolation moved
    into the shell primitive and each name now paints as an inline run carrying
    bidi-name and nothing else: a dir attribute on that run would set its base
    direction, which fixes the order and also re-anchors the row's alignment,
    and that second effect is the defect. U+2066 would force left to right on a
    name whose direction is its own, so it may not appear here either.
    """
    html = rendered["AGY_01"]["html"]
    _assert_isolated_without_a_dir(html, NAME_CLASS, len(AGY_01))
    assert LEFT_TO_RIGHT_ISOLATE not in html and RIGHT_TO_LEFT_ISOLATE not in html, (
        "a name is isolated by a direction-forcing mark rather than by first-strong"
    )


def test_the_section_states_its_count_and_the_file_it_read(payloads, rendered):
    """A name list with no basis cannot be checked against anything."""
    source_file = payloads["AGY_01"]["observed_source_file"]
    assert source_file, "the store must name the daily file, or the basis line is untestable"
    html = rendered["AGY_01"]["html"]
    # Isolation moved to the shell primitive: the count paints as a class, and a
    # dir attribute on it would pull the figure off the column it belongs to.
    assert f'<span class="{FIGURE_CLASS} numeric">{len(AGY_01)}</span>' in html
    _assert_isolated_without_a_dir(html, FIGURE_CLASS, 1)
    assert "<small>מפרסמים</small>" in html
    # The file name is joined into a sentence rather than rendered as its own
    # element, so it is isolated by the marks. First-strong, never left-to-right.
    assert (
        f"נצפו בקובץ הספוטים היומי {FIRST_STRONG_ISOLATE}{source_file}{POP_DIRECTIONAL_ISOLATE}." in html
    ), "the file name is a first-strong isolated run"
    assert LEFT_TO_RIGHT_ISOLATE not in html and RIGHT_TO_LEFT_ISOLATE not in html


def test_an_agency_holding_one_advertiser_reads_in_the_singular(payloads, rendered):
    """Three of the nine hold exactly one, and "1 מפרסמים" is nobody's sentence."""
    singles = [agency_id for agency_id, payload in payloads.items() if len(payload["effective"]) == 1]
    assert singles, "this data must contain a one-advertiser agency, or this proves nothing"
    for agency_id in singles:
        html = rendered[agency_id]["html"]
        # Isolation moved to the shell primitive, so the figure carries a class
        # and no dir. A dir here would re-anchor the count inside its own header.
        assert f'<span class="{FIGURE_CLASS} numeric">1</span><small>מפרסם</small>' in html
        assert "<small>מפרסמים</small>" not in html


def test_a_manual_link_is_labeled_manual_and_an_observed_one_is_not(rendered):
    """Provenance is per link, so the two kinds can never be read as one."""
    case = rendered["manual_link"]
    assert case["names"] == [AGY_01[0], "מפרסם ידני"]
    assert case["sources"] == ["observed", "manual"]
    assert MANUAL_CHIP in case["html"] and OBSERVED_CHIP in case["html"]


def test_the_three_states_that_are_not_a_list_each_say_which_one_they_are(rendered):
    """Real, unavailable and unknown, and never one wearing another's copy."""
    failure = rendered["error"]["html"]
    assert LOAD_FAILED in failure
    assert "agz-link-list" not in failure and "agz-link-count" not in failure

    empty = rendered["empty_with_file"]["html"]
    assert "אף מפרסם בקובץ הספוטים היומי" in empty, "an empty set names the file that was read"
    assert LOAD_FAILED not in empty

    unknown = rendered["empty_no_file"]["html"]
    assert NO_FILE in unknown, "no file loaded is a different state, and it names the path to supply one"
    assert "אף מפרסם בקובץ" not in unknown


def test_with_the_two_key_reader_restored_every_name_vanishes(tmp_path, payloads, shipped):
    """Proof the tests above bite: the defect, restored, fails them."""
    mutant = shipped.replace(SHIPPED_CHOICE, TWO_KEY_READER)
    assert mutant != shipped
    rendered = _render(tmp_path, payloads, mutant)
    for agency_id, payload in payloads.items():
        assert rendered[agency_id]["names"] == [], "this is exactly what was measured on the shipped bundle"
        html = rendered[agency_id]["html"]
        for name in payload["effective"]:
            assert _as_text(name) not in html
        assert "אף מפרסם בקובץ הספוטים היומי" in html, "and the empty state fired on a full payload, which is the defect"
        assert LOAD_FAILED not in html, "while the section's own error branch stayed silent"
