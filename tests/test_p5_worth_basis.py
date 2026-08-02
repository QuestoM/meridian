"""P5: the worth of a second is printed under the formula that made it.

A blind critic measured the shipped rate card printing `totals.yield_per_second`
as 142.0920 with the payload's top-level `basis.formula` directly beneath it
under the caption "How it is computed". That formula is
`frame_revenue_net`'s retention-cost model, its five named inputs produce
36,783,099.42, and nothing in it computes a rate per second. The figure was real
and its stated provenance was false, which is worse than an absent one: a reader
reconciling the number against the code would have started from the wrong
equation.

The defect was invisible to every test in this repository, because each half was
correct on its own. The endpoint returned a true figure and a true formula; the
surface printed both; only the sentence joining them was wrong. So this file
tests the JOIN. It reads the shipped surface, works out which payload path the
headline prints and which payload path the caption prints, resolves both against
the live payload, and requires the caption's formula to name the identifier of
the figure it sits under. A caption that describes a different quantity fails
here no matter how true it is elsewhere.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import kairos_api.core as core

ROOT = Path(__file__).resolve().parents[1]
SURFACE = ROOT / "tv-break-dashboard" / "src" / "rules" / "WorthOfASecond.jsx"

# The same pin the sibling rate-card file uses: the settings document on disk is
# writable by any client of PUT /api/settings, so the scope is fixed here rather
# than read from a file anything can move.
OPERATOR_CHANNEL = "רשת 13"

# A payload path as written in JSX: `totals.yield_per_second`,
# `state.basis?.formula`, `totals.basis.formula`.
PATH = re.compile(r"\b(state|totals)((?:\??\.[A-Za-z_][A-Za-z0-9_]*)+)")


def _settings_at(tmp_path: Path) -> Path:
    copy = tmp_path / "kairos_settings.json"
    shutil.copy(ROOT / "data" / "kairos_settings.json", copy)
    document = json.loads(copy.read_text(encoding="utf-8"))
    document["operator_channel"] = OPERATOR_CHANNEL
    copy.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    return copy


@pytest.fixture()
def payload(tmp_path, monkeypatch) -> dict:
    monkeypatch.setattr(core, "SETTINGS_PATH", _settings_at(tmp_path))
    from kairos_api.yield_api import router as yield_router

    app = FastAPI()
    app.include_router(yield_router)
    body = TestClient(app).get("/api/yield-per-second").json()
    if not body.get("available"):
        pytest.skip(f"no yield figure to check a basis against: {body.get('reason')}")
    return body


def _source() -> str:
    return SURFACE.read_text(encoding="utf-8")


def _element(source: str, tag: str, marker: str) -> str:
    """The inner text of the first ``tag`` element whose opening carries ``marker``."""
    opening = re.search(rf"<{tag}[^>]*{re.escape(marker)}[^>]*>", source)
    assert opening, f"no <{tag}> carrying {marker} on the surface"
    end = source.index(f"</{tag}>", opening.end())
    return source[opening.end():end]


def _paths(fragment: str) -> list[str]:
    return [f"{root}{rest}".replace("?", "") for root, rest in PATH.findall(fragment)]


def _resolve(body: dict, path: str):
    parts = path.split(".")
    node = body if parts[0] == "state" else body.get("totals")
    for part in parts[1:]:
        assert isinstance(node, dict), f"{path} is not a payload path"
        node = node.get(part)
    return node


def _basis_block(source: str) -> str:
    return _element(source, "p", 'className="rules-figure-basis"')


def _formula_paths(fragment: str) -> list[str]:
    """The distinct payload paths ending in ``formula`` that a fragment prints.
    A path guarded before it is printed appears twice in the source and is one
    formula on the screen, so the paths are deduplicated and the order kept."""
    seen: list[str] = []
    for path in _paths(fragment):
        if path.endswith(".formula") and path not in seen:
            seen.append(path)
    return seen


def test_the_caption_states_the_formula_of_the_figure_it_sits_under(payload):
    """The join. This is the assertion the shipped defect walked through."""
    source = _source()
    headline = _paths(_element(source, "strong", 'className="rules-worth-value"'))
    assert headline, "the headline figure is not read from the payload"
    figure_path = headline[0]
    identifier = figure_path.split(".")[-1]
    figure = _resolve(payload, figure_path)
    assert isinstance(figure, (int, float)), f"{figure_path} is not a number on the payload"

    formulas = _formula_paths(_basis_block(source))
    assert len(formulas) == 1, f"the caption prints {len(formulas)} formulas, expected exactly one"
    formula = _resolve(payload, formulas[0])
    assert isinstance(formula, str) and formula, f"{formulas[0]} carries no formula"

    assert identifier in formula, (
        f"the caption under {figure_path} prints {formulas[0]}, whose formula is "
        f"{formula!r} and never computes {identifier}"
    )


def test_the_caption_is_not_the_net_money_formula_that_sits_beside_it(payload):
    """The exact regression. The retention-cost basis is a true disclosure of a
    different quantity, and the shipped surface printed it under this one."""
    printed = _resolve(payload, _formula_paths(_basis_block(_source()))[0])
    net_basis = payload.get("basis") or {}
    if net_basis.get("formula"):
        assert printed != net_basis["formula"], (
            "the rate card is printing the retention-cost formula under the worth of a second again"
        )
        assert "yield_per_second" not in net_basis["formula"], (
            "the net-money formula now names the rate, so this test's premise needs re-measuring"
        )


def test_the_figure_can_be_checked_by_hand_on_the_screen_that_claims_it(payload):
    """Every input the basis names is a figure beside it, and dividing the two
    printed numbers reproduces the printed rate to its last printed digit."""
    totals = payload["totals"]
    basis = totals.get("basis")
    assert basis, "the headline figure carries no basis of its own"
    assert basis["formula"] == "yield_per_second = revenue / ad_seconds"
    assert set(basis["inputs"]) == {"revenue", "ad_seconds"}
    for name, note in basis["inputs"].items():
        assert name in totals, f"the basis names {name}, which is not a figure on this payload"
        assert note.strip(), f"{name} is named without a source"
    assert basis["source"] == "modeled"

    quotient = totals["revenue"] / totals["ad_seconds"]
    assert abs(quotient - totals["yield_per_second"]) < 5e-5, (
        f"{totals['revenue']} / {totals['ad_seconds']} is {quotient}, which is not the "
        f"printed {totals['yield_per_second']} to its last printed digit"
    )

    # And the surface prints that division substituted, not just the identifiers.
    basis_block = _paths(_basis_block(_source()))
    for path in ("totals.revenue", "totals.ad_seconds", "totals.yield_per_second"):
        assert path in basis_block, f"the caption does not print {path}, so nothing can be checked by hand"


def test_the_retention_cost_basis_keeps_the_figures_it_belongs_to(payload):
    """Section 8.4's P5 row, unchanged: the net money keeps its formula and its
    five named inputs, character for character, because Plan pins that string."""
    if not payload.get("revenue_net_available"):
        pytest.skip(f"net money unavailable on this plan: {payload.get('revenue_net_reason')}")
    basis = payload["basis"]
    assert basis["formula"] == (
        "retention_cost_ils = base_rate * baseline_tvr * (1 - retention_share) * "
        "(ad_seconds / unit_seconds); revenue_net_ils = revenue_ils - retention_cost_ils"
    )
    assert set(basis["inputs"]) == {
        "baseline_tvr", "retention_share", "base_rate", "ad_seconds", "unit_seconds",
    }
    for key in ("revenue_net_ils", "retention_cost_ils", "revenue_ils"):
        assert isinstance(payload[key], (int, float))
