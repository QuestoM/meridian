"""P9: the money on the approval card prints the basis it was computed on.

Measured before this round on two stored batches, ``c2799448`` and
``226a039107ad``, read back through ``GET /api/assistant/proposals``: the item's
effect carried ``['before','after','delta']`` and the card printed three money
lines under "מה השינוי הזה יעשה" with no channel, no date and no statement that
the figures are one representative day. Kai's own answer text volunteered the
basis; the surface where the operator presses Apply did not.

The simulation optimizes one representative day of the owned channel on both
sides, so a reader who is not told that reads a one-day figure as a weekly one.
The basis now travels with the money: ``settings_effect`` returns the channel
and the day it ran on, the proposal item carries them beside the money block,
and the card prints them above the first figure.

The money block keeps exactly the three keys ``tests/test_assistant_simulate.py``
pins by equality, which is why the basis rides beside the effect rather than
inside it: a surface gained a disclosure and no other piece's assertion moved.

Nothing here is mocked away. The simulation runs the real owned-channel scenario
runner on the repository's saved data, against a copy of the real settings file
so no test writes the tracked one.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from kairos_api import (
    assistant,
    assistant_actions as actions,
    assistant_simulate as simulate,
    assistant_tools as tools,
    core,
)

ROOT = Path(__file__).resolve().parents[1]
KAI = ROOT / "tv-break-dashboard" / "src" / "kai"
ISO_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@pytest.fixture()
def real_settings(tmp_path, monkeypatch) -> Any:
    """A copy of the saved settings, so the owned channel is the real one."""
    target = tmp_path / "kairos_settings.json"
    shutil.copy(ROOT / "data" / "kairos_settings.json", target)
    monkeypatch.setattr(core, "SETTINGS_PATH", target)
    monkeypatch.setenv(actions.DATA_DIR_ENV, str(tmp_path / "assistant"))
    return core._load_settings()


def _changed_floor() -> float:
    """One value, shared by every test here, so the priced sides are memoized once."""
    saved = core._load_settings().min_retention_floor
    return 0.85 if abs(saved - 0.85) > 1e-9 else 0.80


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(assistant.router)
    return TestClient(app)


# --- the primitive names what it ran on ---------------------------------------
def test_the_simulated_effect_names_the_channel_and_day_it_ran_on(real_settings) -> None:
    effect = simulate.settings_effect({"min_retention_floor": _changed_floor()})

    assert set(effect) == {"channel", "day", "before", "after", "delta"}
    assert effect["channel"] == real_settings.operator_channel
    assert ISO_DAY.match(str(effect["day"])), "the basis is one dated day, not a week"
    # And it is the same day the simulation itself reports, not a second guess.
    sim = simulate.simulate_settings_change({"min_retention_floor": _changed_floor()})
    assert (sim["channel"], sim["day"]) == (effect["channel"], effect["day"])


# --- the item splits the money from its basis ---------------------------------
def test_the_item_carries_the_basis_beside_money_keys_it_may_not_move(real_settings) -> None:
    item = tools.build_proposal_item(
        "propose_settings_change",
        {"changes": {"min_retention_floor": _changed_floor()}, "reason": "basis test"},
    )

    # The money block is exactly what the shared contract suite pins by equality.
    assert set(item["effect"]) == {"before", "after", "delta"}
    assert item["effect_basis"] == {"channel": real_settings.operator_channel,
                                    "day": simulate.settings_effect(
                                        {"min_retention_floor": _changed_floor()})["day"]}


def test_the_basis_survives_the_store_and_the_read_back(real_settings) -> None:
    """The route the defect was measured on: a batch read back after storage."""
    item = tools.build_proposal_item(
        "propose_settings_change",
        {"changes": {"min_retention_floor": _changed_floor()}, "reason": "basis test"},
    )
    batch = actions.create_batch("מה יקרה אם", [item], "tester", "test-model")

    body = _client().get("/api/assistant/proposals").json()
    stored = {row["batch_id"]: row for row in body["batches"]}[batch["batch_id"]]
    read_back = stored["items"][0]
    assert read_back["effect_basis"]["channel"] == real_settings.operator_channel
    assert ISO_DAY.match(read_back["effect_basis"]["day"])
    # The competitor boundary: the basis names the operator's own channel only.
    assert json.dumps(read_back, ensure_ascii=False).count(real_settings.operator_channel) >= 1


def test_an_unavailable_simulation_carries_no_basis_rather_than_a_guess(
    real_settings, monkeypatch
) -> None:
    monkeypatch.setattr("kairos_api.dashboard_api._owned_scope", lambda settings: (None, None))
    item = tools.build_proposal_item(
        "propose_settings_change", {"changes": {"revenue_weight": 61}, "reason": "no scope"})

    assert item["effect"]["status"] == "unavailable"
    assert "channel" in item["effect"]["reason"]
    assert "effect_basis" not in item, "no scope means no basis, never an empty one"


# --- the surface prints it -----------------------------------------------------
def test_the_card_prints_the_basis_above_the_first_figure() -> None:
    source = (KAI / "AssistantEffectView.jsx").read_text(encoding="utf-8")
    priced = source.split("if (!rows.length && !breaks.shown) return null;", 1)[1]
    assert "<EffectBasis" in priced
    assert priced.index("<EffectBasis") < priced.index("<EffectRow"), "the basis is read first"

    card = (KAI / "AssistantProposalCard.jsx").read_text(encoding="utf-8")
    assert "basis={item.effect_basis}" in card
    assert "function EffectView" not in card, "one effect view in the tree, not two that drift"


def test_the_basis_line_is_hebrew_and_says_representative_day_not_week() -> None:
    source = (KAI / "AssistantEffectView.jsx").read_text(encoding="utf-8")
    assert "הסימולציה רצה על יום-ערוץ מייצג אחד (${channel}, ${day}), לא על הסך השבועי." in source
    # And an item stored before the basis existed says which part is missing and
    # how to get it, instead of printing a channel and a date nobody recorded.
    assert "if (!channel || !day)" in source
    assert "הערוץ והתאריך לא נשמרו בפריט הזה." in source
    assert "אפשר לבקש מקאי את ההצעה מחדש כדי לקבל אותם." in source


def test_the_panel_normalizer_keeps_the_basis_with_the_money_it_qualifies() -> None:
    """Measured in a browser before this line existed: the API carried the basis,
    the card printed the unknown state anyway, because normalizeBatch whitelists
    item keys and the new one was not on the list."""
    state = (KAI / "assistant-panel-state.js").read_text(encoding="utf-8")
    normalizer = state.split("export function normalizeBatch", 1)[1].split("export function", 1)[0]
    assert "effect_basis:" in normalizer
    # And a merge of a later read over an earlier one keeps it, exactly as it
    # keeps the effect, so a refresh cannot strip the basis off stored money.
    merge = state.split("const mergeBatches", 1)[1].split("setBatchOrder", 1)[0]
    assert "basisByKey" in merge and "effect_basis: item.effect_basis" in merge


def test_the_missing_basis_is_not_the_faintest_thing_on_the_card() -> None:
    """Measured in a browser at 10px against the card's own background: --subtle
    scored 4.61:1 and --muted 5.75:1, so the first treatment set the one line a
    reader most needs to read the hardest to read. It now carries this file's
    own honest-unavailable language, the single inline-start amber edge, at
    6.22:1, and the edge is logical so it lands on the right in Hebrew."""
    css = (KAI / "assistant-console.css").read_text(encoding="utf-8")
    rule = css.split(".asst-effect-basis.unknown {", 1)[1].split("}", 1)[0]
    assert "var(--subtle)" not in rule
    assert "var(--amber-ink)" in rule
    assert "border-inline-start" in rule and "border-left" not in rule
