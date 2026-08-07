"""The training line: adoption writes under models/, so nobody may reach it.

Section 4.1 of the specification states the test in one sentence: an act is
training if and only if its output is a file under ``models/``. Adoption writes
``models/tv_break_coefficients.json``, so it is training, so it is company staff
only and it may not be reachable from any surface at all.

This suite proves both halves rather than asserting them. It measures what the
adoption act writes, so the classification is a fact and not a claim, and it
searches every route the application publishes and every file the operator's
interface ships for a way in.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from scripts import adopt_candidate_adoption as adoption
from scripts import adopt_candidate_rescore as rescore

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "tv-break-dashboard" / "src"
BACKEND = ROOT / "kairos_api"

# Every name a surface would have to say in order to reach this act.
REACHABLE_NAMES = ("adopt_candidate", "adopt-candidate", "adoptCandidate")


def _tree(tmp_path):
    artifact = {"method": "measured_detrended_pooled",
                "metadata": {"computed_at": "2026-08-01T00:00:00+00:00",
                             "source_fingerprints": {"a": "1"}, "first_break_multiplier": 1.0},
                "coefficients": {"a": -0.05},
                "detail": {"a": {"coefficient": -0.05, "ci_low": -0.08, "ci_high": -0.02, "n": 9}}}
    (tmp_path / "models" / "candidates").mkdir(parents=True)
    (tmp_path / "models" / "releases").mkdir(parents=True)
    (tmp_path / "data").mkdir(parents=True)
    (tmp_path / "models" / "tv_break_coefficients.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (tmp_path / "models" / "candidates" / "tv_break_coefficients_twin.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return rescore.Paths(root=tmp_path)


def _listing(root):
    return {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}


def test_every_file_the_adoption_act_writes_lands_under_models(tmp_path, monkeypatch):
    """The classification is measured, not asserted. Run it and look.

    If a later change made this act write anywhere else, it would either stop
    being training or start being two things at once, and both are worse than
    the rule.
    """
    import pandas as pd

    paths = _tree(tmp_path)
    frame = pd.DataFrame({"channel_name": ["a"] * 20, "log_effect": [-0.05] * 20,
                          "break_start": pd.date_range("2024-11-01", periods=20, freq="h")})
    rescore.save_rescore(rescore.rescore(paths, frame), paths)
    monkeypatch.setattr(adoption, "live_version", lambda: {"id": "mv-1", "name": "n"})
    monkeypatch.setattr(adoption, "ship_decision", lambda i, v: {"decision_id": "md-1"})
    monkeypatch.setattr(adoption, "money_state", lambda i: {
        "state": "measured", "revenue_delta": 0.0, "moved_fields": [], "scope": {"rows": 1}})

    before = _listing(tmp_path)
    plan = adoption.adopt("twin", adopted_by="steward", reason="testing",
                          paths=paths, perform=True)
    assert plan["outcome"] == "adopted"
    written = _listing(tmp_path) - before
    assert written, "the adoption wrote nothing, so this test proves nothing"
    assert all(path.startswith("models/") for path in written), sorted(written)

    adoption.revert(plan["adoption_id"], reverted_by="steward", reason="undo",
                    paths=paths, perform=True)
    assert all(path.startswith("models/") for path in _listing(tmp_path) - before)


def test_no_backend_module_imports_the_adoption_act():
    """A route cannot call what no module in the API package can import."""
    offenders = []
    for path in sorted(BACKEND.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        if any(name in text for name in REACHABLE_NAMES):
            offenders.append(path.name)
    assert offenders == []


def test_no_file_the_operator_interface_ships_names_the_adoption_act():
    offenders = []
    for path in sorted(FRONTEND.rglob("*")):
        if path.suffix not in (".js", ".jsx", ".css", ".html") or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if any(name in text for name in REACHABLE_NAMES):
            offenders.append(path.relative_to(FRONTEND).as_posix())
    assert offenders == []


def test_no_route_the_application_publishes_is_handled_out_of_the_scripts_directory():
    """The strongest form of the check: ask the running application itself."""
    from kairos_api.server import app

    offenders = []
    for route in app.routes:
        endpoint = getattr(route, "endpoint", None)
        module = getattr(endpoint, "__module__", "") if endpoint is not None else ""
        if module.startswith("scripts"):
            offenders.append(f"{getattr(route, 'path', '?')} -> {module}")
    assert offenders == []


def test_the_published_route_table_offers_no_adoption_path():
    from kairos_api.server import app

    paths = [getattr(route, "path", "") for route in app.routes]
    assert [path for path in paths if re.search(r"adopt", path, re.IGNORECASE)] == []


def test_the_entry_point_states_that_it_is_training_and_company_only():
    """The words are load-bearing here, because this file is the only surface."""
    text = (ROOT / "scripts" / "adopt_candidate.py").read_text(encoding="utf-8")
    assert "company staff only" in text
    assert "models/" in text


@pytest.mark.parametrize("module", ["adopt_candidate", "adopt_candidate_rescore",
                                    "adopt_candidate_registry", "adopt_candidate_adoption",
                                    "adopt_candidate_surface"])
def test_every_module_of_this_piece_stays_under_the_file_size_cap(module):
    lines = (ROOT / "scripts" / f"{module}.py").read_text(encoding="utf-8").splitlines()
    assert len(lines) <= 450, f"{module}.py is {len(lines)} lines"


@pytest.mark.parametrize("module", ["adopt_candidate", "adopt_candidate_rescore",
                                    "adopt_candidate_registry", "adopt_candidate_adoption",
                                    "adopt_candidate_surface"])
def test_no_module_of_this_piece_carries_an_em_dash_an_emoji_or_an_exclamation(module):
    text = (ROOT / "scripts" / f"{module}.py").read_text(encoding="utf-8")
    assert "—" not in text
    assert "!" not in text.replace("!=", "")
    assert not re.search(r"[\U0001F300-\U0001FAFF☀-➿]", text)
