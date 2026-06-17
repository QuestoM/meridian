"""Tests for the automatic series-layer held-out gate.

Proves the two data-sufficiency contracts:

  (a) Thin data (single month / few airings per title): the gate fails and
      ``series_layer_active`` is False. The JSON written by the script has no
      series block, meaning load_impact_model falls back to the genre coefficient,
      which is today's behavior.

  (b) Rich data (many distinct airings where one series genuinely diverges from
      the genre mean): the gate passes and ``series_layer_active`` is True. The
      JSON carries the series block, and load_impact_model then returns a per-title
      coefficient that differs from the genre coefficient.

  (c) The metadata fields ``series_layer_active``, ``series_gate_holdout``, and
      ``series_gate_reason`` are always present in the JSON, so any reader can
      audit the decision.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from kairos.model.measure import MeasuredCoefficient, write_coefficients_json
from kairos.model.series import SeriesCoefficient
from kairos.model.series_gate import series_holdout_gate


# ---------------------------------------------------------------------------
# Synthetic effects-frame builders
# ---------------------------------------------------------------------------

def _make_effects(
    rows: list[tuple[str, str, float]],
    *,
    n_copies: int = 1,
) -> pd.DataFrame:
    """Build an effects frame from (channel_name, title, log_effect) tuples.

    ``n_copies`` repeats each row to build larger datasets.
    """
    records = [
        {"channel_name": cell, "title": title, "log_effect": effect}
        for cell, title, effect in rows
        for _ in range(n_copies)
    ]
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# (a) Thin data: gate must fail -> series_layer_active False
# ---------------------------------------------------------------------------


def test_gate_fails_on_empty_effects() -> None:
    """An empty effects frame -> gate fails safely."""
    result = series_holdout_gate(pd.DataFrame(columns=["channel_name", "title", "log_effect"]))
    assert result["series_layer_active"] is False
    assert "series_gate_reason" in result
    assert isinstance(result["series_gate_holdout"], dict)


def test_gate_fails_when_too_few_test_breaks() -> None:
    """Fewer than _MIN_TEST_BREAKS (10) test breaks -> gate abstains, series omitted."""
    # Only 4 rows total -> 20% split = <1 test break.
    effects = _make_effects([
        ("News_first_short", "Show A", -0.10),
        ("News_first_short", "Show A", -0.11),
        ("News_first_short", "Show B", -0.08),
        ("Other_last_long", "Show C", -0.05),
    ])
    result = series_holdout_gate(effects)
    assert result["series_layer_active"] is False
    assert "series_gate_reason" in result
    assert result["series_gate_holdout"]["n_test"] < 10


def test_gate_fails_on_data_where_series_does_not_help() -> None:
    """When all titles have the same retention as their genre cell, series adds nothing.

    Every airing within a genre cell has the same log_effect (the genre-cell
    mean), so series-level means equal genre means. The series RMSE is identical
    to the genre RMSE, and the gate must reject the layer.
    """
    # 200 rows, two genre cells, all titles in each cell identical to the cell mean.
    rows = (
        [("News_first_short", "Show A", -0.10)] * 100
        + [("Other_last_long", "Show B", -0.05)] * 100
    )
    effects = pd.DataFrame(
        [{"channel_name": c, "title": t, "log_effect": e} for c, t, e in rows]
    )
    result = series_holdout_gate(effects)
    # Series predictions equal genre predictions -> improvement is 0%, below threshold.
    assert result["series_layer_active"] is False
    holdout = result["series_gate_holdout"]
    assert holdout["genre_rmse"] is not None
    assert holdout["series_rmse"] is not None
    # RMSE should be near zero (all predictions match the cell mean exactly).
    assert holdout["genre_rmse"] == pytest.approx(holdout["series_rmse"], abs=1e-9)


# ---------------------------------------------------------------------------
# (b) Rich data: gate must pass -> series_layer_active True
# ---------------------------------------------------------------------------


def test_gate_passes_when_series_clearly_diverges() -> None:
    """When one series has a strongly different retention, series beats genre.

    Genre cell mean is near -0.05. Series X has log_effect near -0.20 (very
    different from the genre mean). With many airings the series mean is a much
    better predictor on the held-out set, so the gate must pass.
    """
    rng = np.random.default_rng(0)
    # 120 breaks of "Normal Show" near the genre mean.
    normal = [
        {"channel_name": "News_first_short", "title": "Normal Show",
         "log_effect": float(rng.normal(-0.05, 0.01))}
        for _ in range(120)
    ]
    # 80 breaks of "Big Loser" strongly diverging from the genre mean.
    loser = [
        {"channel_name": "News_first_short", "title": "Big Loser Show",
         "log_effect": float(rng.normal(-0.25, 0.01))}
        for _ in range(80)
    ]
    effects = pd.DataFrame(normal + loser)
    result = series_holdout_gate(effects)
    assert result["series_layer_active"] is True
    holdout = result["series_gate_holdout"]
    assert holdout["series_rmse"] < holdout["genre_rmse"]
    assert holdout["n_test"] >= 10


def test_gate_metadata_fields_always_present() -> None:
    """The three gate metadata fields are always present regardless of outcome."""
    for effects in [
        pd.DataFrame(columns=["channel_name", "title", "log_effect"]),
        _make_effects([("News_first_short", "Show A", -0.10)] * 3),
    ]:
        result = series_holdout_gate(effects)
        assert "series_layer_active" in result
        assert "series_gate_holdout" in result
        assert "series_gate_reason" in result
        holdout = result["series_gate_holdout"]
        assert "genre_rmse" in holdout
        assert "series_rmse" in holdout
        assert "n_test" in holdout


# ---------------------------------------------------------------------------
# (c) JSON write + load_impact_model round-trip
# ---------------------------------------------------------------------------


def test_write_coefficients_includes_series_gate_metadata(tmp_path) -> None:
    """write_coefficients_json persists series gate metadata in the JSON."""
    coeffs = {
        "News_first_short": MeasuredCoefficient(
            channel_name="News_first_short",
            coefficient=-0.05, raw_delta=-0.05, n=100,
            ci_low=-0.08, ci_high=-0.02,
        ),
    }
    gate_metadata = {
        "series_layer_active": True,
        "series_gate_holdout": {"genre_rmse": 0.10, "series_rmse": 0.07, "n_test": 40},
        "series_gate_reason": "series RMSE beats genre by 30%; series layer activated",
    }
    path = tmp_path / "coefficients.json"
    write_coefficients_json(path, coeffs, metadata=gate_metadata)
    payload = json.loads(path.read_text(encoding="utf-8"))
    meta = payload["metadata"]
    assert meta["series_layer_active"] is True
    assert meta["series_gate_holdout"]["genre_rmse"] == pytest.approx(0.10)
    assert "series layer activated" in meta["series_gate_reason"]


def test_load_impact_model_uses_series_coefficient_when_layer_active(tmp_path) -> None:
    """When the JSON carries a series block, load_impact_model returns per-title coefficients.

    This proves the end-to-end path: gate activates -> series block in JSON ->
    load_impact_model builds a PosteriorImpactModel with the series layer ->
    coefficient_for_title returns the series coefficient, not the genre one.
    """
    from kairos.model.impact import load_impact_model
    from kairos.model.measure import MeasuredCoefficient, write_coefficients_json
    from kairos.model.series import SeriesCoefficient

    coeffs = {
        "News_first_short": MeasuredCoefficient(
            channel_name="News_first_short",
            coefficient=-0.05, raw_delta=-0.05, n=100,
            ci_low=-0.08, ci_high=-0.02,
        ),
    }
    # A series that diverges strongly from the genre mean.
    series_coeff = SeriesCoefficient(
        channel_name="News_first_short",
        series_key="האח הגדול",
        coefficient=-0.15,
        raw_delta=-0.15,
        n=30,
        ci_low=-0.20,
        ci_high=-0.10,
    )
    series = {("News_first_short", "האח הגדול"): series_coeff}

    coeff_path = tmp_path / "tv_break_coefficients.json"
    write_coefficients_json(coeff_path, coeffs, series=series)

    # Load model from the path where the json sits.
    model = load_impact_model(
        tmp_path / "tv_break_posterior.pkl",  # pkl does not exist; falls back to json
        coefficients_path=coeff_path,
    )
    assert model.source == "measured"

    # Genre cell coefficient (no title).
    genre_coeff = model.coefficient_for("News", "first", "short")
    assert genre_coeff == pytest.approx(-0.05)

    # Series-aware coefficient (title canonicalizes to האח הגדול).
    assert hasattr(model, "coefficient_for_title"), "model must have coefficient_for_title"
    series_result = model.coefficient_for_title("האח הגדול עונה 3 פרק 5", "News", "first", "short")
    # Must differ from the genre mean.
    assert series_result != pytest.approx(genre_coeff)
    assert series_result == pytest.approx(-0.15)


def test_load_impact_model_falls_back_to_genre_when_no_series_block(tmp_path) -> None:
    """When the JSON has no series block, load_impact_model returns genre coefficients only.

    Proves the omit-series path: gate fails -> no series in JSON ->
    coefficient_for_title falls back to genre mean (honest cold-start).
    """
    from kairos.model.impact import load_impact_model
    from kairos.model.measure import MeasuredCoefficient, write_coefficients_json

    coeffs = {
        "News_first_short": MeasuredCoefficient(
            channel_name="News_first_short",
            coefficient=-0.05, raw_delta=-0.05, n=100,
            ci_low=-0.08, ci_high=-0.02,
        ),
    }
    coeff_path = tmp_path / "tv_break_coefficients.json"
    # Write WITHOUT a series block (gate failed).
    write_coefficients_json(coeff_path, coeffs, series=None)

    model = load_impact_model(
        tmp_path / "tv_break_posterior.pkl",
        coefficients_path=coeff_path,
    )
    genre_coeff = model.coefficient_for("News", "first", "short")
    assert genre_coeff == pytest.approx(-0.05)
    # With no series block, title-aware lookup must fall back to the genre coefficient.
    if hasattr(model, "coefficient_for_title"):
        title_coeff = model.coefficient_for_title(
            "האח הגדול עונה 3 פרק 5", "News", "first", "short"
        )
        assert title_coeff == pytest.approx(genre_coeff)
