"""Impact API helpers should expose model effects without raw CSV artifacts."""

from __future__ import annotations

import json

from kairos_api.server import _load_impact, _load_measured_impact_summary


def test_load_impact_sanitizes_non_finite_csv_values(tmp_path) -> None:
    path = tmp_path / "impact.csv"
    path.write_text("segment,total,count,average\nNews,inf,9,\nOther,-0.1,3,-0.033\n", encoding="utf-8")

    rows = _load_impact(path)

    assert rows[0]["total"] is None
    assert rows[0]["average"] is None
    assert rows[1]["total"] == -0.1
    assert rows[1]["average"] == -0.033


def test_measured_impact_summary_groups_coefficients_for_dashboard(tmp_path) -> None:
    path = tmp_path / "tv_break_coefficients.json"
    payload = {
        "method": "measured_detrended_pooled",
        "metadata": {"total_breaks_measured": 30},
        "detail": {
            "News_first_short": {
                "channel_name": "News_first_short",
                "coefficient": -0.04,
                "raw_delta": -0.04,
                "n": 20,
                "ci_low": -0.06,
                "ci_high": -0.02,
            },
            "News_middle_long": {
                "channel_name": "News_middle_long",
                "coefficient": -0.02,
                "raw_delta": -0.02,
                "n": 10,
                "ci_low": -0.04,
                "ci_high": -0.01,
            },
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    summary = _load_measured_impact_summary(path)

    assert summary["source"] == "measured_detrended_pooled"
    assert summary["metadata"]["total_breaks_measured"] == 30
    assert summary["program_type"][0]["segment"] == "News"
    assert summary["program_type"][0]["average_coefficient"] == -0.033333
    assert summary["program_type"][0]["sample_count"] == 30
    assert {row["segment"] for row in summary["position"]} == {"first", "middle"}
    assert {row["segment"] for row in summary["length"]} == {"short", "long"}
