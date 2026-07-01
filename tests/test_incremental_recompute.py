"""Incremental recompute of the weekly break schedule.

Proves the three contracts of ``build_weekly_schedule(only_days=...)``:

  (i) rebuilding ONE channel-day incrementally produces a frame byte-identical
      (all columns, full CSV text) to a fresh FULL rebuild, against the real
      engine on the real committed reference data;
 (ii) untouched days' rows are preserved byte-identical from the prior CSV
      (proven with a tampered cell that a recompute would never produce);
(iii) classify_change maps a segment-scope override to exactly its channel-day
      and EVERYTHING else (settings included) to 'all'.

Plus the honest escape hatches: a missing CSV, a stale schema and an unknown
requested day each fall back to a FULL build. Runs under pytest or directly
with the venv python (``python tests/test_incremental_recompute.py``).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from kairos.data.loaders import load_programmes
from kairos.export.incremental import classify_change
from kairos.export.schedule import build_weekly_schedule

CHANNEL = "קשת 12"
DAY_A = "2024-11-01"
DAY_B = "2024-11-02"

_CACHE: dict[str, object] = {}


def _real_programmes_subset() -> pd.DataFrame:
    """Two real channel-days from the committed reference data."""
    if "programmes" not in _CACHE:
        frame = load_programmes()
        days = frame["start_dt"].dt.strftime("%Y-%m-%d")
        mask = (frame["Channel"].astype(str) == CHANNEL) & days.isin([DAY_A, DAY_B])
        subset = frame[mask].copy()
        assert not subset.empty, "reference data must carry the two test channel-days"
        _CACHE["programmes"] = subset
    return _CACHE["programmes"]


def _full_frame() -> pd.DataFrame:
    """One fresh FULL rebuild of the two channel-days (cached; deterministic)."""
    if "full" not in _CACHE:
        _CACHE["full"] = build_weekly_schedule(
            programmes=_real_programmes_subset(), revenue_weight=1.0,
        )
    return _CACHE["full"]


def _incremental(csv_path: Path, only_days, progress=None) -> pd.DataFrame:
    return build_weekly_schedule(
        programmes=_real_programmes_subset(), revenue_weight=1.0,
        only_days=only_days, existing_csv=csv_path,
        progress_cb=(None if progress is None else lambda d, t: progress.append((d, t))),
    )


def test_single_day_incremental_matches_full_rebuild() -> None:
    """(i) + (ii): recompute DAY_B only; the merged CSV equals a full rebuild."""
    full = _full_frame()
    pairs = sorted(set(zip(full["channel"], full["date"])))
    assert pairs == [(CHANNEL, DAY_A), (CHANNEL, DAY_B)]
    with tempfile.TemporaryDirectory() as td:
        csv_path = Path(td) / "weekly_break_schedule.csv"
        full.to_csv(csv_path, index=False, encoding="utf-8")
        progress: list = []
        merged = _incremental(csv_path, [(CHANNEL, DAY_B)], progress)
    assert merged.to_csv(index=False) == full.to_csv(index=False), (
        "incremental merge must be byte-identical to the full rebuild"
    )
    assert progress == [(1, 1)], "progress_cb must fire once for the one recomputed day"


def test_untouched_day_rows_come_from_the_prior_csv() -> None:
    """(ii) hard proof: a tampered untouched-day cell survives the merge verbatim,
    while the recomputed day's rows are fresh (equal to a full rebuild's)."""
    full = _full_frame()
    with tempfile.TemporaryDirectory() as td:
        csv_path = Path(td) / "weekly_break_schedule.csv"
        full.to_csv(csv_path, index=False, encoding="utf-8")
        tampered = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding="utf-8")
        day_a_index = tampered.index[tampered["date"] == DAY_A][0]
        tampered.loc[day_a_index, "program_type"] = "TAMPERED-PROGRAMME"
        tampered.to_csv(csv_path, index=False, encoding="utf-8")
        merged = _incremental(csv_path, [(CHANNEL, DAY_B)])
    day_a_merged = merged[merged["date"] == DAY_A].to_csv(index=False)
    day_a_tampered = tampered[tampered["date"] == DAY_A].to_csv(index=False)
    assert day_a_merged == day_a_tampered, (
        "untouched-day rows must be preserved verbatim from the prior CSV"
    )
    assert "TAMPERED-PROGRAMME" in set(merged["program_type"])
    day_b_merged = merged[merged["date"] == DAY_B].to_csv(index=False)
    day_b_full = full[full["date"] == DAY_B].to_csv(index=False)
    assert day_b_merged == day_b_full, "the recomputed day must carry fresh engine rows"


def test_missing_csv_falls_back_to_full_build() -> None:
    full = _full_frame()
    with tempfile.TemporaryDirectory() as td:
        progress: list = []
        frame = _incremental(Path(td) / "absent.csv", [(CHANNEL, DAY_B)], progress)
    assert frame.to_csv(index=False) == full.to_csv(index=False)
    assert progress == [(1, 2), (2, 2)], "the fallback full run reports full-run progress"


def test_stale_schema_falls_back_to_full_build() -> None:
    full = _full_frame()
    with tempfile.TemporaryDirectory() as td:
        csv_path = Path(td) / "weekly_break_schedule.csv"
        full.drop(columns=["segment_id"]).to_csv(csv_path, index=False, encoding="utf-8")
        frame = _incremental(csv_path, [(CHANNEL, DAY_B)])
    assert frame.to_csv(index=False) == full.to_csv(index=False)


def test_unknown_requested_day_falls_back_to_full_build() -> None:
    full = _full_frame()
    with tempfile.TemporaryDirectory() as td:
        csv_path = Path(td) / "weekly_break_schedule.csv"
        full.to_csv(csv_path, index=False, encoding="utf-8")
        frame = _incremental(csv_path, [("no-such-channel", DAY_B)])
    assert frame.to_csv(index=False) == full.to_csv(index=False)


def test_classify_change_segment_override_maps_to_exactly_its_day() -> None:
    result = classify_change(
        "override", {"scope": "segment", "target_id": f"{DAY_B}|{CHANNEL}|007"},
    )
    assert result == [(CHANNEL, DAY_B)]


def test_classify_change_everything_else_is_all() -> None:
    cases = [
        ("settings", {"revenue_weight": 80}),
        ("pricing", {"pricing_overrides": {}}),
        ("constraints", {}),
        ("coefficients", {}),
        ("reference", {}),
        ("some-future-kind", {}),
        ("", {}),
        ("override", {"scope": "spot", "target_id": "adv|camp|2024-11-01|3"}),
        ("override", {"scope": "segment", "target_id": "not-a-segment-id"}),
        ("override", {"scope": "segment", "target_id": "2024-13-45|ch|001"}),
        ("override", {"scope": "segment", "target_id": "2024-11-01|ch|xx"}),
        ("override", {"scope": "segment", "target_id": "2024-11-01||001"}),
        ("override", {"scope": "segment"}),
        ("override", None),
    ]
    for kind, payload in cases:
        assert classify_change(kind, payload) == "all", (kind, payload)


def main() -> int:
    tests = [
        test_classify_change_segment_override_maps_to_exactly_its_day,
        test_classify_change_everything_else_is_all,
        test_single_day_incremental_matches_full_rebuild,
        test_untouched_day_rows_come_from_the_prior_csv,
        test_missing_csv_falls_back_to_full_build,
        test_stale_schema_falls_back_to_full_build,
        test_unknown_requested_day_falls_back_to_full_build,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
