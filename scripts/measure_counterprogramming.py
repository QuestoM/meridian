"""Measure the counter-programming covariate on the real month, end to end.

Produces every number the adoption decision needs:

1. The fitted within-cell betas for the three FORWARD features (rival audience
   strength, same-genre contrast, rival programme start near the window) and
   the training-only control, with standard errors and 95 percent intervals.
2. The held-out gate: out-of-sample RMSE WITH vs WITHOUT the covariate
   (:mod:`kairos.model.competitor_gate`), which is the adoption criterion.
3. A round-trip proof that the future-EPG contract
   (:mod:`kairos.model.future_epg`) computes EXACTLY the features the trainer
   measured, by feeding the real EPG through the prediction-time path.
4. The honest absent state: with no future EPG file on disk the adjustment is
   exactly 0.0 and says so.

Writes ``models/candidates/tv_break_coefficients_competitor.json``: the
competition-adjusted coefficients with the gate verdict, betas and future-EPG
status in the metadata. Shipped artifacts untouched; the lead decides adoption.

    PYTHONUTF8=1 python scripts/measure_counterprogramming.py
"""

from __future__ import annotations

import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kairos.data import ProgramClassifier
from kairos.data.loaders import load_dayparts, load_programmes, load_spots
from kairos.model.competitor_gate import counterprogramming_holdout_gate
from kairos.model.competitor_model import (
    adjust_effects_for_forward_competition,
    fit_competitor_betas,
    measure_effects_with_competitors,
)
from kairos.model.future_epg import (
    counterprogramming_features_for_window,
    forward_adjustment,
    load_future_competitor_epg,
)
from kairos.model.measure import (
    _baseline_levels,
    _dayparts_frame,
    between_cell_variance,
    channel_coefficients,
    write_coefficients_json,
)
from kairos.model.competitor_features import _category_at, _programme_category_lookup
from kairos.observability.run_log import checksum_file

CANDIDATE = ROOT / "models" / "candidates" / "tv_break_coefficients_competitor.json"


def main() -> None:
    spots = load_spots()
    programmes = load_programmes()
    dayparts = load_dayparts()
    classifier = ProgramClassifier.from_yaml()

    effects = measure_effects_with_competitors(
        spots=spots, programmes=programmes, dayparts=dayparts, classifier=classifier
    )
    print(f"measured breaks with competitor features: {len(effects)}")
    for name in ("competitor_strength", "competitor_genre_contrast",
                 "competitor_prog_start", "competitor_in_break"):
        col = effects[name]
        print(f"  {name}: mean {col.mean():.4f}, std {col.std():.4f}, "
              f"nonzero {100*(col > 0).mean():.1f}%")

    # --- full-data betas (for the artifact), on the EXTENDED forward set -------
    from kairos.model.competitor_features import EXTENDED_ALL_FEATURES

    betas = fit_competitor_betas(effects, feature_names=EXTENDED_ALL_FEATURES)
    print("\nwithin-cell betas (full month):")
    for name, cb in betas.items():
        sig = "significant" if (cb.ci_low > 0 or cb.ci_high < 0) else "not significant"
        print(f"  {name} [{cb.role}]: beta {cb.beta:+.5f} "
              f"(se {cb.se:.5f}, 95% CI [{cb.ci_low:+.5f}, {cb.ci_high:+.5f}]) {sig}")

    # --- the adoption gate ------------------------------------------------------
    gate = counterprogramming_holdout_gate(effects)
    hold = gate["counterprogramming_holdout"]
    print(f"\nheld-out gate: {gate['counterprogramming_reason']}")
    print(f"  rmse_without {hold['rmse_without']}, rmse_with {hold['rmse_with']}, "
          f"n_test {hold['n_test']}")

    # --- future-EPG contract round-trip ----------------------------------------
    baseline = _baseline_levels(_dayparts_frame(dayparts))
    lookup = _programme_category_lookup(programmes, classifier)
    sample = effects.head(200)
    max_diff = 0.0

    def _mid_minute(start: pd.Timestamp, end: pd.Timestamp) -> pd.Timestamp:
        # The trainer anchors the own-programme category at the break's middle
        # minute (competitor_features.attach_competitor_features); parity here
        # makes the round-trip exact.
        from kairos.model.competitor_features import _break_minutes

        minutes = _break_minutes(start, end)
        return minutes[len(minutes) // 2]

    for row in sample.itertuples(index=False):
        own_cat = _category_at(
            lookup, str(row.channel),
            _mid_minute(pd.Timestamp(row.break_start), pd.Timestamp(row.break_end)),
        )
        feats = counterprogramming_features_for_window(
            window_start=pd.Timestamp(row.break_start),
            window_end=pd.Timestamp(row.break_end),
            epg=programmes,
            classifier=classifier,
            baseline=baseline,
            own_channel=str(row.channel),
            own_category=own_cat,
        )
        for name in ("competitor_strength", "competitor_genre_contrast",
                     "competitor_prog_start"):
            max_diff = max(max_diff, abs(feats[name] - getattr(row, name)))
    print(f"\nfuture-EPG round-trip on {len(sample)} real breaks: "
          f"max |contract feature - trained feature| = {max_diff:.2e}")

    # File-contract proof: write a rival-only slice as CompetitorProgrammes.csv
    # in a temp dir, parse through the contract loader, recompute one window.
    rival_rows = programmes[programmes["start_dt"].notna()].copy()
    with tempfile.TemporaryDirectory() as tmp:
        contract_csv = Path(tmp) / "CompetitorProgrammes.csv"
        out = pd.DataFrame({
            "Channel": rival_rows["Channel"],
            "Title": rival_rows["Title"],
            "Date": rival_rows["start_dt"].dt.strftime("%d/%m/%Y"),
            "Start time": rival_rows["start_dt"].dt.strftime("%H:%M:%S"),
            "End time": rival_rows["end_dt"].dt.strftime("%H:%M:%S"),
            "Duration": rival_rows["Duration"],
        })
        out.to_csv(contract_csv, index=False, encoding="utf-8-sig")
        epg, status = load_future_competitor_epg(contract_csv)
        print(f"file contract parse: present={status['present']}, rows={status['rows']}, "
              f"channels={len(status['channels'])}, window {status['window_start']}..{status['window_end']}")
        row = sample.iloc[0]
        feats = counterprogramming_features_for_window(
            window_start=pd.Timestamp(row["break_start"]),
            window_end=pd.Timestamp(row["break_end"]),
            epg=epg,
            classifier=classifier,
            baseline=baseline,
            own_channel=str(row["channel"]),
            own_category=_category_at(
                lookup, str(row["channel"]),
                _mid_minute(pd.Timestamp(row["break_start"]), pd.Timestamp(row["break_end"])),
            ),
        )
        beta_map = {
            name: {"beta": cb.beta, "reference": cb.reference, "role": cb.role}
            for name, cb in betas.items()
        }
        adj = forward_adjustment(feats, beta_map)
        print(f"one real window through the file contract: features {feats}")
        print(f"  adjustment {adj['adjustment']:+.5f} (applied {adj['applied']})")

    # Honest absent state (no CompetitorProgrammes file ships in the repo).
    epg_missing, missing_status = load_future_competitor_epg()
    adj_missing = forward_adjustment(
        None if epg_missing is None else {}, beta_map
    )
    print(f"\nabsent-state check: present={missing_status['present']}; "
          f"adjustment {adj_missing['adjustment']} (applied {adj_missing['applied']})")
    print(f"  reason: {missing_status['reason']}")

    # --- candidate artifact: competition-adjusted coefficients ------------------
    adjusted = adjust_effects_for_forward_competition(effects, betas)
    coefficients = channel_coefficients(adjusted)
    diagnostics = between_cell_variance(adjusted)
    plain = channel_coefficients(effects)
    shifts = [abs(coefficients[n].coefficient - plain[n].coefficient)
              for n in coefficients if n in plain]
    print(f"\nde-confounding shift vs plain coefficients: mean |shift| "
          f"{np.mean(shifts):.6f}, max {np.max(shifts):.6f}")

    CANDIDATE.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "candidate": True,
        "purpose": "competition-adjusted coefficients (forward de-confounding) + counter-programming gate",
        "total_breaks_measured": int(sum(c.n for c in coefficients.values())),
        "pooling_method": diagnostics["method"],
        "between_cell_variance_tau2": diagnostics["tau2"],
        "counterprogramming_active_recommended": gate["counterprogramming_active"],
        "counterprogramming_holdout": gate["counterprogramming_holdout"],
        "counterprogramming_reason": gate["counterprogramming_reason"],
        "competitor_betas": gate["counterprogramming_betas"],
        "forward_features": gate["forward_features"],
        "future_epg_status": missing_status,
        "future_epg_roundtrip_max_diff": max_diff,
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "source_fingerprints": {
            f"data/reference/{name}": checksum_file(ROOT / "data" / "reference" / name)
            for name in ("Spots.xlsx", "Programmes.xlsx", "Dayparts.xlsx")
        },
    }
    write_coefficients_json(CANDIDATE, coefficients, metadata=metadata)
    print(f"\nwrote candidate: {CANDIDATE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
