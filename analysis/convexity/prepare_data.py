"""Prepare per-break measured shedding data for the convexity analysis.

READ-ONLY over data/. Writes CSVs only under analysis/convexity/.

Runs the shipped measurement (kairos.model.measure.break_effects, default
config: 3-minute windows, after-window clip ON) and joins the continuous
break length in seconds from keyed_breaks onto each measured break.

Outputs:
  breaks_measured.csv   one row per measured break with continuous length
  instances.csv         one row per fully-measured programme instance
  prep_summary.json     join rates, length distribution, sample sizes
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from kairos.data.classifier import ProgramClassifier
from kairos.data.loaders import load_dayparts, load_programmes, load_spots
from kairos.model.measure import break_effects
from kairos.model.prepare import keyed_breaks

OUT = Path(__file__).resolve().parent

DAYPART_BINS = [-1, 5, 11, 16, 19, 23]
DAYPART_LABELS = ["overnight", "morning", "afternoon", "early_evening", "late_evening"]


def daypart_of(hours: pd.Series) -> pd.Series:
    return pd.cut(hours, bins=DAYPART_BINS, labels=DAYPART_LABELS).astype(str)


def main() -> None:
    summary: dict[str, object] = {}
    spots = load_spots()
    programmes = load_programmes()
    dayparts = load_dayparts()
    classifier = ProgramClassifier.from_yaml()

    kb = keyed_breaks(spots, programmes, classifier)
    eff = break_effects(spots, programmes, dayparts, classifier)
    summary["keyed_breaks"] = int(len(kb))
    summary["measured_breaks"] = int(len(eff))

    # Join continuous break_seconds and num_spots on (channel, floored start).
    kbj = kb.copy()
    kbj["start_min"] = pd.to_datetime(kbj["break_start"]).dt.floor("min")
    dup = kbj.duplicated(subset=["channel", "start_min"], keep=False).sum()
    summary["keyed_dup_channel_startmin_rows"] = int(dup)
    kbj = kbj.drop_duplicates(subset=["channel", "start_min"], keep="first")
    kbj = kbj[["channel", "start_min", "break_seconds", "num_spots"]]

    em = eff.copy()
    em["start_min"] = pd.to_datetime(em["break_start"]).dt.floor("min")
    em = em.merge(kbj, on=["channel", "start_min"], how="left")
    summary["join_matched"] = int(em["break_seconds"].notna().sum())
    summary["join_unmatched"] = int(em["break_seconds"].isna().sum())
    em = em[em["break_seconds"].notna()].copy()

    em["len_min"] = em["break_seconds"] / 60.0
    em["shed"] = -em["log_effect"]  # positive = audience lost vs expected
    em["hour"] = pd.to_datetime(em["break_start"]).dt.hour
    em["daypart"] = daypart_of(em["hour"])
    em["date"] = pd.to_datetime(em["break_start"]).dt.strftime("%Y-%m-%d")
    em["chan_day"] = em["channel"].astype(str) + "|" + em["date"]
    em["first_break"] = (pd.to_numeric(em["ordinal"], errors="coerce") == 1).astype(int)
    em["cluster"] = em["prog_key"].astype(str)
    em.loc[em["prog_key"].isna(), "cluster"] = "cd:" + em.loc[
        em["prog_key"].isna(), "chan_day"
    ]

    keep = [
        "channel", "channel_name", "program_type", "break_position",
        "break_length", "daypart", "date", "hour", "len_min", "break_seconds",
        "num_spots", "shed", "log_effect", "observed_ratio", "expected_ratio",
        "ordinal", "first_break", "prog_key", "cluster", "start_min",
    ]
    em[keep].to_csv(OUT / "breaks_measured.csv", index=False)

    q = em["len_min"].quantile([0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0])
    summary["len_min_quantiles"] = {str(k): round(float(v), 3) for k, v in q.items()}
    summary["rows_written"] = int(len(em))
    summary["shed_mean"] = round(float(em["shed"].mean()), 5)
    summary["shed_std"] = round(float(em["shed"].std()), 5)

    # Fully-measured programme instances (every keyed break in the instance
    # survived measurement) for the equal-total-minutes split comparison.
    kbp = kb[kb["prog_key"].notna()].copy()
    kbp["hour"] = pd.to_datetime(kbp["break_start"]).dt.hour
    inst = kbp.groupby("prog_key").agg(
        n_breaks=("break_seconds", "size"),
        total_seconds=("break_seconds", "sum"),
        channel=("channel", "first"),
        program_type=("program_type", "first"),
        hour=("hour", "first"),
        day=("day", "first"),
    ).reset_index()
    inst["daypart"] = daypart_of(inst["hour"])
    inst["total_minutes_rounded"] = (inst["total_seconds"] / 60.0).round().astype(int)

    emp = em[em["prog_key"].notna()]
    meas = emp.groupby("prog_key").agg(
        n_meas=("shed", "size"),
        total_shed=("shed", "sum"),
        mean_len=("len_min", "mean"),
    ).reset_index()
    keyed_n = kbp.groupby("prog_key").size().rename("n_keyed").reset_index()
    inst = inst.merge(meas, on="prog_key", how="left").merge(keyed_n, on="prog_key", how="left")
    inst["fully_measured"] = inst["n_meas"] == inst["n_keyed"]
    full = inst[inst["fully_measured"].fillna(False)].copy()
    full.to_csv(OUT / "instances.csv", index=False)
    summary["instances_fully_measured"] = int(len(full))
    summary["instances_total"] = int(len(inst))

    with open(OUT / "prep_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
