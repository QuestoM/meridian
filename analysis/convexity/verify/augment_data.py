"""Build the attack dataset: all keyed breaks with gaps, survival, and title.

READ-ONLY over data/. Writes only under analysis/convexity/verify/.

For every keyed break (the full population before the after-window clip):
  gap_prev_min  minutes from previous keyed break's end to this break's start
  gap_next_min  minutes from this break's end to next keyed break's start
  measured      1 if the break survived into break_effects (the analysis set)
  title         programme title from the measured effects frame (measured only)

Used by the refutation attacks: clip-selection vs length, spacing vs shed,
title-level selection.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from kairos.data.classifier import ProgramClassifier
from kairos.data.loaders import load_dayparts, load_programmes, load_spots
from kairos.model.measure import break_effects
from kairos.model.prepare import keyed_breaks

OUT = Path(__file__).resolve().parent


def main() -> None:
    spots = load_spots()
    programmes = load_programmes()
    dayparts = load_dayparts()
    classifier = ProgramClassifier.from_yaml()

    kb = keyed_breaks(spots, programmes, classifier)
    eff = break_effects(spots, programmes, dayparts, classifier)

    kb = kb.copy()
    kb["start_min"] = pd.to_datetime(kb["break_start"]).dt.floor("min")
    kb["end_min"] = pd.to_datetime(kb["break_end"]).dt.floor("min")
    kb = kb.sort_values(["channel", "start_min"]).reset_index(drop=True)
    kb["prev_end"] = kb.groupby("channel")["end_min"].shift(1)
    kb["next_start"] = kb.groupby("channel")["start_min"].shift(-1)
    kb["gap_prev_min"] = (kb["start_min"] - kb["prev_end"]).dt.total_seconds() / 60.0
    kb["gap_next_min"] = (kb["next_start"] - kb["end_min"]).dt.total_seconds() / 60.0

    em = eff.copy()
    em["start_min"] = pd.to_datetime(em["break_start"]).dt.floor("min")
    em["measured"] = 1
    em = em[["channel", "start_min", "measured", "title", "log_effect"]]
    em = em.drop_duplicates(subset=["channel", "start_min"], keep="first")

    kb = kb.merge(em, on=["channel", "start_min"], how="left")
    kb["measured"] = kb["measured"].fillna(0).astype(int)
    kb["len_min"] = kb["break_seconds"] / 60.0
    kb["shed"] = -kb["log_effect"]

    keep = [
        "channel", "channel_name", "start_min", "end_min", "break_seconds",
        "len_min", "num_spots", "program_type", "break_position", "break_length",
        "ordinal", "prog_key", "gap_prev_min", "gap_next_min", "measured",
        "title", "shed",
    ]
    kb[keep].to_csv(OUT / "keyed_breaks_augmented.csv", index=False)
    print("rows:", len(kb), "measured:", int(kb["measured"].sum()))
    print("titles nonempty among measured:",
          int((kb.loc[kb["measured"] == 1, "title"].fillna("") != "").sum()))


if __name__ == "__main__":
    main()
