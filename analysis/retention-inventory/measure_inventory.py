"""Retention measurement data inventory. READ-ONLY over data/.

Runs the real break_effects measurement and reports sample sizes plus the two
feasibility questions (split-vs-long, habit horizon). Writes only under this
analysis lane.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from kairos.data.classifier import ProgramClassifier
from kairos.data.loaders import load_dayparts, load_programmes, load_spots
from kairos.model.measure import (
    break_effects,
    between_cell_variance,
    channel_coefficients,
    first_break_gate,
)
from kairos.model.prepare import keyed_breaks, identify_breaks, length_bucket

OUT = Path(__file__).resolve().parent
report: dict[str, object] = {}

spots = load_spots()
programmes = load_programmes()
dayparts = load_dayparts()
classifier = ProgramClassifier.from_yaml()

# ---- raw scale of the source logs ----
report["spots_rows"] = int(len(spots))
report["programmes_rows"] = int(len(programmes))
report["dayparts_rows"] = int(len(dayparts))
report["channels"] = sorted(str(c) for c in dayparts["channel"].dropna().unique())

dp_dates = pd.to_datetime(dayparts["date"], errors="coerce").dropna()
report["dayparts_date_min"] = str(dp_dates.min().date())
report["dayparts_date_max"] = str(dp_dates.max().date())
report["dayparts_distinct_days"] = int(dp_dates.dt.normalize().nunique())
# minute granularity check
report["dayparts_distinct_timebands"] = int(dayparts["timeband"].nunique())

# ---- keyed breaks (all detected, before measurement drops) ----
kb = keyed_breaks(spots, programmes, classifier)
report["keyed_breaks_total"] = int(len(kb))
report["keyed_breaks_days"] = int(kb["day"].nunique()) if len(kb) else 0
report["keyed_breaks_channel_name_cells"] = int(kb["channel_name"].nunique()) if len(kb) else 0
report["keyed_breaks_with_prog_key"] = int(kb["prog_key"].notna().sum()) if len(kb) else 0
report["keyed_breaks_distinct_prog_key"] = int(kb["prog_key"].nunique()) if len(kb) else 0

# ---- measured effects (after window clip + positive-audience drops) ----
effects = break_effects(spots, programmes, dayparts, classifier)
report["measured_breaks"] = int(len(effects))
if len(effects):
    eff = effects.copy()
    eff["date"] = pd.to_datetime(eff["break_start"]).dt.normalize()
    report["measured_days"] = int(eff["date"].nunique())
    report["measured_channel_name_cells"] = int(eff["channel_name"].nunique())
    report["measured_raw_channels"] = sorted(str(c) for c in eff["channel"].dropna().unique())
    report["measured_program_types"] = (
        eff["program_type"].value_counts().to_dict()
    )
    report["measured_break_length_buckets"] = (
        eff["break_length"].value_counts().to_dict()
    )
    report["measured_break_position"] = (
        eff["break_position"].value_counts().to_dict()
    )
    report["measured_with_ordinal"] = int(eff["ordinal"].notna().sum())
    report["measured_distinct_prog_key"] = int(eff["prog_key"].nunique())
    # drop rate from keyed -> measured
    report["drop_keyed_to_measured"] = int(len(kb) - len(effects))

# ---- EB fit + first-break gate on the real data ----
diag = between_cell_variance(effects)
report["eb_diagnostics"] = {k: (None if v is None else v) for k, v in diag.items()}
coefs = channel_coefficients(effects)
report["coef_cells"] = int(len(coefs))
fbg = first_break_gate(effects)
report["first_break_gate"] = fbg

# ---- FEASIBILITY 1: split-vs-long ----
# Per programme instance: total break seconds and number of breaks inside it.
# Group by (program title, daypart hour) and look for equal total-minutes cells
# with different split counts (different number of breaks).
raw_breaks = identify_breaks(spots)  # channel, break_start, break_end, break_seconds, num_spots
# attach programme + daypart to each break via keyed_breaks (has prog_key, day)
kb2 = kb.copy()
kb2 = kb2[kb2["prog_key"].notna()].copy()
kb2["break_minutes"] = kb2["break_seconds"] / 60.0
kb2["hour"] = pd.to_datetime(kb2["break_start"]).dt.hour
# derive a programme title per prog_key from programmes span match is expensive;
# instead use prog_key (channel|day|programme-index) as the instance and
# program_type + channel_name-agnostic program-slot. We approximate the
# repeatable "program cell" by (channel, program_type, hour) daypart bucket.
inst = kb2.groupby("prog_key").agg(
    n_breaks=("break_seconds", "size"),
    total_seconds=("break_seconds", "sum"),
    channel=("channel", "first"),
    program_type=("program_type", "first"),
    hour=("hour", "first"),
    day=("day", "first"),
).reset_index()
inst["total_minutes_rounded"] = (inst["total_seconds"] / 60.0).round().astype(int)

# Cell = (channel, program_type, daypart hour). Within a cell, find total-minute
# values that occur with >=2 distinct split counts (n_breaks).
inst["daypart"] = pd.cut(
    inst["hour"],
    bins=[-1, 5, 11, 16, 19, 23],
    labels=["overnight", "morning", "afternoon", "prime", "latenight"],
).astype(str)

split_cells = []
grp = inst.groupby(["channel", "program_type", "daypart", "total_minutes_rounded"])
for (ch, pt, dp, tm), g in grp:
    distinct_splits = g["n_breaks"].nunique()
    if distinct_splits >= 2 and len(g) >= 2:
        split_cells.append({
            "channel": str(ch),
            "program_type": str(pt),
            "daypart": str(dp),
            "total_minutes": int(tm),
            "n_instances": int(len(g)),
            "distinct_split_counts": int(distinct_splits),
            "split_counts": sorted(int(x) for x in g["n_breaks"].unique()),
        })

split_cells.sort(key=lambda d: d["n_instances"], reverse=True)
report["split_vs_long_cells_found"] = len(split_cells)
report["split_vs_long_top20"] = split_cells[:20]
report["split_vs_long_total_instances_in_such_cells"] = int(
    sum(c["n_instances"] for c in split_cells)
)
# also: overall distribution of breaks-per-programme-instance
report["breaks_per_instance_distribution"] = (
    inst["n_breaks"].value_counts().sort_index().to_dict()
)
report["instances_total"] = int(len(inst))

# ---- FEASIBILITY 2: habit horizon ----
# Can we link the same programme slot / audience across consecutive days?
# Granularity available: daypart TVR is (channel, date, minute). Programmes have
# Title + Channel + Date. Check recurrence of programme Titles across dates.
prog = programmes.copy()
prog["date"] = pd.to_datetime(prog["start_dt"], errors="coerce").dt.normalize()
prog = prog[prog["date"].notna()]
report["programme_distinct_days"] = int(prog["date"].nunique())
report["programme_date_min"] = str(prog["date"].min().date())
report["programme_date_max"] = str(prog["date"].max().date())
# Titles recurring on >=2 distinct days (candidate daily strips)
title_days = prog.groupby("Title")["date"].nunique()
recurring = title_days[title_days >= 2]
report["programme_titles_total"] = int(prog["Title"].nunique())
report["programme_titles_recurring_2plus_days"] = int(len(recurring))
# consecutive-day recurrence: for each title, count consecutive-day pairs
consec_pairs = 0
strip_titles = []
for title, g in prog.groupby("Title"):
    days = sorted(set(g["date"].dt.normalize()))
    dd = [(days[i + 1] - days[i]).days for i in range(len(days) - 1)]
    cp = sum(1 for x in dd if x == 1)
    if cp > 0:
        consec_pairs += cp
        strip_titles.append({"title": str(title), "consecutive_day_pairs": int(cp),
                             "distinct_days": int(len(days))})
strip_titles.sort(key=lambda d: d["consecutive_day_pairs"], reverse=True)
report["consecutive_day_title_pairs_total"] = int(consec_pairs)
report["top_strip_titles"] = strip_titles[:15]

with open(OUT / "inventory_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False, default=str)
print(json.dumps(report, indent=2, ensure_ascii=False, default=str))

# ---- FEASIBILITY 1 refinement: on MEASURED effects only ----
# Join measured breaks to their programme instance and require all breaks in the
# instance to have survived measurement, so per-instance total shedding is clean.
if len(effects):
    em = effects.copy()
    em["hour"] = pd.to_datetime(em["break_start"]).dt.hour
    em["daypart"] = pd.cut(em["hour"], bins=[-1, 5, 11, 16, 19, 23],
                           labels=["overnight", "morning", "afternoon", "prime", "latenight"]).astype(str)
    em = em[em["prog_key"].notna()].copy()
    # breaks measured per instance
    meas_per_inst = em.groupby("prog_key").size().rename("n_meas")
    kb_per_inst = kb[kb["prog_key"].notna()].groupby("prog_key").size().rename("n_keyed")
    inst_full = inst.set_index("prog_key").join(meas_per_inst).join(kb_per_inst)
    inst_full["fully_measured"] = inst_full["n_meas"] == inst_full["n_keyed"]
    full = inst_full[inst_full["fully_measured"].fillna(False)].reset_index()
    full["daypart"] = pd.cut(full["hour"], bins=[-1, 5, 11, 16, 19, 23],
                             labels=["overnight", "morning", "afternoon", "prime", "latenight"]).astype(str)
    scells = []
    for (ch, pt, dp, tm), g in full.groupby(["channel", "program_type", "daypart", "total_minutes_rounded"]):
        if g["n_breaks"].nunique() >= 2 and len(g) >= 2:
            scells.append({"channel": str(ch), "program_type": str(pt), "daypart": str(dp),
                           "total_minutes": int(tm), "n_instances": int(len(g)),
                           "split_counts": sorted(int(x) for x in g["n_breaks"].unique())})
    scells.sort(key=lambda d: d["n_instances"], reverse=True)
    extra = {
        "split_measured_fully_instances": int(len(full)),
        "split_measured_cells_found": len(scells),
        "split_measured_total_instances_in_cells": int(sum(c["n_instances"] for c in scells)),
        "split_measured_top10": scells[:10],
    }
    report.update(extra)
    with open(OUT / "inventory_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(json.dumps(extra, indent=2, ensure_ascii=False, default=str))
