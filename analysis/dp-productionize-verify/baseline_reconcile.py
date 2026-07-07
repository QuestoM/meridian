"""Reconcile the golden export baseline: DP on vs forced-off, through the SHIPPED
export path (build_weekly_schedule), on the real committed settings."""
from __future__ import annotations

import functools
import json
import re
from datetime import date
from pathlib import Path

import kairos.export.schedule as sched
from kairos.optimize.optimizer import optimize_breaks

ROOT = Path("/Users/home/Code/questo/meridian")
SETTINGS = json.loads((ROOT / "data" / "kairos_settings.json").read_text(encoding="utf-8"))


def total_rev(frame):
    return float(frame["predicted_revenue"].astype(float).sum())


def run(dp_on: bool):
    if dp_on:
        sched.optimize_breaks = optimize_breaks
    else:
        sched.optimize_breaks = functools.partial(optimize_breaks, dp_refine=False)
    frame = sched.build_weekly_schedule(
        settings=SETTINGS,
        revenue_weight=SETTINGS["revenue_weight"] / 100.0,
        risk_lambda=SETTINGS["risk_lambda"],
        operator_channel=SETTINGS["operator_channel"],
        today=date.today(),
    )
    sched.optimize_breaks = optimize_breaks
    # per channel-day revenue and breaks
    agg = frame.groupby(["channel", "date"]).agg(
        rev=("predicted_revenue", "sum"), br=("num_breaks", "sum")
    ).reset_index()
    per = {(r.channel, r.date): (float(r.rev), int(r.br)) for r in agg.itertuples(index=False)}
    return total_rev(frame), per


def main():
    on_total, on_per = run(True)
    off_total, off_per = run(False)
    print(f"EXPORT dp_refine ON  total predicted_revenue = {on_total:.2f}")
    print(f"EXPORT dp_refine OFF total predicted_revenue = {off_total:.2f}")
    print(f"EXPORT delta (on-off) = {on_total-off_total:.2f} ({100*(on_total-off_total)/off_total:.4f}%)")

    # never-worse through export: no channel-day may regress revenue... actually the
    # invariant is the OBJECTIVE, but report revenue regressions too.
    rev_regress = []
    for key in on_per:
        r_on = on_per[key][0]
        r_off = off_per[key][0]
        if r_on < r_off - 1.0:
            rev_regress.append((key, r_off, r_on))
    print(f"channel-days where export revenue went DOWN with DP on: {len(rev_regress)}")
    for k, ro, rn in rev_regress[:10]:
        print(f"  {k} off={ro:.2f} on={rn:.2f} diff={rn-ro:.2f}")

    # count how many days changed
    changed = sum(1 for k in on_per if on_per[k][1] != off_per[k][1] or abs(on_per[k][0]-off_per[k][0])>0.01)
    print(f"channel-days changed by DP: {changed} of {len(on_per)}")

    # Compare export-OFF total to the OLD committed golden baseline sum (215.34M)
    src_old = json.loads(re.search(r"_BASELINE_AGG_JSON = r'''(\[.*?\])'''",
        (ROOT/"tests"/"golden_weekly_schedule.py").read_text(), re.S).group(1)) if False else None
    print(f"\nNEW committed golden sum   = 221891590.23 (matches EXPORT ON? {abs(on_total-221891590.23)<1.0})")
    print(f"OLD committed golden sum   = 215338683.05 (matches EXPORT OFF? {abs(off_total-215338683.05)<1.0})")


if __name__ == "__main__":
    main()
