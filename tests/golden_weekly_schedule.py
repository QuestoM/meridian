"""Golden-master safety net for the exported weekly break schedule.

Runs :func:`kairos.export.schedule.build_weekly_schedule` on the committed
reference data along the exact path ``POST /api/recompute-schedule`` uses
(saved settings, ``revenue_weight`` / 100, saved ``risk_lambda``, saved
``operator_channel``, and a frozen ``today`` so the byte hash cannot drift with
the wall clock), then asserts the output is the committed golden to the byte:

  * a full-CSV content hash (the CSV carries no timestamp column, so its bytes
    are the whole content), and
  * a per-channel-day aggregate hash over ``predicted_revenue``,
    ``predicted_retention`` and ``num_breaks``.

Any drift in a single per-day total flips both hashes; the test then diffs the
recomputed aggregate against the embedded baseline and names the channel-days
that moved. This gates the Phase-1 engine-core consolidation: the consolidated
engine must reproduce this schedule exactly.

A full run optimises every channel-day and takes roughly 45-70s, which is
acceptable for a safety net. The delivery-pacing signal is the only consumer of
``today``, and it is inert while ``campaign_flights.csv`` is header-only
(``load_campaigns()`` returns ``[]``), so the schedule does not depend on the run
date. ``today`` is frozen here anyway, so the golden cannot start depending on the
clock even after campaign flights land; the companion
``tests/test_qa2_golden_freeze.py`` asserts the campaign inputs are still empty so
that this freeze is provably a no-op today.

Run directly (``python tests/golden_weekly_schedule.py``) or under pytest.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kairos.export.schedule import build_weekly_schedule  # noqa: E402

SETTINGS_PATH = ROOT / "data" / "kairos_settings.json"
SHIPPED_PLAN_PATH = ROOT / "output" / "weekly_break_schedule.csv"
SHIPPED_FINGERPRINT_PATH = ROOT / "output" / "weekly_break_schedule.csv.fingerprint.json"

# Reference date for the delivery-pacing urgency signal. Frozen (never date.today())
# so this byte-hash golden is deterministic across days. Pacing is the only consumer
# of this date and it is inert while campaign flights are header-only, so the exact
# value does not move the golden; the companion test proves that emptiness.
FROZEN_PACING_DATE = date(2026, 6, 15)

# Committed golden, captured from a real recompute on the current engine.
# Rebased when the exact interval-sweep DP shipped as the optimizer's top refiner
# tier default on (kairos/optimize/dp_refine.py): it recovers value greedy+F1 left
# on the table on the real channel-days. Measured against the current engine's own
# pre-DP baseline (the same recompute with dp_refine=False, which sums to
# 209,941,254.97 ILS), the tier lifts predicted revenue to 221,891,590.23 ILS,
# +11,950,335.26 (+5.69%) across the week. The structural guarantee is that no
# channel-day regresses on the optimizer's per-group OBJECTIVE (adopt-only-if-
# strictly-better gate); empirically revenue also rose on every changed day and
# retention cost rose with it (more breaks earn and cost more), so "no regression"
# is about the objective, not revenue or cost. Note the diff against this file's
# PRIOR committed baseline is only +6,552,907 (+3.04%) because that prior golden
# (215,338,683.05 ILS) was stale, captured under an earlier engine/data state and
# already superseded; the honest DP delta is the +11.95M measured off the current
# dp-off baseline. The committed CSV and both hashes were recomputed from the same
# shipped recompute path; this baseline is the new plan of record.
#
# REBASED 2026-08-23 for the segment capacity guard. A segment cannot carry more
# advertising than it is long, and nothing enforced that: MEASURED on the prior
# baseline, 846 of 8,704 segments were sold MORE ad seconds than their own
# duration -- 89,576 seconds of inventory that cannot exist. The worst was a
# SIX-SECOND programme row sold 480 seconds of advertising carrying 130,290 ILS,
# and a four-second row sold the same 480. ProgramSegment now clamps max_breaks
# to duration // break_length at construction, so no solver, refiner or override
# path can route around it.
#
# The plan loses 11,804,030 ILS (221,891,590 -> 210,087,560, -5.32%) and 66
# breaks; mean retention rises slightly (68.8004 -> 68.8176). That money was
# never sellable. This is a REDUCTION IN CLAIMED REVENUE and it is the correct
# direction: the prior number priced airtime that no clock has room for.
GOLDEN_CSV_SHA256 = "d17e230e5387445e748594309692180508f50d82937030d7112cbdf3f796968e"
GOLDEN_ROWS = 8704
GOLDEN_AGG_SHA256 = "f6ea46fdb7125825277b755febd1333da1105821b5643115d69dd59232254334"

# Per-channel-day aggregate baseline, one entry per channel-day as
# [channel, date, predicted_revenue, predicted_retention, num_breaks], sorted by
# (date, channel). Embedded so a drift is pinpointed to the exact channel-day
# with no external file. This is the committed baseline the harness stores.
_BASELINE_AGG_JSON = r'''[["כאן 11","2024-11-01",874144.74,69.1738,78],["עכשיו 14","2024-11-01",482152.82,26.0639,40],["קשת 12","2024-11-01",3567618.78,92.0685,80],["רשת 13","2024-11-01",1067845.56,78.1067,80],["כאן 11","2024-11-02",1648851.73,79.9955,80],["עכשיו 14","2024-11-02",876833.69,14.53,29],["קשת 12","2024-11-02",4400414.98,91.0188,80],["רשת 13","2024-11-02",1458354.36,71.0784,80],["כאן 11","2024-11-03",774174.2,74.9506,80],["עכשיו 14","2024-11-03",1624631.73,50.1703,77],["קשת 12","2024-11-03",3913864.54,84.9761,80],["רשת 13","2024-11-03",1643821.27,87.0678,80],["כאן 11","2024-11-04",782880.1,77.9361,80],["עכשיו 14","2024-11-04",1529218.0,49.276,75],["קשת 12","2024-11-04",3638619.34,85.9949,80],["רשת 13","2024-11-04",1698547.38,84.068,80],["כאן 11","2024-11-05",644117.87,64.33,73],["עכשיו 14","2024-11-05",1766960.29,43.3997,73],["קשת 12","2024-11-05",4237124.05,89.991,80],["רשת 13","2024-11-05",1344688.6,80.0749,80],["כאן 11","2024-11-06",747213.46,56.4688,70],["עכשיו 14","2024-11-06",2090900.06,41.314,74],["קשת 12","2024-11-06",4318069.18,68.0091,80],["רשת 13","2024-11-06",1403195.57,60.4016,73],["כאן 11","2024-11-07",678583.91,79.9361,80],["עכשיו 14","2024-11-07",1852282.75,51.1024,78],["קשת 12","2024-11-07",4079669.46,85.9422,80],["רשת 13","2024-11-07",1506618.06,81.0622,80],["כאן 11","2024-11-08",934951.34,49.5198,71],["עכשיו 14","2024-11-08",617998.83,20.1958,37],["קשת 12","2024-11-08",3629663.08,69.2709,76],["רשת 13","2024-11-08",1234763.31,61.1942,78],["כאן 11","2024-11-09",1071037.28,77.0287,80],["עכשיו 14","2024-11-09",964555.52,15.5789,28],["קשת 12","2024-11-09",4670374.2,95.0115,80],["רשת 13","2024-11-09",1318751.08,79.0779,80],["כאן 11","2024-11-10",632326.28,67.9896,80],["עכשיו 14","2024-11-10",1570900.55,49.1776,77],["קשת 12","2024-11-10",3981436.75,86.9936,80],["רשת 13","2024-11-10",1404645.68,89.0725,80],["כאן 11","2024-11-11",715844.11,77.9804,80],["עכשיו 14","2024-11-11",1388505.38,42.5423,70],["קשת 12","2024-11-11",3837761.32,83.9837,80],["רשת 13","2024-11-11",1414695.19,91.0736,80],["כאן 11","2024-11-12",1082024.5,77.9798,80],["עכשיו 14","2024-11-12",1352987.9,46.0513,80],["קשת 12","2024-11-12",4116742.86,87.0151,80],["רשת 13","2024-11-12",1190924.38,80.0579,80],["כאן 11","2024-11-13",854841.34,71.9907,80],["עכשיו 14","2024-11-13",1499695.55,47.3678,73],["קשת 12","2024-11-13",3665137.2,84.9841,80],["רשת 13","2024-11-13",1334945.8,87.0649,80],["כאן 11","2024-11-14",695272.23,67.0182,80],["עכשיו 14","2024-11-14",1749428.09,44.1062,78],["קשת 12","2024-11-14",3842956.39,91.9752,80],["רשת 13","2024-11-14",1314546.58,83.065,80],["כאן 11","2024-11-15",777363.35,67.2917,76],["עכשיו 14","2024-11-15",589321.8,25.9703,42],["קשת 12","2024-11-15",3132161.81,88.037,80],["רשת 13","2024-11-15",1061721.48,80.1069,80],["כאן 11","2024-11-16",1257737.54,66.0334,80],["עכשיו 14","2024-11-16",897133.72,15.5857,28],["קשת 12","2024-11-16",3997932.44,90.9981,80],["רשת 13","2024-11-16",1282731.24,78.0838,80],["כאן 11","2024-11-17",693039.98,68.0025,80],["עכשיו 14","2024-11-17",1618355.63,49.2675,75],["קשת 12","2024-11-17",3535156.53,84.9816,80],["רשת 13","2024-11-17",1309478.66,85.061,80],["כאן 11","2024-11-18",734941.33,69.4165,71],["עכשיו 14","2024-11-18",1702906.34,49.0744,79],["קשת 12","2024-11-18",3565406.69,87.9746,80],["רשת 13","2024-11-18",1603370.11,90.0638,80],["כאן 11","2024-11-19",800110.19,67.9949,80],["עכשיו 14","2024-11-19",1578109.83,49.2376,76],["קשת 12","2024-11-19",3758341.78,86.989,80],["רשת 13","2024-11-19",1455895.51,88.0676,80],["כאן 11","2024-11-20",710980.03,68.9975,80],["עכשיו 14","2024-11-20",1508784.99,48.2342,76],["קשת 12","2024-11-20",3714346.62,91.9904,80],["רשת 13","2024-11-20",1520736.38,86.0688,80],["כאן 11","2024-11-21",865917.43,66.0117,80],["עכשיו 14","2024-11-21",1644394.91,48.2675,75],["קשת 12","2024-11-21",3644554.22,88.9648,80],["רשת 13","2024-11-21",1468335.22,84.0744,80],["כאן 11","2024-11-22",828854.49,63.0889,80],["עכשיו 14","2024-11-22",615176.71,27.8127,45],["קשת 12","2024-11-22",3301489.17,99.0513,80],["רשת 13","2024-11-22",835998.69,77.1066,80],["כאן 11","2024-11-23",995034.4,66.0257,80],["עכשיו 14","2024-11-23",877049.69,16.5863,28],["קשת 12","2024-11-23",4365220.36,96.0343,80],["רשת 13","2024-11-23",1542178.08,80.0706,80],["כאן 11","2024-11-24",911353.66,52.9938,80],["עכשיו 14","2024-11-24",1658767.01,46.2681,75],["קשת 12","2024-11-24",4091054.76,82.9927,80],["רשת 13","2024-11-24",1404424.87,76.0742,80],["כאן 11","2024-11-25",786940.86,69.0002,80],["עכשיו 14","2024-11-25",1535924.72,52.1696,77],["קשת 12","2024-11-25",3613916.34,89.9855,80],["רשת 13","2024-11-25",1303642.81,81.0679,80],["כאן 11","2024-11-26",801527.07,65.0015,80],["עכשיו 14","2024-11-26",1679756.51,51.1797,77],["קשת 12","2024-11-26",4190081.78,87.9912,80],["רשת 13","2024-11-26",1532829.15,79.0732,80],["כאן 11","2024-11-27",699460.23,67.0043,80],["עכשיו 14","2024-11-27",1650845.67,51.0946,79],["קשת 12","2024-11-27",3796591.92,88.9908,80],["רשת 13","2024-11-27",1454765.62,88.0727,80],["כאן 11","2024-11-28",713732.11,70.0633,79],["עכשיו 14","2024-11-28",1779061.65,49.0116,80],["קשת 12","2024-11-28",3583396.76,85.9739,80],["רשת 13","2024-11-28",1454260.56,85.0799,80],["כאן 11","2024-11-29",919649.14,68.0694,80],["עכשיו 14","2024-11-29",365481.9,35.5229,51],["קשת 12","2024-11-29",3024221.96,80.0528,80],["רשת 13","2024-11-29",1023330.87,77.115,80],["כאן 11","2024-11-30",1367353.03,64.0362,80],["עכשיו 14","2024-11-30",893357.0,18.5387,29],["קשת 12","2024-11-30",3771768.46,79.0357,80],["רשת 13","2024-11-30",1354717.26,75.0894,80]]'''
GOLDEN_AGG = json.loads(_BASELINE_AGG_JSON)


def settings_map() -> dict:
    """Load the committed operator settings the recompute path reads."""
    return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))


def build_reference_frame():
    """Reproduce the ``POST /api/recompute-schedule`` commit path byte-for-byte."""
    settings = settings_map()
    return build_weekly_schedule(
        settings=settings,
        revenue_weight=settings["revenue_weight"] / 100.0,
        risk_lambda=settings["risk_lambda"],
        operator_channel=settings["operator_channel"],
        # A fixed reference date, not date.today(), so the byte-hash golden cannot
        # silently start depending on the wall clock. The pacing urgency signal is
        # the only consumer of this date, and it is inert while campaign flights are
        # empty (header-only campaign_flights.csv -> load_campaigns() == []), so this
        # freeze is a provable no-op today. tests/test_qa2_golden_freeze.py asserts
        # that emptiness so the day the campaigns land, that guard fails loudly rather
        # than this golden drifting unnoticed.
        today=FROZEN_PACING_DATE,
    )


def csv_hash(frame) -> str:
    """SHA-256 of the CSV bytes, matching ``write_weekly_schedule``'s utf-8 output."""
    return hashlib.sha256(frame.to_csv(index=False).encode("utf-8")).hexdigest()


def aggregate_records(frame) -> list:
    """Per-channel-day totals as [channel, date, revenue, retention, breaks]."""
    agg = (
        frame.groupby(["channel", "date"]).agg(
            predicted_revenue=("predicted_revenue", "sum"),
            predicted_retention=("predicted_retention", "sum"),
            num_breaks=("num_breaks", "sum"),
        ).reset_index()
    )
    records = [
        [
            r.channel,
            r.date,
            round(float(r.predicted_revenue), 2),
            round(float(r.predicted_retention), 4),
            int(r.num_breaks),
        ]
        for r in agg.itertuples(index=False)
    ]
    records.sort(key=lambda x: (x[1], x[0]))
    return records


def agg_hash(records: list) -> str:
    canon = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def per_day_drift(records: list) -> list:
    """Channel-days whose totals differ from the committed baseline."""
    baseline = {(c, d): [rev, ret, br] for c, d, rev, ret, br in GOLDEN_AGG}
    current = {(c, d): [rev, ret, br] for c, d, rev, ret, br in records}
    drift = []
    for key in sorted(set(baseline) | set(current), key=lambda k: (k[1], k[0])):
        b = baseline.get(key)
        c = current.get(key)
        if b != c:
            drift.append((key, b, c))
    return drift


def evaluate():
    """Build once, return (frame, records, csv_hash, agg_hash, problems)."""
    frame = build_reference_frame()
    problems = []
    n = len(frame)
    if n != GOLDEN_ROWS:
        problems.append(f"row-count drift: {n} != {GOLDEN_ROWS}")
    h = csv_hash(frame)
    if h != GOLDEN_CSV_SHA256:
        problems.append(f"full-CSV hash drift: {h} != {GOLDEN_CSV_SHA256}")
    try:
        shipped = SHIPPED_PLAN_PATH.read_bytes()
        fingerprint = json.loads(SHIPPED_FINGERPRINT_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        problems.append(f"shipped plan or fingerprint is unreadable: {exc}")
    else:
        built = frame.to_csv(index=False).encode("utf-8")
        if built != shipped:
            problems.append("the rebuilt golden bytes differ from the shipped plan")
        shipped_hash = hashlib.sha256(shipped).hexdigest()
        if shipped_hash != fingerprint.get("sha256"):
            problems.append("the shipped plan differs from its committed fingerprint")
    records = aggregate_records(frame)
    ah = agg_hash(records)
    if ah != GOLDEN_AGG_SHA256:
        drift = per_day_drift(records)
        detail = "\n".join(
            f"  {c} {d}: baseline={b} current={cur}" for (c, d), b, cur in drift[:40]
        )
        more = "" if len(drift) <= 40 else f"\n  ... and {len(drift) - 40} more"
        problems.append(
            f"per-channel-day aggregate drift ({len(drift)} day(s)):\n{detail}{more}"
        )
    return frame, records, h, ah, problems


def test_golden_weekly_schedule():
    _, _, _, _, problems = evaluate()
    assert not problems, "Weekly-schedule golden-master drift:\n" + "\n".join(problems)


def main() -> int:
    frame, records, h, ah, problems = evaluate()
    print(f"rows: {len(frame)} (golden {GOLDEN_ROWS})")
    print(f"full-CSV sha256:  {h}")
    print(f"  matches golden: {h == GOLDEN_CSV_SHA256}")
    print(f"channel-days: {len(records)} (golden {len(GOLDEN_AGG)})")
    print(f"aggregate sha256: {ah}")
    print(f"  matches golden: {ah == GOLDEN_AGG_SHA256}")
    if problems:
        print("DRIFT:")
        for p in problems:
            print(p)
        return 1
    print("GOLDEN OK: schedule reproduces the committed baseline exactly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
