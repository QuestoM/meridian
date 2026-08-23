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
#
# REBASED AGAIN 2026-08-24, twice over. First: PROMO ROWS ARE BREAKS, NOT
# INVENTORY (owner-delegated decision). The EPG is 59.1% promo/commercial-block
# rows and the optimizer was scheduling breaks inside them; a Promo-classified
# segment now carries max_breaks=0 at the builder. The plan moves 210,087,560 ->
# 191,860,240 ILS (-8.68%) and 8,960 -> 8,510 breaks; mean retention rises
# 68.8176 -> 68.9972. All 5,142 Promo segments remain in the plan with an honest
# zero. Second: the 2026-08-23 rebase of the PER-DAY BASELINE TABLE silently
# failed -- the regex targeted `GOLDEN_AGG = [...]` while the table lives in
# _BASELINE_AGG_JSON -- and nothing noticed because the per-day drift check only
# runs when the aggregate hash mismatches. This rebase replaces the table itself
# and VERIFIES the sum in-file before claiming success, which the failed one did
# not.
GOLDEN_CSV_SHA256 = "e0b3980f671fc2b7ef561d1ae79da79055317c1968948780e30e80ccf753f53a"
GOLDEN_ROWS = 8704
GOLDEN_AGG_SHA256 = "fa06eb6df70dc39ee86033ed8dd50c602ec94f2f7aa249d07e5df1c943bb7d8d"

# Per-channel-day aggregate baseline, one entry per channel-day as
# [channel, date, predicted_revenue, predicted_retention, num_breaks], sorted by
# (date, channel). Embedded so a drift is pinpointed to the exact channel-day
# with no external file. This is the committed baseline the harness stores.
_BASELINE_AGG_JSON = r'''[["כאן 11","2024-11-01",825657.39,69.218,77],["עכשיו 14","2024-11-01",460217.94,26.1169,39],["קשת 12","2024-11-01",2607802.45,92.5725,69],["רשת 13","2024-11-01",899901.7,78.4332,73],["כאן 11","2024-11-02",1569162.89,79.9938,80],["עכשיו 14","2024-11-02",827826.44,14.6277,27],["קשת 12","2024-11-02",3952047.16,90.9853,80],["רשת 13","2024-11-02",1363753.02,71.2175,77],["כאן 11","2024-11-03",710442.98,74.9321,80],["עכשיו 14","2024-11-03",1481125.98,50.4586,71],["קשת 12","2024-11-03",3304601.97,85.2956,73],["רשת 13","2024-11-03",1374658.39,87.1614,78],["כאן 11","2024-11-04",746527.33,77.9143,80],["עכשיו 14","2024-11-04",1389036.89,49.5444,69],["קשת 12","2024-11-04",3173741.44,86.1103,77],["רשת 13","2024-11-04",1412881.53,84.2542,76],["כאן 11","2024-11-05",555871.15,64.5045,69],["עכשיו 14","2024-11-05",1549138.35,43.8832,63],["קשת 12","2024-11-05",3795129.22,90.0217,79],["רשת 13","2024-11-05",1158073.88,80.206,77],["כאן 11","2024-11-06",699327.45,56.5848,67],["עכשיו 14","2024-11-06",1900932.27,41.6379,67],["קשת 12","2024-11-06",3516587.25,68.9672,60],["רשת 13","2024-11-06",1198514.22,60.8422,64],["כאן 11","2024-11-07",652751.39,79.9394,80],["עכשיו 14","2024-11-07",1651922.7,51.586,68],["קשת 12","2024-11-07",3441310.31,86.2747,73],["רשת 13","2024-11-07",1260509.91,81.3566,74],["כאן 11","2024-11-08",773040.22,49.8519,64],["עכשיו 14","2024-11-08",528284.14,20.3424,34],["קשת 12","2024-11-08",2610795.63,70.1952,57],["רשת 13","2024-11-08",945220.12,61.8701,64],["כאן 11","2024-11-09",1001074.31,77.0106,80],["עכשיו 14","2024-11-09",950748.1,15.6277,27],["קשת 12","2024-11-09",4165310.44,94.9778,80],["רשת 13","2024-11-09",1214038.1,79.0709,80],["כאן 11","2024-11-10",586133.8,67.9919,80],["עכשיו 14","2024-11-10",1447846.99,49.5025,70],["קשת 12","2024-11-10",3379452.18,87.1682,76],["רשת 13","2024-11-10",1195930.12,89.4539,72],["כאן 11","2024-11-11",680515.74,77.9753,80],["עכשיו 14","2024-11-11",1237733.26,43.0136,60],["קשת 12","2024-11-11",3262285.48,84.1088,77],["רשת 13","2024-11-11",1240077.86,91.1433,78],["כאן 11","2024-11-12",936809.88,78.0157,79],["עכשיו 14","2024-11-12",1241826.57,46.3906,73],["קשת 12","2024-11-12",3400373.35,87.3182,73],["רשת 13","2024-11-12",974218.95,80.2561,76],["כאן 11","2024-11-13",794692.89,71.9717,80],["עכשיו 14","2024-11-13",1353166.06,47.7466,65],["קשת 12","2024-11-13",3086452.97,85.0517,78],["רשת 13","2024-11-13",1137811.5,87.3076,75],["כאן 11","2024-11-14",660961.17,66.9919,80],["עכשיו 14","2024-11-14",1589936.54,44.4844,70],["קשת 12","2024-11-14",3206278.46,92.236,74],["רשת 13","2024-11-14",1198046.88,83.1567,78],["כאן 11","2024-11-15",713576.01,67.4352,73],["עכשיו 14","2024-11-15",518606.5,26.1168,39],["קשת 12","2024-11-15",2470299.61,88.5703,69],["רשת 13","2024-11-15",925040.21,80.3847,74],["כאן 11","2024-11-16",1231404.53,66.0338,80],["עכשיו 14","2024-11-16",880810.7,15.6315,27],["קשת 12","2024-11-16",3672103.85,90.9424,80],["רשת 13","2024-11-16",1111793.25,78.1176,79],["כאן 11","2024-11-17",666689.2,67.9847,80],["עכשיו 14","2024-11-17",1446722.67,49.595,68],["קשת 12","2024-11-17",2870264.11,85.4071,71],["רשת 13","2024-11-17",1160935.45,85.3,75],["כאן 11","2024-11-18",679148.37,69.4999,69],["עכשיו 14","2024-11-18",1534063.1,49.5051,70],["קשת 12","2024-11-18",2940834.31,88.1435,76],["רשת 13","2024-11-18",1338780.74,90.4444,72],["כאן 11","2024-11-19",750409.26,67.9829,80],["עכשיו 14","2024-11-19",1421271.97,49.5548,69],["קשת 12","2024-11-19",3211959.16,87.0698,78],["רשת 13","2024-11-19",1180451.09,88.2054,77],["כאן 11","2024-11-20",669593.38,69.1326,77],["עכשיו 14","2024-11-20",1371830.44,48.556,69],["קשת 12","2024-11-20",3078563.38,92.1147,77],["רשת 13","2024-11-20",1234490.41,86.2103,77],["כאן 11","2024-11-21",809652.41,66.0044,80],["עכשיו 14","2024-11-21",1494183.3,48.5858,68],["קשת 12","2024-11-21",3155398.01,89.1491,76],["רשת 13","2024-11-21",1252833.58,84.1508,78],["כאן 11","2024-11-22",746581.46,63.1687,78],["עכשיו 14","2024-11-22",533747.74,28.0606,40],["קשת 12","2024-11-22",2572824.24,99.4622,71],["רשת 13","2024-11-22",679091.31,77.5796,70],["כאן 11","2024-11-23",955700.12,66.0221,80],["עכשיו 14","2024-11-23",862531.66,16.5711,28],["קשת 12","2024-11-23",3950975.5,96.0233,80],["רשת 13","2024-11-23",1289214.4,80.0712,80],["כאן 11","2024-11-24",842017.91,52.9923,80],["עכשיו 14","2024-11-24",1470796.19,46.6466,67],["קשת 12","2024-11-24",3470872.68,83.2652,74],["רשת 13","2024-11-24",1159675.1,76.4099,73],["כאן 11","2024-11-25",735857.32,69.0341,79],["עכשיו 14","2024-11-25",1323788.21,52.5526,69],["קשת 12","2024-11-25",3042437.96,90.1002,77],["רשת 13","2024-11-25",1156769.15,81.3949,73],["כאן 11","2024-11-26",778312.53,64.9934,80],["עכשיו 14","2024-11-26",1464970.17,51.5115,70],["קשת 12","2024-11-26",3646347.38,87.9771,80],["רשת 13","2024-11-26",1220137.11,79.41,73],["כאן 11","2024-11-27",667561.88,66.9931,80],["עכשיו 14","2024-11-27",1451008.17,51.5115,70],["קשת 12","2024-11-27",3187812.32,89.1194,77],["רשת 13","2024-11-27",1199441.01,88.4492,72],["כאן 11","2024-11-28",698901.21,70.0074,80],["עכשיו 14","2024-11-28",1574509.92,49.3909,72],["קשת 12","2024-11-28",2971728.08,86.251,74],["רשת 13","2024-11-28",1247116.42,85.3076,75],["כאן 11","2024-11-29",822884.63,68.1196,79],["עכשיו 14","2024-11-29",334026.63,35.7184,47],["קשת 12","2024-11-29",2267996.39,80.5146,70],["רשת 13","2024-11-29",806117.41,77.6446,69],["כאן 11","2024-11-30",1303611.76,64.0367,80],["עכשיו 14","2024-11-30",863181.83,18.579,28],["קשת 12","2024-11-30",3314593.08,79.0085,80],["רשת 13","2024-11-30",1176876.71,75.0712,80]]'''
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
