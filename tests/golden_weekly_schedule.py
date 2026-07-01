"""Golden-master safety net for the exported weekly break schedule.

Runs :func:`kairos.export.schedule.build_weekly_schedule` on the committed
reference data along the exact path ``POST /api/recompute-schedule`` uses
(saved settings, ``revenue_weight`` / 100, saved ``risk_lambda``, saved
``operator_channel``, ``today=date.today()``), then asserts the output is the
committed golden to the byte:

  * a full-CSV content hash (the CSV carries no timestamp column, so its bytes
    are the whole content), and
  * a per-channel-day aggregate hash over ``predicted_revenue``,
    ``predicted_retention`` and ``num_breaks``.

Any drift in a single per-day total flips both hashes; the test then diffs the
recomputed aggregate against the embedded baseline and names the channel-days
that moved. This gates the Phase-1 engine-core consolidation: the consolidated
engine must reproduce this schedule exactly.

A full run optimises every channel-day and takes roughly 45-70s, which is
acceptable for a safety net. The demand-signal inputs on disk (a header-only
``campaign_flights.csv`` and no inventory file) make pacing and inventory exact
identities, so the schedule does not depend on the run date and the golden is
stable across days.

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

# Committed golden, captured from a real recompute on the current engine.
GOLDEN_CSV_SHA256 = "eb24f6e4503e00baae31b2e57e643f08309ca08274b957db87287d93c0e90af8"
GOLDEN_ROWS = 8704
GOLDEN_AGG_SHA256 = "c62addcb23431fb05b82280e7a2a9e03e1fe487767985ae6c13131576cf73842"

# Per-channel-day aggregate baseline, one entry per channel-day as
# [channel, date, predicted_revenue, predicted_retention, num_breaks], sorted by
# (date, channel). Embedded so a drift is pinpointed to the exact channel-day
# with no external file. This is the committed baseline the harness stores.
_BASELINE_AGG_JSON = r'''[["כאן 11","2024-11-01",842276.43,70.1822,74],["עכשיו 14","2024-11-01",480422.23,26.5308,39],["קשת 12","2024-11-01",3450224.79,92.9314,80],["רשת 13","2024-11-01",1060339.25,78.9801,80],["כאן 11","2024-11-02",1685241.62,80.8684,80],["עכשיו 14","2024-11-02",900821.68,14.8509,29],["קשת 12","2024-11-02",3951207.74,91.9093,80],["רשת 13","2024-11-02",1493420.8,71.9633,80],["כאן 11","2024-11-03",787742.42,75.8234,80],["עכשיו 14","2024-11-03",1660387.2,51.0398,76],["קשת 12","2024-11-03",3766795.05,85.8474,80],["רשת 13","2024-11-03",1551207.86,87.9547,80],["כאן 11","2024-11-04",790071.97,78.8149,80],["עכשיו 14","2024-11-04",1558103.15,50.1244,74],["קשת 12","2024-11-04",3244118.89,86.8733,80],["רשת 13","2024-11-04",1609378.78,84.95,80],["כאן 11","2024-11-05",660235.47,65.1294,73],["עכשיו 14","2024-11-05",1811393.58,44.188,73],["קשת 12","2024-11-05",3981116.58,90.8767,80],["רשת 13","2024-11-05",1332660.1,80.9457,80],["כאן 11","2024-11-06",767212.08,57.1755,71],["עכשיו 14","2024-11-06",2129539.64,42.167,73],["קשת 12","2024-11-06",4095805.69,69.0906,75],["רשת 13","2024-11-06",1435022.69,61.2107,73],["כאן 11","2024-11-07",687940.7,80.8305,80],["עכשיו 14","2024-11-07",1889373.48,51.9887,77],["קשת 12","2024-11-07",3836666.88,86.837,80],["רשת 13","2024-11-07",1512215.55,81.9393,80],["כאן 11","2024-11-08",940302.24,50.3417,70],["עכשיו 14","2024-11-08",613212.49,20.6441,36],["קשת 12","2024-11-08",3557732.04,70.2504,72],["רשת 13","2024-11-08",1260249.49,62.0575,78],["כאן 11","2024-11-09",1044743.16,77.8978,80],["עכשיו 14","2024-11-09",948692.21,15.9405,27],["קשת 12","2024-11-09",4172732.4,95.909,80],["רשת 13","2024-11-09",1334340.34,79.9626,80],["כאן 11","2024-11-10",629269.34,68.8709,80],["עכשיו 14","2024-11-10",1601141.45,50.0405,76],["קשת 12","2024-11-10",3750443.62,87.8721,80],["רשת 13","2024-11-10",1415443.21,89.9407,80],["כאן 11","2024-11-11",694560.41,78.8807,80],["עכשיו 14","2024-11-11",1423623.59,43.2949,70],["קשת 12","2024-11-11",3842616.08,84.8518,80],["רשת 13","2024-11-11",1272812.13,91.9599,80],["כאן 11","2024-11-12",1095463.71,78.8495,80],["עכשיו 14","2024-11-12",1380020.2,46.911,80],["קשת 12","2024-11-12",3923854.07,87.8942,80],["רשת 13","2024-11-12",1161251.33,80.9487,80],["כאן 11","2024-11-13",859290.13,72.8666,80],["עכשיו 14","2024-11-13",1539451.76,48.1509,73],["קשת 12","2024-11-13",3362862.15,85.897,80],["רשת 13","2024-11-13",1342156.72,87.9498,80],["כאן 11","2024-11-14",684851.67,67.895,80],["עכשיו 14","2024-11-14",1781991.42,44.9909,77],["קשת 12","2024-11-14",3493599.81,92.8707,80],["רשת 13","2024-11-14",1300836.61,83.9489,80],["כאן 11","2024-11-15",762715.79,68.2555,72],["עכשיו 14","2024-11-15",603596.5,26.4178,42],["קשת 12","2024-11-15",3035215.26,88.8967,80],["רשת 13","2024-11-15",1055842.68,80.9803,80],["כאן 11","2024-11-16",1247106.88,66.9153,80],["עכשיו 14","2024-11-16",880613.42,15.9451,27],["קשת 12","2024-11-16",3874875.43,91.8615,80],["רשת 13","2024-11-16",1288991.04,78.952,80],["כאן 11","2024-11-17",692380.72,68.8698,80],["עכשיו 14","2024-11-17",1638296.41,50.0763,75],["קשת 12","2024-11-17",3441343.29,85.8568,80],["רשת 13","2024-11-17",1328935.25,85.9439,80],["כאן 11","2024-11-18",753804.98,70.1854,71],["עכשיו 14","2024-11-18",1744114.23,49.9364,79],["קשת 12","2024-11-18",3568284.53,88.8325,80],["רשת 13","2024-11-18",1619971.29,90.9342,80],["כאן 11","2024-11-19",808801.53,68.8561,80],["עכשיו 14","2024-11-19",1616899.81,50.0541,76],["קשת 12","2024-11-19",3509020.13,87.8919,80],["רשת 13","2024-11-19",1405082.44,88.9551,80],["כאן 11","2024-11-20",724513.08,69.8668,80],["עכשיו 14","2024-11-20",1542521.24,49.0881,75],["קשת 12","2024-11-20",3316623.21,92.8829,80],["רשת 13","2024-11-20",1505498.09,86.9501,80],["כאן 11","2024-11-21",885655.62,66.8697,80],["עכשיו 14","2024-11-21",1671026.4,49.1062,74],["קשת 12","2024-11-21",3300793.51,89.8776,80],["רשת 13","2024-11-21",1387152.47,84.966,80],["כאן 11","2024-11-22",834514.32,63.9502,80],["עכשיו 14","2024-11-22",612323.85,28.3373,44],["קשת 12","2024-11-22",3102248.68,99.9481,80],["רשת 13","2024-11-22",836012.08,77.9748,80],["כאן 11","2024-11-23",998582.28,66.9023,80],["עכשיו 14","2024-11-23",900914.6,16.8919,28],["קשת 12","2024-11-23",4282860.83,96.9028,80],["רשת 13","2024-11-23",1530431.11,80.9573,80],["כאן 11","2024-11-24",896042.37,53.8868,80],["עכשיו 14","2024-11-24",1700039.61,47.0849,75],["קשת 12","2024-11-24",3812722.3,83.8986,80],["רשת 13","2024-11-24",1368315.08,76.9493,80],["כאן 11","2024-11-25",790843.58,69.894,79],["עכשיו 14","2024-11-25",1571835.5,53.006,77],["קשת 12","2024-11-25",3396954.7,90.8713,80],["רשת 13","2024-11-25",1257394.65,81.9493,80],["כאן 11","2024-11-26",807121.71,65.8677,80],["עכשיו 14","2024-11-26",1709978.22,52.0529,76],["קשת 12","2024-11-26",4052512.3,88.8676,80],["רשת 13","2024-11-26",1466217.38,79.9581,80],["כאן 11","2024-11-27",710419.91,67.861,80],["עכשיו 14","2024-11-27",1685652.34,51.9439,79],["קשת 12","2024-11-27",3407349.79,89.8865,80],["רשת 13","2024-11-27",1465658.96,88.9452,80],["כאן 11","2024-11-28",723983.39,70.9438,78],["עכשיו 14","2024-11-28",1823734.43,49.874,80],["קשת 12","2024-11-28",3104099.29,86.9132,80],["רשת 13","2024-11-28",1454514.1,85.9382,80],["כאן 11","2024-11-29",918129.23,68.9387,80],["עכשיו 14","2024-11-29",372601.6,36.0588,51],["קשת 12","2024-11-29",2742899.47,80.9174,80],["רשת 13","2024-11-29",1006538.06,78.0567,78],["כאן 11","2024-11-30",1320456.2,64.9131,80],["עכשיו 14","2024-11-30",915057.73,18.8572,29],["קשת 12","2024-11-30",3775836.41,79.924,80],["רשת 13","2024-11-30",1375725.68,75.9724,80]]'''
GOLDEN_AGG = json.loads(_BASELINE_AGG_JSON)


def settings_map() -> dict:
    """The saved dashboard settings, exactly as the recompute endpoint reads them."""
    return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))


def build_reference_frame():
    """Reproduce the ``POST /api/recompute-schedule`` commit path byte-for-byte."""
    settings = settings_map()
    return build_weekly_schedule(
        settings=settings,
        revenue_weight=settings["revenue_weight"] / 100.0,
        risk_lambda=settings["risk_lambda"],
        operator_channel=settings["operator_channel"],
        today=date.today(),
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
