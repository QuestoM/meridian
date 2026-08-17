"""Seed competing day versions for the demonstration store.

Development fixture, same philosophy as ``seed_trade_agreements.py``: nothing
is invented. Every proposal is priced by the real engine through the same
``rows_api.proposal_rows`` the API route calls, against the plan of record as
it stands, and written through the same store. The only thing this script
supplies that the API route would not is the author names — the route reads
the login session, and a seed has none.

Run from the repository root:

    ~/.venvs/meridian/bin/python scripts/seed_day_proposals.py [DAY]
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kairos_api import day_proposal_rows as rows_api  # noqa: E402
from kairos_api import day_proposal_store as store  # noqa: E402


def main() -> None:
    day = sys.argv[1] if len(sys.argv) > 1 else ""
    if not day:
        from kairos_api import break_store

        days = break_store.plan_days()
        if not days:
            raise SystemExit("no operator plan covers any day; recompute first")
        day = days[0]

    baseline = rows_api.proposal_rows(day, [])
    rows = baseline["rows"]  # segment-level frame: segment_id, num_breaks, money
    print(f"day {baseline['day']} on {baseline['channel']}: {len(rows)} baseline rows")
    if len(store.list_for_day(baseline["channel"], baseline["day"])) > 0:
        raise SystemExit("this day already has proposals; seeding again would duplicate")

    # A break is addressed <segment_id>~<ordinal>; the two edits below target
    # the first break of the two highest-revenue segments that hold one.
    import pandas as pd

    frame = rows.copy()
    frame["predicted_revenue"] = pd.to_numeric(frame["predicted_revenue"], errors="coerce")
    frame["num_breaks"] = pd.to_numeric(frame["num_breaks"], errors="coerce").fillna(0)
    holders = frame[frame["num_breaks"] >= 1].sort_values("predicted_revenue", ascending=False)
    if len(holders) < 2:
        raise SystemExit("fewer than two break-holding segments; nothing to vary")
    from kairos_api.break_store import break_id

    top = break_id(str(holders.iloc[0]["segment_id"]), 1)
    second = break_id(str(holders.iloc[1]["segment_id"]), 1)

    def create(name: str, author: str, note: str, moves: list[dict]) -> None:
        built = rows_api.proposal_rows(day, moves)
        edit_map = {str(m["break_id"]): {k: v for k, v in m.items()
                                          if k != "break_id" and v is not None}
                    for m in moves}
        manifest = store.create_proposal(
            channel=built["channel"], date=built["day"], name=name, author=author,
            rows=built["rows"], baseline_ref=built["baseline_ref"], edits=edit_map,
            note=note, rows_source=built["rows_source"], engine=built["engine"],
        )
        print(f"  {manifest['proposal_id']}: {name} — {author}")

    create(
        "היום כפי שהמנוע מתכנן", "דנה לוי",
        "נקודת הפתיחה המשותפת: בלי עריכות, המחיר שהמנוע נותן ליום כמות שהוא",
        [],
    )
    create(
        "ברייק זהב במוקד ההכנסה", "יואב כהן",
        "הברייק המכניס ביותר מסומן זהב; המנוע מתמחר מחדש את היום סביבו",
        [{"break_id": top, "is_gold": True}],
    )
    create(
        "הרחבת שני המוקדים", "מיכל אברהם",
        "שני הברייקים המכניסים מוארכים בחצי דקה; בדיקה אם ההרחבה מצדיקה את עלות הריטנשן",
        [
            {"break_id": top, "duration_seconds": 150.0},
            {"break_id": second, "duration_seconds": 150.0},
        ],
    )
    print("seeded", len(store.list_for_day(baseline["channel"], baseline["day"])), "proposals")


if __name__ == "__main__":
    main()
