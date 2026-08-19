"""How much of a coming week each activated factor can actually speak about.

A family's gate percentage answers "how much better is the model where this
factor applies", and it is read as "how much better is the plan". Those are the
same sentence only when the factor applies to the plan, and MEASURED on the
files in this repository one of them does not:

* ``weekday_slot`` cells are a weekday crossed with a slot band. Every future
  broadcast has both, so the family reaches 100% of any week.
* ``series`` cells are fitted programme titles. On the 704 broadcasts pulled for
  the fortnight of 2026-08-18 it finds a cell for **57 of them, 8.1%** (כאן 11
  5.0%, קשת 12 8.3%, עכשיו 14 11.5%), because the factor is substantially
  memorising titles rather than encoding an identity that carries forward.
  ``docs/liveness-and-the-gate.md`` holds the measurement that establishes that:
  held-out accuracy tracks cell COUNT, and the raw unnormalised title scores
  0.63299 against the shipped key's 0.63075.

Neither fact is a reason to change a key. Both are a reason to stop reading one
number as if it were the other, which is what this module exists to prevent: it
states, per family, the share of a given forward schedule that family can reach.

It is a measurement and never a gate. Nothing here activates, deactivates or
weights anything; a factor that reaches 8% of the week is still the best thing
available on the 8%, and the pooled base answers the rest. What changes is that
the operator is told which is happening.

Reach is not accuracy. A family that reaches every row can still be wrong on all
of them, and this module says nothing about that; the gate does.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from kairos.model.audience_factors import family_cells

# Where the forward schedule this engine pulls actually lands. Named here rather
# than passed by every caller so a reach report and the feed cannot disagree
# about which week they are describing.
FORWARD_SCHEDULE = Path("data/reference/CompetitorProgrammes.csv")


def _cells_of(model: Any, family: str) -> set[str]:
    """The cell keys a fitted family holds, or an empty set if it holds none."""
    factor = (getattr(model, "factors", None) or {}).get(family)
    if isinstance(factor, Mapping):
        table = factor.get("cells", factor)
        if isinstance(table, Mapping):
            return {str(key) for key in table}
    return set()


def reach(model: Any, schedule, families: Optional[Sequence[str]] = None) -> dict[str, Any]:
    """Per family, the share of ``schedule`` rows whose cell the model holds.

    ``schedule`` carries :data:`PREDICTION_COLUMNS` and is put through the SAME
    :func:`prediction_frame` the forecast uses. Deriving the cells here instead
    would let this report and the forecast disagree about what a cell is, which
    on the two keys in this repository is not hypothetical: the contract's own
    ``SeriesKey`` column is ``series_join_key`` while the fitted cells are keyed
    by ``canonicalize_series``, and reading the wrong one produces a coverage
    figure that is confidently wrong.

    A row that cannot be given a cell counts as unreached, which is what the
    forecast does with it: no factor, and the pooled base answers.
    """
    from kairos.model.audience_frame import prediction_frame

    names = list(families) if families is not None else list(getattr(model, "factors", {}) or {})
    total = int(len(schedule))
    report: dict[str, Any] = {"rows": total, "families": {}}
    if total == 0:
        report["note"] = "the forward schedule is empty, so no reach can be measured"
        return report
    scored = prediction_frame(schedule).reset_index(drop=True)
    for family in names:
        known = _cells_of(model, family)
        try:
            cells = family_cells(scored, family)
        except (KeyError, AttributeError, TypeError, ValueError):
            report["families"][family] = {
                "reached": None, "share": None,
                "note": f"this schedule carries nothing the {family} family reads",
            }
            continue
        reached = sum(1 for cell in cells if cell and str(cell) in known)
        report["families"][family] = {
            "reached": reached,
            "share": round(reached / total, 4),
            "note": (
                f"the {family} factor has a cell for every one of the {total} "
                f"broadcasts in this schedule"
                if reached == total else
                f"the {family} factor has a cell for {reached} of {total} "
                f"broadcasts in this schedule ({100.0 * reached / total:.1f} percent); "
                f"the pooled base answers the other {total - reached}"
            ),
        }
    return report


def forward_rows(path: str | Path = FORWARD_SCHEDULE):
    """The pulled competitor schedule as prediction rows, or None if absent.

    Absent is a real and common answer. It is returned as None rather than as an
    empty frame so a caller cannot report "this factor reaches 0% of the week"
    when the truth is that no week was pulled.
    """
    import pandas as pd

    target = Path(path)
    if not target.exists():
        return None
    frame = pd.read_csv(target, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    if frame.empty or "Start time" not in frame.columns:
        return None
    clock = frame["Start time"].astype(str).str.split(":", expand=True)
    seconds = (
        pd.to_numeric(clock[0], errors="coerce").fillna(0) * 3600
        + pd.to_numeric(clock[1], errors="coerce").fillna(0) * 60
        + (pd.to_numeric(clock[2], errors="coerce").fillna(0) if clock.shape[1] > 2 else 0)
    )
    return pd.DataFrame({
        "date": pd.to_datetime(frame["Date"], format="%d/%m/%Y", errors="coerce"),
        "channel": frame["Channel"].astype(str),
        "program_title": frame["Title"].astype(str),
        "start_seconds": seconds.astype(float),
        "duration_seconds": pd.to_numeric(frame.get("Duration"), errors="coerce"),
    })


def forward_reach(model: Any, path: str | Path = FORWARD_SCHEDULE) -> Optional[dict[str, Any]]:
    """Reach against the schedule this engine last pulled, or None if none was."""
    rows = forward_rows(path)
    if rows is None:
        return None
    report = reach(model, rows)
    report["schedule"] = str(path)
    report["channels"] = sorted({str(c) for c in rows["channel"]})
    return report


def sentence(report: Mapping[str, Any], family: str) -> str:
    """One line an operator can read, or an honest silence."""
    entry = (report.get("families") or {}).get(family)
    if not isinstance(entry, Mapping):
        return ""
    return str(entry.get("note") or "")
