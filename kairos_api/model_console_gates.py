"""Every gate the two models ran, in one ledger, each verdict with its basis.

The verdicts already exist. They are written in full sentences into the two
artifacts and, until this module, they were rendered on an operator's calendar
page as a grey chip that said "Off" for three completely different pieces of
news. This module states the difference, because it is the difference a model
steward is paid to read:

- **Tested and lost.** A gate ran, produced a real held-out figure, and the
  figure did not clear the bar. There is nothing to wait for; the factor is not
  in the data.
- **No contrast.** A gate could not run, because the training window holds no
  observations on both sides of the question. Waiting is exactly the remedy,
  and :mod:`kairos_api.model_console_coverage` says what would end the wait.
- **Not yet measured.** No gate record exists for a factor the model knows
  about. Nobody has asked the question.

A gate's basis is the number that decided it, its bar, how many observations
and folds it was measured on, and the artifact's own sentence. Where the
artifact does not carry the bar as a number, that is said rather than guessed:
the sentence carries it, and the sentence is printed verbatim.

Gates are separated from **layers** here, and that separation is the second
honest thing this module does. A gate is decided by a measurement. A layer is
decided by a person, with the measurement recorded either way. Putting the two
in one table would let an owner's choice read as evidence.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from kairos_api import model_console_artifacts as artifacts

ACTIVE = "active"
TESTED_AND_LOST = "tested_and_lost"
NO_CONTRAST = "no_contrast"
NOT_MEASURED = "not_measured"

STATE_LABELS = {
    ACTIVE: {"en": "Active", "he": "פעיל"},
    TESTED_AND_LOST: {"en": "Tested and lost", "he": "נבחן ולא עבר"},
    NO_CONTRAST: {"en": "No contrast", "he": "אין ניגודיות"},
    NOT_MEASURED: {"en": "Not yet measured", "he": "טרם נמדד"},
}

STATE_MEANINGS = {
    ACTIVE: {
        "en": "The gate ran, the factor beat its bar, and runs use it.",
        "he": "השער רץ, הגורם עבר את הרף, וההרצות משתמשות בו.",
    },
    TESTED_AND_LOST: {
        "en": "The gate ran and the measured figure did not clear the bar. More of the same data will not change this.",
        "he": "השער רץ והמספר הנמדד לא עבר את הרף. עוד נתונים מאותו סוג לא ישנו זאת.",
    },
    NO_CONTRAST: {
        "en": "The gate could not run: the window holds observations on only one side of the question.",
        "he": "השער לא יכול היה לרוץ: החלון מכיל תצפיות רק בצד אחד של השאלה.",
    },
    NOT_MEASURED: {
        "en": "No gate record exists for this factor. The question has not been asked.",
        "he": "לא קיים רישום שער לגורם הזה. השאלה לא נשאלה.",
    },
}

# The audience model's factor families, in disclosure order, with the words the
# product already uses for them.
AUDIENCE_FAMILIES: dict[str, dict[str, str]] = {
    "weekday_slot": {"en": "Weekday and slot", "he": "יום ורצועה"},
    "series": {"en": "Series", "he": "סדרה"},
    "calendar_school_and_chol_hamoed": {"en": "School holidays and Chol HaMoed", "he": "חול המועד וחופשות"},
    "calendar_hanukkah": {"en": "Hanukkah", "he": "חנוכה"},
    "calendar_religious_blackout": {"en": "Shabbat and holy days", "he": "שבתות וימים טובים"},
    "season": {"en": "Season", "he": "עונה"},
    "operator_events": {"en": "Operator events", "he": "אירועי מפעיל"},
    "competitor_lineup": {"en": "Competing lineup", "he": "ליינאפ מתחרים"},
}

_HELD_OUT_RMSE = {
    "en": "held-out RMSE improvement over the pooled base, averaged over temporal folds",
    "he": "שיפור RMSE במבחן מוחזק מול הבסיס המאוחד, בממוצע על פני קיפולים כרונולוגיים",
}


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _as_int(value: Any) -> Optional[int]:
    number = _number(value)
    return None if number is None else int(number)


def _pct(value: Any) -> Optional[float]:
    number = _number(value)
    return None if number is None else round(number * 100, 4)


# Every bar below is read from the engine constant that actually applies it, so
# no threshold is ever asserted in this file. A constant that moves moves here,
# and a constant that disappears reports an honest absence rather than a stale
# number. ``bar_source`` names the module and the constant for each one.
_BAR_SOURCES = {
    "series": ("kairos.model.series_gate", "SERIES_GATE_MIN_RELATIVE_IMPROVEMENT", 100.0),
    "counterprogramming": ("kairos.model.competitor_gate", "COUNTERPROGRAMMING_MIN_RELATIVE_IMPROVEMENT", 100.0),
    "event_layer": ("kairos.model.event_gate", "EVENT_GATE_MIN_RELATIVE_IMPROVEMENT", 100.0),
    "detrend_seasonality": ("kairos.model.detrend_gate", "DETREND_GATE_MIN_RELATIVE_IMPROVEMENT", 100.0),
    "audience": ("kairos.model.audience_factors", "AUDIENCE_GATE_MIN_RELATIVE_IMPROVEMENT", 100.0),
    "first_break": ("kairos.model.measure", "_FIRST_BREAK_MIN_P", 1.0),
}


def engine_bar(name: str) -> "tuple[Optional[float], str]":
    """The bar this gate is measured against, and where it was read from."""
    module_name, constant, scale = _BAR_SOURCES[name]
    try:
        import importlib

        value = getattr(importlib.import_module(module_name), constant)
    except Exception:  # pragma: no cover - a missing constant is reported, not guessed
        return None, f"{module_name}.{constant} is not available in this build"
    number = _number(value)
    if number is None:
        return None, f"{module_name}.{constant} is not a number in this build"
    return round(number * scale, 6), f"{module_name}.{constant}"


def _state_from(verdict: str, measured: Optional[float]) -> str:
    if verdict == "on":
        return ACTIVE
    if verdict == "unknown":
        return NOT_MEASURED
    return TESTED_AND_LOST if measured is not None else NO_CONTRAST


def _row(gate_id: str, model: str, family: str, labels: dict[str, str], state: str,
         reason: str, basis: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": gate_id,
        "model": model,
        "family": family,
        "label_en": labels["en"],
        "label_he": labels["he"],
        "state": state,
        "state_label_en": STATE_LABELS[state]["en"],
        "state_label_he": STATE_LABELS[state]["he"],
        "state_meaning_en": STATE_MEANINGS[state]["en"],
        "state_meaning_he": STATE_MEANINGS[state]["he"],
        "reason": reason,
        "basis": basis,
    }


def _basis(statistic: dict[str, str], value: Optional[float], unit: str,
           bar: Optional[float], bar_unit: str, bar_source: str,
           n: Optional[int], folds: Optional[int], fold_sd: Optional[float],
           measured_at: Optional[str], detail: Any) -> dict[str, Any]:
    return {
        "statistic_en": statistic["en"],
        "statistic_he": statistic["he"],
        "value": value,
        "unit": unit,
        "bar": bar,
        "bar_unit": bar_unit,
        "bar_source": bar_source,
        "n": n,
        "folds": folds,
        "fold_sd": fold_sd,
        "measured_at": measured_at,
        "detail": detail,
    }


# ---------------------------------------------------------------------------
# The audience model: eight families, one uniform gate record each
# ---------------------------------------------------------------------------


def audience_rows(gates: Optional[dict[str, Any]] = None,
                  n_observations: Optional[int] = None) -> list[dict[str, Any]]:
    records = artifacts.audience_gates() if gates is None else gates
    rows: list[dict[str, Any]] = []
    bar, bar_source = engine_bar("audience")
    families = list(AUDIENCE_FAMILIES.items())
    extra = [(key, {"en": key, "he": key}) for key in sorted(records)
             if key not in AUDIENCE_FAMILIES]
    for family, labels in families + extra:
        record = records.get(family)
        if not isinstance(record, dict):
            rows.append(_row(
                f"audience.{family}", "audience", family, labels, NOT_MEASURED,
                "The artifact carries no gate record for this family.",
                _basis(_HELD_OUT_RMSE, None, "percent", bar, "percent", bar_source,
                       n_observations, None, None, None, None)))
            continue
        measured = _number(record.get("held_out_delta_pct"))
        verdict = str(record.get("verdict") or "unknown")
        rows.append(_row(
            f"audience.{family}", "audience", family, labels,
            _state_from(verdict if verdict in ("on", "off") else "unknown", measured),
            str(record.get("reason") or ""),
            _basis(_HELD_OUT_RMSE, measured, "percent", bar, "percent", bar_source,
                   n_observations, _folds(record), None,
                   record.get("measured_at"), record)))
    return rows


def _folds(record: dict[str, Any]) -> Optional[int]:
    """The fold count the gate's own reason states, never a default.

    The audience gate writes its fold count into the sentence rather than into
    a field, so it is read from the sentence or reported as unknown.
    """
    import re

    match = re.search(r"over (\d+) temporal folds", str(record.get("reason") or ""))
    return int(match.group(1)) if match else None


# ---------------------------------------------------------------------------
# The retention coefficients: five gates, each written into metadata its own way
# ---------------------------------------------------------------------------


def _holdout_gate(md: dict[str, Any], *, gate_id: str, family: str, labels: dict[str, str],
                  active_key: str, holdout_key: str, reason_key: str, bar_name: str,
                  statistic: dict[str, str], n_key: str = "n_test") -> dict[str, Any]:
    """A gate whose artifact record is a held-out block with a relative improvement.

    Four of the five retention gates are written this way, so they are read one
    way. The bar comes from the engine constant that applies it; the artifact's
    own ``min_relative_improvement`` is carried in the detail beside it, and a
    test asserts the two agree wherever the artifact records one.
    """
    holdout = md.get(holdout_key) if isinstance(md.get(holdout_key), dict) else {}
    improvement = _number(holdout.get("relative_improvement"))
    bar, bar_source = engine_bar(bar_name)
    return _row(
        gate_id, "retention", family, labels,
        ACTIVE if md.get(active_key) is True else _state_from("off", improvement),
        str(md.get(reason_key) or ""),
        _basis(statistic, _pct(improvement), "percent", bar, "percent", bar_source,
               _as_int(holdout.get(n_key)), _as_int(holdout.get("folds")),
               _number(holdout.get("fold_sd")), md.get("computed_at"), holdout))


def _series_gate(md: dict[str, Any]) -> dict[str, Any]:
    return _holdout_gate(
        md, gate_id="retention.series", family="series_layer",
        labels={"en": "Series-level retention", "he": "שימור ברמת הסדרה"},
        active_key="series_layer_active", holdout_key="series_gate_holdout",
        reason_key="series_gate_reason", bar_name="series",
        statistic={"en": "held-out RMSE of series cells against genre cells, fold mean",
                   "he": "RMSE במבחן מוחזק של תאי סדרה מול תאי ז'אנר, ממוצע קיפולים"})


def _counterprogramming_gate(md: dict[str, Any]) -> dict[str, Any]:
    return _holdout_gate(
        md, gate_id="retention.counterprogramming", family="counterprogramming",
        labels={"en": "Counter-programming covariate", "he": "משתנה תוכנית נגדית"},
        active_key="counterprogramming_active", holdout_key="counterprogramming_holdout",
        reason_key="counterprogramming_reason", bar_name="counterprogramming",
        statistic={"en": "held-out RMSE with the covariate against without it, fold mean",
                   "he": "RMSE במבחן מוחזק עם המשתנה מול בלעדיו, ממוצע קיפולים"})


def _seasonality_gate(md: dict[str, Any]) -> dict[str, Any]:
    return _holdout_gate(
        md, gate_id="retention.detrend_seasonality", family="detrend_seasonality",
        labels={"en": "Seasonal detrend baseline", "he": "בסיס ניכוי מגמה עונתי"},
        active_key="detrend_seasonality_recommended", holdout_key="detrend_seasonality_holdout",
        reason_key="detrend_seasonality_reason", bar_name="detrend_seasonality",
        statistic={"en": "held-out RMSE of a month-and-minute baseline against the global one",
                   "he": "RMSE במבחן מוחזק של בסיס חודש-ודקה מול הבסיס הגלובלי"},
        n_key="n_test_minutes")


def _first_break_gate(md: dict[str, Any]) -> dict[str, Any]:
    p_value = _number(md.get("first_break_p_value"))
    n_first = _as_int(md.get("first_break_n_first"))
    n_later = _as_int(md.get("first_break_n_later"))
    bar, bar_source = engine_bar("first_break")
    detail = {key: md.get(key) for key in md if key.startswith("first_break_")}
    return _row(
        "retention.first_break", "retention", "first_break",
        {"en": "First break in a programme", "he": "הברייק הראשון בתוכנית"},
        ACTIVE if md.get("first_break_active") is True else _state_from("off", p_value),
        str(md.get("first_break_reason") or ""),
        _basis({"en": "two-sample p-value on the log effect of first breaks against later ones",
                "he": "ערך p בשני מדגמים על האפקט הלוגריתמי של ברייקים ראשונים מול מאוחרים"},
               p_value, "p", bar, "p", bar_source,
               None if n_first is None or n_later is None else n_first + n_later,
               None, None, md.get("computed_at"), detail))


def _event_layer_gate(md: dict[str, Any]) -> dict[str, Any]:
    gate = md.get("event_layer_gate") if isinstance(md.get("event_layer_gate"), dict) else {}
    verdict = str(gate.get("verdict") or "unknown")
    measured = _number(gate.get("held_out_delta_pct"))
    bar, bar_source = engine_bar("event_layer")
    state = NOT_MEASURED if not gate else _state_from(
        verdict if verdict in ("on", "off") else "unknown", measured)
    return _row(
        "retention.event_layer", "retention", "event_layer",
        {"en": "Operator events (retention)", "he": "אירועי מפעיל (שימור)"},
        state,
        str(gate.get("reason") or "The artifact carries no event-layer gate record."),
        _basis(_HELD_OUT_RMSE, measured, "percent", bar, "percent", bar_source,
               _as_int(md.get("total_breaks_measured")), None, None,
               gate.get("measured_at") or md.get("computed_at"), gate))


_RETENTION_GATES: tuple[Callable[[dict[str, Any]], dict[str, Any]], ...] = (
    _first_break_gate,
    _series_gate,
    _counterprogramming_gate,
    _event_layer_gate,
    _seasonality_gate,
)


def retention_rows(metadata: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
    md = artifacts.retention_metadata() if metadata is None else metadata
    if not md:
        return [_row(f"retention.{name}", "retention", name,
                     {"en": name.replace("_", " "), "he": name},
                     NOT_MEASURED,
                     "No retention coefficients artifact is on disk, so no gate has run.",
                     _basis(_HELD_OUT_RMSE, None, "percent", None, "percent",
                            "no artifact", None, None, None, None, None))
                for name in ("first_break", "series", "counterprogramming",
                             "event_layer", "detrend_seasonality")]
    return [build(md) for build in _RETENTION_GATES]


# ---------------------------------------------------------------------------
# Layers: decided by a person, measured either way
# ---------------------------------------------------------------------------

LAYER_SPECS = (
    ("placebo_correction", "placebo_correction_active", "placebo_correction_reason",
     {"en": "Placebo drift correction", "he": "תיקון סחיפת פלצבו"}),
    ("interval_calibration", None, "interval_calibration_reason",
     {"en": "Interval calibration", "he": "כיול רווחי הביטחון"}),
    ("moderated_variances", "moderated_variances", None,
     {"en": "Moderated variances", "he": "שונויות ממותנות"}),
)

LAYER_NOTES = {
    "placebo_correction": {
        "en": "There is no automatic gate: applying it moves the per-break charge, so it is an owner decision. The drift is measured either way.",
        "he": "אין שער אוטומטי: הפעלתו מזיזה את החיוב לכל ברייק, ולכן זו החלטת בעלים. הסחיפה נמדדת כך או כך.",
    },
    "interval_calibration": {
        "en": "There is no automatic gate: the calibrated bands widen the lower bound the risk setting prices. The width factor is measured either way.",
        "he": "אין שער אוטומטי: הרצועות המכוילות מרחיבות את הגבול התחתון שהגדרת הסיכון מתמחרת. מקדם הרוחב נמדד כך או כך.",
    },
    "moderated_variances": {
        "en": "It never self-activates, because it moves the point coefficients. The prior degrees of freedom are measured either way.",
        "he": "לעולם אינו מופעל מעצמו, כי הוא מזיז את המקדמים עצמם. דרגות החופש של הפריור נמדדות כך או כך.",
    },
}


def layer_rows(metadata: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
    md = artifacts.retention_metadata() if metadata is None else metadata
    rows: list[dict[str, Any]] = []
    for name, active_key, reason_key, labels in LAYER_SPECS:
        active = md.get(active_key) is True if active_key else None
        if active is None:
            # No boolean key: the layer states itself in its reason sentence.
            reason = str(md.get(reason_key) or "")
            active = reason.startswith("active by default") if reason else None
        rows.append({
            "id": f"layer.{name}",
            "family": name,
            "label_en": labels["en"],
            "label_he": labels["he"],
            "on": active,
            "decided_by_en": "an owner decision, not a gate",
            "decided_by_he": "החלטת בעלים, לא שער",
            "reason": str(md.get(reason_key) or "") if reason_key else "",
            "note_en": LAYER_NOTES[name]["en"],
            "note_he": LAYER_NOTES[name]["he"],
            "measured": _layer_measurements(name, md),
        })
    return rows


def _layer_measurements(name: str, md: dict[str, Any]) -> dict[str, Any]:
    if name == "placebo_correction":
        block = md.get("placebo_correction") if isinstance(md.get("placebo_correction"), dict) else {}
        return {key: block.get(key) for key in
                ("pooled_drift", "se", "n_pseudo", "n_clusters", "baseline", "method", "seed")}
    if name == "interval_calibration":
        return {"width_factor_measured": md.get("width_factor_measured"),
                "interval_method": md.get("interval_method"),
                "bootstrap_B": md.get("bootstrap_B"),
                "interval_seed": md.get("interval_seed")}
    return {"prior_df": md.get("prior_df")}


def ledger() -> dict[str, Any]:
    """Every gate, every layer, and the count in each state."""
    payload = artifacts.read_artifact(artifacts.AUDIENCE_ARTIFACT) or {}
    base = payload.get("base") if isinstance(payload.get("base"), dict) else {}
    observations = base.get("n_observations")
    metadata = artifacts.retention_metadata()
    rows = retention_rows(metadata) + audience_rows(
        payload.get("gates") if isinstance(payload.get("gates"), dict) else {},
        int(observations) if isinstance(observations, (int, float)) else None)
    counts = {state: sum(1 for row in rows if row["state"] == state)
              for state in (ACTIVE, TESTED_AND_LOST, NO_CONTRAST, NOT_MEASURED)}
    return {
        "gates": rows,
        "layers": layer_rows(metadata),
        "counts": counts,
        "total": len(rows),
        "states": [{"id": state, **STATE_LABELS[state], "meaning_en": STATE_MEANINGS[state]["en"],
                    "meaning_he": STATE_MEANINGS[state]["he"]}
                   for state in (ACTIVE, TESTED_AND_LOST, NO_CONTRAST, NOT_MEASURED)],
    }
