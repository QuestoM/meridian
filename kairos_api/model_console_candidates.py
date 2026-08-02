"""Candidate coefficient artifacts, and the money adopting one would move.

Five candidate artifacts sit in ``models/candidates/`` and nothing in the
product has ever read them. Each is a full coefficients file produced by the
same training script under a different flag, so the honest question a model
steward asks about one is in four parts, and this module answers all four
from the files themselves:

- **What did its gates decide differently.** A field-by-field comparison of the
  gate metadata against the shipped artifact.
- **What was each verdict decided on.** The held-out figures behind the gates,
  paired against the shipped ones by
  :mod:`kairos_api.model_console_candidates_holdout`, because a candidate can
  agree on every flag while the evidence under it has moved.
- **What did its coefficients do.** The per-cell deltas, summarised, plus the
  largest single move.
- **What money would adopting it move.** Not a direction and not an adjective:
  the weekly plan computed twice, once with the shipped artifact and once with
  the candidate, and the difference in shekels with the scope printed beside it.

The money question costs about a hundred seconds of optimizer per artifact, so
it is measured on demand and stored against a fingerprint of everything that
went into it (the candidate bytes, the shipped bytes, the settings bytes and
the engine version). A stored measurement whose fingerprint no longer matches
is reported as stale rather than served as current, and a candidate that has
never been measured says so. Nothing here estimates, interpolates or scales a
figure it did not compute.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional

from kairos_api import model_console_candidates_holdout as holdout

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_DIR = ROOT / "models" / "candidates"
SHIPPED_PATH = ROOT / "models" / "tv_break_coefficients.json"
SETTINGS_PATH = ROOT / "data" / "kairos_settings.json"

# The candidate file naming the training script uses. The id is what the file
# says it is, never a label chosen here.
_NAME_PATTERN = re.compile(r"^tv_break_coefficients_(?P<id>[a-z0-9_]+)\.json$")

# What each candidate flag changes, in the words the training script's own
# --help uses. A candidate whose id is not in this table still lists, with its
# purpose read from its metadata; the table only supplies the plain sentence
# where the artifact carries none.
CANDIDATE_SUBJECTS: dict[str, dict[str, str]] = {
    "afterwindow": {
        "en": "Widens the after-break measurement window, so a retention effect is not cut short by the next break.",
        "he": "מרחיב את חלון המדידה שאחרי הברייק, כדי שאפקט השימור לא ייקטע על ידי הברייק הבא.",
    },
    "calibrated": {
        "en": "Replaces the confidence bands with seeded bootstrap quantiles that carry the estimation error of tau squared.",
        "he": "מחליף את רצועות הביטחון בקוונטילים מבוטסטרפ עם זרע, שנושאים את שגיאת האמידה של טאו בריבוע.",
    },
    "competitor": {
        "en": "Adds the competing lineup as a covariate, measured as an unnamed aggregate.",
        "he": "מוסיף את הליינאפ המתחרה כמשתנה מסביר, נמדד כאגרגט ללא שמות.",
    },
    "placebo_corrected": {
        "en": "Subtracts each genre's measured no-break drift before pooling, using the content-only baseline.",
        "he": "מחסיר מכל ז'אנר את הסחיפה הנמדדת ללא ברייק לפני האיחוד, על בסיס תוכן בלבד.",
    },
    "spotclip": {
        "en": "Clips a break at the first spot that leaves it, so the measured window holds only real break time.",
        "he": "גוזם את הברייק בתשדיר הראשון שיוצא ממנו, כך שחלון המדידה מכיל רק זמן ברייק אמיתי.",
    },
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> "dict[str, Any] | None":
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("candidate artifact %s is unreadable (%s)", path, exc)
        return None
    return payload if isinstance(payload, dict) else None


def candidate_paths() -> "list[Path]":
    if not CANDIDATE_DIR.is_dir():
        return []
    return sorted(p for p in CANDIDATE_DIR.glob("*.json") if _NAME_PATTERN.match(p.name))


def candidate_id(path: Path) -> str:
    match = _NAME_PATTERN.match(path.name)
    return match.group("id") if match else path.stem


def candidate_path(identifier: str) -> "Path | None":
    for path in candidate_paths():
        if candidate_id(path) == identifier:
            return path
    return None


def engine_version() -> str:
    """The engine version the run log stamps on every run, or unknown."""
    try:
        from kairos.observability.run_log import KAIROS_ENGINE_VERSION

        return str(KAIROS_ENGINE_VERSION)
    except Exception:  # pragma: no cover - defensive, a fingerprint must not raise
        return "unknown"


def measurement_inputs(path: Path) -> dict[str, str]:
    """Everything a money measurement depends on, one digest per input.

    The candidate bytes, the shipped bytes it is compared against, the saved
    settings the plan is computed under and the engine version. Kept per input
    rather than fused into one number so that a stale measurement can say which
    input moved instead of only that something did. The whole settings file is
    digested rather than the three fields the totals read directly, because the
    engine receives the whole document and any field in it can reach the plan.
    """
    return {
        "candidate": _sha256(path) if path.is_file() else "absent",
        "shipped": _sha256(SHIPPED_PATH) if SHIPPED_PATH.is_file() else "absent",
        "settings": _sha256(SETTINGS_PATH) if SETTINGS_PATH.is_file() else "absent",
        "engine": engine_version(),
    }


def measurement_fingerprint(path: Path) -> str:
    """The same inputs as one digest, for an equality check in one comparison."""
    digest = hashlib.sha256()
    for key, value in sorted(measurement_inputs(path).items()):
        digest.update(f"{key}={value};".encode("utf-8"))
    return digest.hexdigest()


# What a person calls each input when the console says it moved.
INPUT_LABELS = {
    "candidate": {"en": "the candidate artifact", "he": "קובץ המועמד"},
    "shipped": {"en": "the shipped model", "he": "המודל המשודר"},
    "settings": {"en": "the saved settings", "he": "ההגדרות השמורות"},
    "engine": {"en": "the engine version", "he": "גרסת המנוע"},
}


def changed_inputs(path: Path, stored: dict[str, Any]) -> list[str]:
    """Which inputs moved since a stored measurement, named rather than counted."""
    before = stored.get("inputs")
    if not isinstance(before, dict):
        return []
    now = measurement_inputs(path)
    return sorted(key for key, value in now.items() if before.get(key) != value)


# ---------------------------------------------------------------------------
# What the artifact itself says: gate deltas and coefficient deltas
# ---------------------------------------------------------------------------

# The metadata keys that carry a gate decision. A difference on one of these is
# a different verdict, which is what a steward reads first.
GATE_KEYS = (
    "series_layer_active",
    "counterprogramming_active",
    "first_break_active",
    "first_break_multiplier",
    "first_break_p_value",
    "placebo_correction_active",
    "detrend_seasonality_recommended",
    "moderated_variances",
    "interval_method",
    "detrend_baseline_mode",
)


def _comparable(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 9)
    return value


def gate_deltas(shipped: dict[str, Any], candidate: dict[str, Any]) -> list[dict[str, Any]]:
    """Every gate key whose value differs, with both values and neither judged."""
    rows: list[dict[str, Any]] = []
    for key in GATE_KEYS:
        before = _comparable(shipped.get(key))
        after = _comparable(candidate.get(key))
        if before == after:
            continue
        rows.append({
            "key": key,
            "shipped": shipped.get(key),
            "candidate": candidate.get(key),
            "shipped_absent": key not in shipped,
            "candidate_absent": key not in candidate,
        })
    return rows


def coefficient_deltas(shipped: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """The per-cell coefficient movement, summarised, with the largest move named.

    Cells are the (programme type x position x length) keys both artifacts
    carry. A cell present on one side only is counted and named rather than
    silently dropped.
    """
    left = shipped.get("coefficients") if isinstance(shipped.get("coefficients"), dict) else {}
    right = candidate.get("coefficients") if isinstance(candidate.get("coefficients"), dict) else {}
    shared = sorted(set(left) & set(right))
    moves: list[tuple[str, float, float]] = []
    for cell in shared:
        try:
            before = float(left[cell])
            after = float(right[cell])
        except (TypeError, ValueError):
            continue
        moves.append((cell, before, after))
    if not moves:
        return {
            "cells_compared": 0,
            "cells_only_in_shipped": sorted(set(left) - set(right)),
            "cells_only_in_candidate": sorted(set(right) - set(left)),
            "max_abs_delta": None,
            "max_abs_delta_cell": None,
            "mean_delta": None,
            "cells_moved": 0,
        }
    deltas = [(cell, after - before) for cell, before, after in moves]
    largest = max(deltas, key=lambda item: abs(item[1]))
    return {
        "cells_compared": len(moves),
        "cells_only_in_shipped": sorted(set(left) - set(right)),
        "cells_only_in_candidate": sorted(set(right) - set(left)),
        "max_abs_delta": round(largest[1], 6),
        "max_abs_delta_cell": largest[0],
        "mean_delta": round(sum(delta for _, delta in deltas) / len(deltas), 6),
        "cells_moved": sum(1 for _, delta in deltas if abs(delta) > 1e-9),
    }


# ---------------------------------------------------------------------------
# What money adopting it would move: the plan, computed twice
# ---------------------------------------------------------------------------


def _totals(frame: Any, operator_channel: str) -> dict[str, Any]:
    """The plan's money at two printed scopes, and no third.

    The operator's own channel is named because the operator owns it. The other
    channels are reported only as an unnamed aggregate, which is the one shape
    a competitor may take in any payload.
    """
    owned = frame[frame["channel"] == operator_channel] if "channel" in frame.columns else frame
    others = frame[frame["channel"] != operator_channel] if "channel" in frame.columns else frame.iloc[0:0]
    return {
        "operator_channel": {
            "revenue": float(owned["predicted_revenue"].sum()),
            "retention_sum": float(owned["predicted_retention"].sum()),
            "breaks": int(owned["num_breaks"].sum()),
            "rows": int(len(owned)),
        },
        "whole_plan": {
            "revenue": float(frame["predicted_revenue"].sum()),
            "retention_sum": float(frame["predicted_retention"].sum()),
            "breaks": int(frame["num_breaks"].sum()),
            "rows": int(len(frame)),
            "channels": int(frame["channel"].nunique()) if "channel" in frame.columns else 1,
        },
        "other_channels_aggregate": {
            "revenue": float(others["predicted_revenue"].sum()),
            "rows": int(len(others)),
            "channels": int(others["channel"].nunique()) if "channel" in others.columns else 0,
        },
    }


def build_plan_totals(coefficients_path: Optional[Path] = None) -> dict[str, Any]:
    """Run the weekly plan once and total it, optionally on a candidate artifact.

    This is the same path the recompute takes: the saved settings, the saved
    risk weight and the saved operator channel. Nothing is written: no output
    CSV, no artifact, no version. Measured cost is about a hundred seconds.
    """
    from kairos.export.schedule import DEFAULT_IMPACT_MODEL_PATH, build_weekly_schedule
    from kairos.model.impact import load_impact_model
    from kairos.optimize.pricing import OptimizerAssumptions

    settings = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    impact_model = None
    if coefficients_path is not None:
        impact_model = load_impact_model(
            DEFAULT_IMPACT_MODEL_PATH,
            assumptions=OptimizerAssumptions(),
            coefficients_path=coefficients_path,
        )
    frame = build_weekly_schedule(
        settings=settings,
        revenue_weight=settings["revenue_weight"] / 100.0,
        risk_lambda=settings["risk_lambda"],
        operator_channel=settings["operator_channel"],
        today=date.today(),
        impact_model=impact_model,
    )
    return _totals(frame, str(settings.get("operator_channel") or ""))


def _delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    revenue_before = float(before["revenue"])
    revenue_after = float(after["revenue"])
    moved = revenue_after - revenue_before
    return {
        "revenue_shipped": round(revenue_before, 2),
        "revenue_candidate": round(revenue_after, 2),
        "revenue_delta": round(moved, 2),
        "revenue_delta_pct": round(100.0 * moved / revenue_before, 4) if revenue_before else None,
        "breaks_delta": int(after.get("breaks", 0)) - int(before.get("breaks", 0)),
        "retention_sum_delta": round(
            float(after.get("retention_sum", 0.0)) - float(before.get("retention_sum", 0.0)), 3),
    }


def measure_money_movement(identifier: str, baseline: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """The measured money movement for one candidate, ready to store.

    ``baseline`` is the shipped artifact's own totals; pass one in to compare a
    set of candidates against a single baseline run rather than recomputing it
    per candidate.
    """
    path = candidate_path(identifier)
    if path is None:
        raise FileNotFoundError(f"no candidate artifact called {identifier!r}")
    settings = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    started = time.monotonic()
    base = baseline if baseline is not None else build_plan_totals(None)
    got = build_plan_totals(path)
    return {
        "candidate_id": identifier,
        "fingerprint": measurement_fingerprint(path),
        "inputs": measurement_inputs(path),
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": round(time.monotonic() - started, 1),
        "engine_version": engine_version(),
        "operator_channel": str(settings.get("operator_channel") or ""),
        "scope": {
            "operator_channel": {
                "rows": base["operator_channel"]["rows"],
                "basis": "the weekly plan the run path computes, summed over the operator's own channel",
            },
            "whole_plan": {
                "rows": base["whole_plan"]["rows"],
                "channels": base["whole_plan"]["channels"],
                "basis": "the same plan summed over every channel the optimizer schedules",
            },
        },
        "shipped_totals": base,
        "candidate_totals": got,
        "operator_channel_delta": _delta(base["operator_channel"], got["operator_channel"]),
        "whole_plan_delta": _delta(base["whole_plan"], got["whole_plan"]),
    }


def summary_row(path: Path, shipped_metadata: dict[str, Any],
                measurement: Optional[dict[str, Any]]) -> dict[str, Any]:
    """One candidate as the console lists it: identity, deltas, money state."""
    identifier = candidate_id(path)
    payload = _read_json(path) or {}
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    subject = CANDIDATE_SUBJECTS.get(identifier)
    fingerprint = measurement_fingerprint(path)
    if measurement is None:
        money = {"state": "not_measured",
                 "reason_en": "The money this would move has not been measured yet.",
                 "reason_he": "הכסף שזה יזיז טרם נמדד."}
    elif str(measurement.get("fingerprint") or "") != fingerprint:
        # Name what moved. A measurement stored before the inputs were recorded
        # per key cannot name anything, and says that rather than blaming "an
        # input", which reads like a fact somebody checked.
        moved = changed_inputs(path, measurement)
        if moved:
            names_en = ", ".join(INPUT_LABELS[key]["en"] for key in moved)
            names_he = ", ".join(INPUT_LABELS[key]["he"] for key in moved)
            reason_en = f"Not current. What changed since it was measured: {names_en}."
            reason_he = f"אינו עדכני. מה שהשתנה מאז המדידה: {names_he}."
        else:
            reason_en = "Not current. The stored measurement does not record its inputs, so what moved cannot be named. Measuring again records them."
            reason_he = "אינו עדכני. המדידה השמורה אינה רושמת את הקלטים שלה, ולכן לא ניתן לציין מה השתנה. מדידה חוזרת תרשום אותם."
        money = {"state": "stale",
                 "changed": moved,
                 "reason_en": reason_en,
                 "reason_he": reason_he,
                 "measured_at": measurement.get("measured_at"),
                 "operator_channel_delta": measurement.get("operator_channel_delta"),
                 "whole_plan_delta": measurement.get("whole_plan_delta")}
    else:
        money = {"state": "measured", **measurement}
    gate_rows = gate_deltas(shipped_metadata, metadata)
    held_out = holdout.held_out_deltas(shipped_metadata, metadata, GATE_KEYS)
    cells = coefficient_deltas(_read_json(SHIPPED_PATH) or {}, payload)
    return {
        "id": identifier,
        "file": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "computed_at": metadata.get("computed_at"),
        "purpose": metadata.get("purpose"),
        "subject_en": (subject or {}).get("en"),
        "subject_he": (subject or {}).get("he"),
        "gate_deltas": gate_rows,
        "held_out_deltas": held_out,
        "coefficient_deltas": cells,
        "differences": holdout.differences(gate_rows, held_out, cells),
        "money": money,
        "fingerprint": fingerprint,
    }
