"""Score every coefficients artifact against one common set of measured breaks.

A model steward's question is "is this candidate genuinely better than what is
live". Nothing in the product answered it. Each artifact carries the held-out
figures of its own fit, taken under its own split, against its own target, so
comparing two of them by reading their metadata compares two different
experiments and not two predictors.

This module compares the predictors. It rebuilds the measured per-break effects
from the sources on disk, then scores the shipped artifact, every candidate and
two honest baselines on exactly the same breaks with exactly the same metric, so
a difference between two rows is a difference between two models.

**The limit of this evaluation, stated first because it decides how the numbers
may be read.** Every artifact in this tree was fitted on all 2,532 breaks in
``data/reference``, and there is no second month anywhere in the repository. So
no unseen data exists and the absolute figure for every artifact is optimistic.
What survives that is the paired comparison: the artifacts have the same 36
cells fitted on the same breaks, so the optimism is common-mode and the
difference between two of them is still readable. Every payload this module
emits carries that limit in words, plus the condition that would lift it.

**The two baselines are genuinely out of sample and they are the point.** The
leave-one-out cell mean predicts each break from the other breaks in its own
cell, and the leave-one-out global mean predicts it from every other break.
Neither has ever seen the break it predicts. They answer the question the
artifacts cannot answer about themselves: does the 36-cell structure earn its
place out of sample, or is a single constant as good.

**A verdict needs two things, not one.** A paired test over 2,532 breaks can
call a difference significant that is smaller than the dispersion between one
temporal fold and the next. So a candidate is reported as distinguishable only
when the paired statistic clears the bar AND the movement in RMSE exceeds the
fold-to-fold dispersion of that same movement. Anything else is reported as not
distinguishable, which is a finding and not a failure to decide.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Contiguous blocks in break_start order, the same fold construction and the
# same count the series gate uses (kairos/model/series_gate.py, GATE_FOLDS), so
# a dispersion figure here is comparable with the one the gates report.
FOLDS = 5

# The paired statistic bar. Two standard errors on the mean per-break squared
# error difference, which is the usual two-sided bar at roughly five percent.
PAIRED_T_BAR = 2.0

RESCORE_FILE = "holdout_rescores.json"

# The three files every artifact in this tree was measured from. Digested so a
# stored re-score can say the data moved rather than being served as current.
SOURCE_FILES = ("Spots.xlsx", "Programmes.xlsx", "Dayparts.xlsx")

IN_SAMPLE_LIMIT = {
    "state": "in_sample",
    "en": "Every artifact scored here was fitted on all of these breaks, so each absolute figure is optimistic. Only the difference between two rows is readable, because both carry the same optimism.",
    "he": "כל קובץ שנמדד כאן אומן על כל הברייקים האלה, ולכן כל מספר מוחלט הוא אופטימי. רק ההפרש בין שתי שורות ניתן לקריאה, כי שתיהן נושאות את אותה אופטימיות.",
    "unblocked_by_en": "A second month of measured breaks that no artifact here was fitted on.",
    "unblocked_by_he": "חודש נוסף של ברייקים נמדדים שאף קובץ כאן לא אומן עליו.",
}

VERDICTS = {
    "identical": {
        "en": "Predicts exactly what the shipped model predicts, break for break.",
        "he": "חוזה בדיוק את מה שהמודל המשודר חוזה, ברייק אחר ברייק.",
    },
    "better": {
        "en": "Closer to the measured effects than the shipped model, by more than the fold dispersion.",
        "he": "קרוב יותר לאפקטים הנמדדים מהמודל המשודר, ביותר מפיזור המקטעים.",
    },
    "worse": {
        "en": "Further from the measured effects than the shipped model, by more than the fold dispersion.",
        "he": "רחוק יותר מהאפקטים הנמדדים מהמודל המשודר, ביותר מפיזור המקטעים.",
    },
    "not_distinguishable": {
        "en": "Not distinguishable from the shipped model on this evaluation. The movement is inside the noise this data carries.",
        "he": "אינו ניתן להבחנה מהמודל המשודר במדידה הזו. התנועה נמצאת בתוך הרעש שהנתונים האלה נושאים.",
    },
}


@dataclass(frozen=True)
class Paths:
    """Where the artifacts and the store live, so a test can point elsewhere."""

    root: Path = ROOT

    @property
    def shipped(self) -> Path:
        return self.root / "models" / "tv_break_coefficients.json"

    @property
    def candidates_dir(self) -> Path:
        return self.root / "models" / "candidates"

    @property
    def releases_dir(self) -> Path:
        return self.root / "models" / "releases"

    @property
    def reference_dir(self) -> Path:
        return self.root / "data" / "reference"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "absent"


def read_artifact(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def candidate_files(paths: Paths) -> list[Path]:
    """Every candidate artifact, named the way the training script names one."""
    if not paths.candidates_dir.is_dir():
        return []
    return sorted(p for p in paths.candidates_dir.glob("tv_break_coefficients_*.json"))


def candidate_id(path: Path) -> str:
    return path.stem.replace("tv_break_coefficients_", "", 1)


def data_fingerprint(paths: Paths) -> dict[str, str]:
    """One digest per measured source, so a stale re-score can name what moved."""
    return {name: sha256_file(paths.reference_dir / name) for name in SOURCE_FILES}


def measured_effects() -> pd.DataFrame:
    """The evaluation set: every break, its cell and its measured log effect.

    Built by the same call the training script makes, so the breaks scored here
    are the breaks the coefficients were fitted on and the cell keys are the
    artifacts' own keys. Sorted by start time because the folds are temporal.
    """
    from kairos.data.classifier import ProgramClassifier
    from kairos.data.loaders import load_dayparts, load_programmes, load_spots
    from kairos.model.measure import break_effects

    frame = break_effects(load_spots(), load_programmes(), load_dayparts(),
                          ProgramClassifier.from_yaml())
    return frame.sort_values("break_start").reset_index(drop=True)


def _predictions(coefficients: dict[str, Any], cells: np.ndarray,
                 fallback: float) -> tuple[np.ndarray, int]:
    """One prediction per break, and how many cells the artifact does not carry.

    A cell the artifact has never heard of is predicted at the artifact's own
    mean over the cells it does carry, and counted, because silently predicting
    zero would flatter an artifact with holes in it.
    """
    values = np.array([coefficients.get(cell, np.nan) for cell in cells], dtype=float)
    missing = int(np.isnan(values).sum())
    return np.nan_to_num(values, nan=fallback), missing


def _leave_one_out(y: np.ndarray, groups: Optional[np.ndarray]) -> np.ndarray:
    """Predict each break from the others, globally or inside its own cell."""
    total, count = y.sum(), len(y)
    global_loo = (total - y) / (count - 1) if count > 1 else np.zeros_like(y)
    if groups is None:
        return global_loo
    frame = pd.DataFrame({"g": groups, "y": y})
    sums = frame.groupby("g")["y"].transform("sum").to_numpy(dtype=float)
    sizes = frame.groupby("g")["y"].transform("size").to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        inside = (sums - y) / (sizes - 1.0)
    return np.where(sizes > 1.0, inside, global_loo)


def _rmse(errors: np.ndarray) -> float:
    return float(np.sqrt(errors.mean())) if errors.size else 0.0


def _fold_slices(count: int) -> list[np.ndarray]:
    return [block for block in np.array_split(np.arange(count), FOLDS) if block.size]


def _paired(candidate: np.ndarray, shipped: np.ndarray,
            folds: list[np.ndarray]) -> dict[str, Any]:
    """The candidate against the shipped model on the same breaks, paired.

    The unit is the per-break squared error, so every break contributes one
    difference and the pairing removes the break-to-break variance that swamps
    an unpaired comparison of two RMSE figures.
    """
    difference = candidate - shipped
    count = int(difference.size)
    mean = float(difference.mean()) if count else 0.0
    sem = float(difference.std(ddof=1) / np.sqrt(count)) if count > 1 else 0.0
    statistic = float(mean / sem) if sem else 0.0
    fold_moves = [_rmse(candidate[block]) - _rmse(shipped[block]) for block in folds]
    dispersion = float(np.std(fold_moves, ddof=1)) if len(fold_moves) > 1 else 0.0
    moved = _rmse(candidate) - _rmse(shipped)
    return {
        "rmse_delta": round(moved, 9),
        "mean_squared_error_delta": mean,
        "standard_error": sem,
        "paired_statistic": round(statistic, 4),
        "paired_bar": PAIRED_T_BAR,
        "fold_rmse_deltas": [round(value, 9) for value in fold_moves],
        "fold_dispersion": round(dispersion, 9),
        "folds": len(fold_moves),
        "breaks_improved": int((candidate < shipped).sum()),
        "breaks_worsened": int((candidate > shipped).sum()),
        "breaks": count,
    }


def verdict(paired: dict[str, Any], identical: bool) -> dict[str, Any]:
    """Better, worse, or not distinguishable, with the rule that decided it.

    Both bars must be cleared. A paired statistic on 2,532 breaks can call a
    movement significant that is smaller than the difference between one
    temporal fold and the next, and a movement that small is not a property of
    the model, it is a property of which month you looked at.
    """
    if identical:
        return {"state": "identical", **VERDICTS["identical"],
                "rule_en": "Both artifacts predict the same value for every break."}
    clears_paired = abs(float(paired["paired_statistic"])) >= PAIRED_T_BAR
    moved = float(paired["rmse_delta"])
    dispersion = float(paired["fold_dispersion"])
    clears_dispersion = abs(moved) > dispersion > 0
    rule_en = (
        "Distinguishable only when the paired statistic reaches 2.0 and the movement in RMSE exceeds the fold dispersion. "
        f"Measured: statistic {paired['paired_statistic']}, movement {moved:.6f}, dispersion {dispersion:.6f}."
    )
    if clears_paired and clears_dispersion:
        state = "better" if moved < 0 else "worse"
    else:
        state = "not_distinguishable"
    return {"state": state, **VERDICTS[state], "rule_en": rule_en,
            "clears_paired_bar": clears_paired, "clears_fold_dispersion": clears_dispersion}


def _row(name: str, kind: str, errors: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    return {
        "id": name,
        "kind": kind,
        "rmse": round(_rmse(errors), 9),
        "mae": round(float(np.abs(np.sqrt(errors)).mean()), 9) if errors.size else 0.0,
        "breaks": int(errors.size),
        "target_sd": round(float(y.std()), 9),
    }


def rescore(paths: Optional[Paths] = None,
            effects: Optional[pd.DataFrame] = None) -> dict[str, Any]:
    """Score the shipped artifact, every candidate and both baselines at once."""
    paths = paths or Paths()
    frame = effects if effects is not None else measured_effects()
    y = frame["log_effect"].to_numpy(dtype=float)
    cells = frame["channel_name"].to_numpy()
    folds = _fold_slices(len(y))

    baselines = [
        _row("global_mean_loo", "baseline", (y - _leave_one_out(y, None)) ** 2, y),
        _row("cell_mean_loo", "baseline", (y - _leave_one_out(y, cells)) ** 2, y),
    ]
    for row in baselines:
        row["out_of_sample"] = True
        row["basis_en"] = ("Predicts each break from the other breaks, never from itself."
                           if row["id"] == "global_mean_loo" else
                           "Predicts each break from the other breaks in its own cell, never from itself.")

    shipped_payload = read_artifact(paths.shipped)
    shipped_coefficients = shipped_payload.get("coefficients") or {}
    fallback = float(np.mean(list(shipped_coefficients.values()))) if shipped_coefficients else 0.0
    shipped_predictions, shipped_missing = _predictions(shipped_coefficients, cells, fallback)
    shipped_errors = (y - shipped_predictions) ** 2
    shipped_row = _row("shipped", "shipped", shipped_errors, y)
    shipped_row.update({"out_of_sample": False, "cells": len(shipped_coefficients),
                        "cells_not_carried": shipped_missing,
                        "sha256": sha256_file(paths.shipped),
                        "file": paths.shipped.relative_to(paths.root).as_posix()})

    rows: list[dict[str, Any]] = []
    signatures: dict[str, list[str]] = {}
    for path in candidate_files(paths):
        identifier = candidate_id(path)
        payload = read_artifact(path)
        coefficients = payload.get("coefficients") or {}
        candidate_fallback = float(np.mean(list(coefficients.values()))) if coefficients else fallback
        predictions, missing = _predictions(coefficients, cells, candidate_fallback)
        errors = (y - predictions) ** 2
        row = _row(identifier, "candidate", errors, y)
        identical = bool(np.array_equal(predictions, shipped_predictions))
        paired = _paired(errors, shipped_errors, folds)
        row.update({
            "out_of_sample": False,
            "cells": len(coefficients),
            "cells_not_carried": missing,
            "sha256": sha256_file(path),
            "file": path.relative_to(paths.root).as_posix(),
            "breaks_fitted_on": (payload.get("metadata") or {}).get("total_breaks_measured"),
            "paired": paired,
            "verdict": verdict(paired, identical),
        })
        signature = hashlib.sha256(predictions.tobytes()).hexdigest()
        signatures.setdefault(signature, []).append(identifier)
        row["prediction_signature"] = signature[:12]
        rows.append(row)

    duplicates = [sorted(group) for group in signatures.values() if len(group) > 1]
    for row in rows:
        row["duplicate_of"] = sorted(
            name for group in duplicates if row["id"] in group for name in group if name != row["id"])

    return {
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "evaluation": {
            "breaks": int(len(y)),
            "cells": int(pd.unique(cells).size),
            "window": f"{frame['break_start'].min():%Y-%m-%d} to {frame['break_start'].max():%Y-%m-%d}",
            "target_en": "The detrended log effect measured on each break, rebuilt from the sources on disk.",
            "target_he": "אפקט הלוג המנוכה מגמה שנמדד בכל ברייק, נבנה מחדש מהמקורות שעל הדיסק.",
            "metric_en": "Root mean squared error against that measured effect, over the same breaks for every row.",
            "folds": FOLDS,
        },
        "limit": dict(IN_SAMPLE_LIMIT),
        "fingerprint": rescore_fingerprint(paths),
        "inputs": rescore_inputs(paths),
        "baselines": baselines,
        "shipped": shipped_row,
        "candidates": rows,
        "duplicate_groups": duplicates,
        "cell_structure": _cell_structure(baselines),
    }


def _cell_structure(baselines: list[dict[str, Any]]) -> dict[str, Any]:
    """Does the 36-cell split beat one constant, out of sample and honestly.

    This is the only figure on the whole surface that is free of the in-sample
    limit, because both baselines predict each break from breaks that are not
    it. It is reported whichever way it lands.
    """
    by_id = {row["id"]: row for row in baselines}
    cell = float(by_id["cell_mean_loo"]["rmse"])
    glob = float(by_id["global_mean_loo"]["rmse"])
    moved = cell - glob
    return {
        "cell_mean_loo_rmse": round(cell, 9),
        "global_mean_loo_rmse": round(glob, 9),
        "rmse_delta": round(moved, 9),
        "earns_its_place": bool(moved < 0),
        "out_of_sample": True,
        "reading_en": (
            "Out of sample the per-cell split predicts better than a single constant."
            if moved < 0 else
            "Out of sample the per-cell split does not predict better than a single constant."),
        "reading_he": (
            "מחוץ למדגם החלוקה לתאים חוזה טוב יותר מקבוע יחיד."
            if moved < 0 else
            "מחוץ למדגם החלוקה לתאים אינה חוזה טוב יותר מקבוע יחיד."),
    }


def rescore_inputs(paths: Paths) -> dict[str, str]:
    """Everything a re-score depends on, one digest per input."""
    inputs = {"shipped": sha256_file(paths.shipped)}
    for path in candidate_files(paths):
        inputs[f"candidate:{candidate_id(path)}"] = sha256_file(path)
    for name, digest in data_fingerprint(paths).items():
        inputs[f"data:{name}"] = digest
    return inputs


def rescore_fingerprint(paths: Paths) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(rescore_inputs(paths).items()):
        digest.update(f"{key}={value};".encode("utf-8"))
    return digest.hexdigest()


def rescore_path(paths: Paths) -> Path:
    return paths.releases_dir / RESCORE_FILE


def save_rescore(payload: dict[str, Any], paths: Optional[Paths] = None) -> Path:
    paths = paths or Paths()
    path = rescore_path(paths)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def load_rescore(paths: Optional[Paths] = None) -> Optional[dict[str, Any]]:
    paths = paths or Paths()
    path = rescore_path(paths)
    if not path.is_file():
        return None
    payload = read_artifact(path)
    return payload or None


def rescore_state(paths: Optional[Paths] = None,
                  stored: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Current, stale or never measured, and when stale, which input moved."""
    paths = paths or Paths()
    stored = stored if stored is not None else load_rescore(paths)
    if stored is None:
        return {"state": "not_measured",
                "reason_en": "The re-score has not been run on this tree yet.",
                "reason_he": "המדידה החוזרת טרם הורצה על העץ הזה."}
    if str(stored.get("fingerprint") or "") == rescore_fingerprint(paths):
        return {"state": "current", "measured_at": stored.get("measured_at")}
    before = stored.get("inputs") if isinstance(stored.get("inputs"), dict) else {}
    now = rescore_inputs(paths)
    moved = sorted(key for key in set(before) | set(now) if before.get(key) != now.get(key))
    return {"state": "stale", "measured_at": stored.get("measured_at"), "changed": moved,
            "reason_en": "Not current. What changed since it was measured: " + ", ".join(moved) + ".",
            "reason_he": "אינו עדכני. מה שהשתנה מאז המדידה: " + ", ".join(moved) + "."}
