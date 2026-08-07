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
may be read, and measured rather than asserted.** There is no second month
anywhere in the repository, so no unseen data exists and the absolute figure for
every artifact is optimistic. What usually survives that is the paired
comparison, on the argument that the optimism is common-mode. That argument
holds only when every row was fitted on the breaks it is scored on, and until
round 6 this module asserted it of every tree without checking.

Measured on this tree it does not hold. Five artifacts record a fit over 2,532
breaks and ``spotclip`` records 2,336, having dropped 196 of them, and it is
scored on all 2,532 like everything else. So the optimism is not common-mode,
and the row it is uneven on is the row this table ranks first. Which of three
limit sentences a payload carries is now decided by ``adopt_candidate_basis.py``
from what the artifacts themselves record, and the condition that would lift the
limit is carried beside it.

**The two baselines are genuinely out of sample and they are the point.** They
live in ``adopt_candidate_baselines.py`` with the argument for them written out,
and they answer the question the artifacts cannot answer about themselves: does
the 36-cell structure earn its place out of sample, or is a single constant as
good.

**Every candidate row also carries its coefficient delta**, computed in
``adopt_candidate_cells.py``, because this is the only place that holds both the
coefficients and the per-break errors and so the only place a moved cell can be
attributed to what it bought rather than merely listed.

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

from scripts.adopt_candidate_baselines import (  # noqa: E402
    cell_structure,
    squared_errors as baseline_errors,
)
from scripts import adopt_candidate_basis as basis  # noqa: E402
from scripts import adopt_candidate_cells as cells_module  # noqa: E402
from scripts import adopt_candidate_words as words  # noqa: E402

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

# Both live in adopt_candidate_words.py, which is where every authored string
# this piece emits keeps its two halves. Re-exported under their old names
# because they are read from three modules and from the tests.
IN_SAMPLE_LIMIT = words.IN_SAMPLE_LIMIT
VERDICTS = words.VERDICTS


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
                **words.pair(words.RULE, "identical", "rule")}
    clears_paired = abs(float(paired["paired_statistic"])) >= PAIRED_T_BAR
    moved = float(paired["rmse_delta"])
    dispersion = float(paired["fold_dispersion"])
    clears_dispersion = abs(moved) > dispersion > 0
    rule = words.pair(words.RULE, "two_bars", "rule", bar=f"{PAIRED_T_BAR:.1f}",
                      statistic=paired["paired_statistic"], moved=f"{moved:.6f}",
                      dispersion=f"{dispersion:.6f}")
    if clears_paired and clears_dispersion:
        state = "better" if moved < 0 else "worse"
    else:
        state = "not_distinguishable"
    return {"state": state, **VERDICTS[state], **rule,
            "clears_paired_bar": clears_paired, "clears_fold_dispersion": clears_dispersion}


def _row(name: str, kind: str, errors: np.ndarray) -> dict[str, Any]:
    return {
        "id": name,
        "kind": kind,
        "rmse": round(_rmse(errors), 9),
        "mae": round(float(np.abs(np.sqrt(errors)).mean()), 9) if errors.size else 0.0,
        "breaks": int(errors.size),
    }


def rescore(paths: Optional[Paths] = None,
            effects: Optional[pd.DataFrame] = None) -> dict[str, Any]:
    """Score the shipped artifact, every candidate and both baselines at once."""
    paths = paths or Paths()
    frame = effects if effects is not None else measured_effects()
    y = frame["log_effect"].to_numpy(dtype=float)
    # ``channel_name`` is the column name the measured frame carries, and its
    # value is the composite cell key the artifacts are keyed by, of the form
    # News_first_long. It is not a channel and no channel name reaches any
    # payload from here. Measured on this tree: 36 distinct values, every one a
    # programme class, break position and length, and cells_not_carried is 0 on
    # all six artifacts. The column is the frozen measure module's and renaming
    # it is not this piece's to do, so the name is explained rather than moved.
    cells = frame["channel_name"].to_numpy()
    folds = _fold_slices(len(y))

    baselines = [_row(name, "baseline", errors)
                 for name, errors in baseline_errors(y, cells).items()]
    for row in baselines:
        row["out_of_sample"] = True
        row.update(words.pair(words.BASIS, row["id"], "basis"))

    shipped_payload = read_artifact(paths.shipped)
    shipped_coefficients = shipped_payload.get("coefficients") or {}
    fallback = float(np.mean(list(shipped_coefficients.values()))) if shipped_coefficients else 0.0
    shipped_predictions, shipped_missing = _predictions(shipped_coefficients, cells, fallback)
    shipped_errors = (y - shipped_predictions) ** 2
    shipped_metadata = shipped_payload.get("metadata") or {}
    shipped_row = _row("shipped", "shipped", shipped_errors)
    shipped_row.update({"out_of_sample": False, "cells": len(shipped_coefficients),
                        "cells_not_carried": shipped_missing,
                        "sha256": sha256_file(paths.shipped),
                        "file": paths.shipped.relative_to(paths.root).as_posix(),
                        # The live artifact is a row in this comparison like any
                        # other, so its fit basis is stated like any other. It
                        # was the one row that carried no fit basis at all,
                        # which is how the whole evaluation could assert that
                        # every row was fitted on the same breaks.
                        "breaks_fitted_on": shipped_metadata.get("total_breaks_measured"),
                        "self_reported": basis.self_reported("shipped", shipped_metadata)})
    scored_on = int(len(y))
    basis_rows = [basis.basis_row("shipped", shipped_metadata, scored_on)]

    rows: list[dict[str, Any]] = []
    signatures: dict[str, list[str]] = {}
    for path in candidate_files(paths):
        identifier = candidate_id(path)
        payload = read_artifact(path)
        metadata = payload.get("metadata") or {}
        basis_rows.append(basis.basis_row(identifier, metadata, scored_on))
        coefficients = payload.get("coefficients") or {}
        candidate_fallback = float(np.mean(list(coefficients.values()))) if coefficients else fallback
        predictions, missing = _predictions(coefficients, cells, candidate_fallback)
        errors = (y - predictions) ** 2
        row = _row(identifier, "candidate", errors)
        identical = bool(np.array_equal(predictions, shipped_predictions))
        paired = _paired(errors, shipped_errors, folds)
        row.update({
            "out_of_sample": False,
            "cells": len(coefficients),
            "cells_not_carried": missing,
            "sha256": sha256_file(path),
            "file": path.relative_to(paths.root).as_posix(),
            "breaks_fitted_on": metadata.get("total_breaks_measured"),
            # What the artifact's own producer recorded about adopting it. Not
            # ranked and not compared: it is the artifact's own split under its
            # own fit, and it is carried because a recommendation is the one
            # thing about an artifact that only its producer knows.
            "self_reported": basis.self_reported(identifier, metadata),
            "paired": paired,
            "verdict": verdict(paired, identical),
            # JS-19's done condition names the coefficient deltas beside the
            # gate deltas and the money. Computed here because this is the only
            # place that holds both the coefficients and the per-break errors,
            # so every cell can be attributed rather than merely listed.
            "cell_deltas": cells_module.cell_deltas(
                shipped_coefficients, coefficients, cells, shipped_errors, errors),
        })
        signature = hashlib.sha256(predictions.tobytes()).hexdigest()
        signatures.setdefault(signature, []).append(identifier)
        row["prediction_signature"] = signature[:12]
        rows.append(row)

    fit_basis = basis.fit_basis(basis_rows, scored_on)
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
            "metric_en": words.METRIC["en"],
            "metric_he": words.METRIC["he"],
            # One figure, not one per row: it is a property of the target and it
            # is identical for every artifact scored against it. An rmse read
            # without it cannot be judged at all.
            "target_sd": round(float(y.std()), 9),
            "target_sd_en": words.TARGET_SD["en"],
            "target_sd_he": words.TARGET_SD["he"],
            "folds": FOLDS,
        },
        # Measured, not asserted. Which of the three limit sentences this
        # evaluation carries is decided by what the artifacts record they were
        # fitted on, against what they are scored on here.
        "limit": basis.limit_for(fit_basis),
        "fit_basis": fit_basis,
        "fingerprint": rescore_fingerprint(paths),
        "inputs": rescore_inputs(paths),
        "baselines": baselines,
        "shipped": shipped_row,
        "candidates": rows,
        "duplicate_groups": duplicates,
        "cell_structure": cell_structure(baselines),
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


def candidate_row(identifier: str, paths: Optional[Paths] = None) -> Optional[dict[str, Any]]:
    """One candidate's stored re-score row, or nothing when there is no such name.

    Nothing rather than an empty row, because a name that is not a candidate and
    a candidate that has not been scored are two different states and a caller
    that cannot tell them apart will report the wrong one.
    """
    paths = paths or Paths()
    if identifier not in {candidate_id(path) for path in candidate_files(paths)}:
        return None
    for row in (load_rescore(paths) or {}).get("candidates") or []:
        if row.get("id") == identifier:
            return row
    return {"id": identifier}


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
