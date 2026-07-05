"""Shared helpers for the model-validation review scripts.

Validation-only code: nothing here is imported by product source. Every script
in scripts/validation/ prints an environment snapshot (git SHA + sha256 of the
model files under review) so results can be tied to an exact code state; a peer
is making behavior-preserving edits in kairos/model concurrently, and the
snapshot makes any resulting inconsistency detectable.

All randomness uses numpy default_rng with explicit seeds. Scripts write their
numeric results as JSON under scripts/validation/out/ so the findings document
can quote exact numbers.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent / "out"
ARTIFACT = ROOT / "models" / "tv_break_coefficients.json"

# The model files whose behavior this review measures.
MODEL_FILES = (
    "kairos/model/measure.py",
    "kairos/model/impact.py",
    "kairos/model/series_gate.py",
    "kairos/model/competitor_gate.py",
    "kairos/model/competitor_model.py",
    "kairos/model/prepare.py",
)

# z multipliers for the nominal central-interval levels under review. The
# shipped pipeline hard-codes 1.96 (95%); other levels rescale the same
# posterior sd, which is an exact inversion of the shipped interval, not a
# reimplementation.
Z_LEVELS = {0.50: 0.6744897501960817, 0.80: 1.2815515655446004, 0.95: 1.959963984540054}
Z95 = Z_LEVELS[0.95]

# Temporal holdout: November 2024 has 30 days of measurable breaks
# (2024-11-01 .. 2024-11-30). Train = first 23 days, test = last 7 days.
TRAIN_END = pd.Timestamp("2024-11-23 23:59:59")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def env_snapshot() -> dict[str, object]:
    """Git SHA + model-file hashes + interpreter, for the concurrent-edit check."""
    try:
        sha = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - snapshot must never break a study
        sha = "unknown"
    return {
        "git_head": sha,
        "python": sys.version.split()[0],
        "model_file_sha256": {name: sha256_file(ROOT / name) for name in MODEL_FILES},
    }


def print_snapshot(snapshot: dict[str, object]) -> None:
    print(f"[env] git HEAD {snapshot['git_head']}  python {snapshot['python']}")
    for name, digest in snapshot["model_file_sha256"].items():  # type: ignore[union-attr]
        print(f"[env] {name} sha256 {str(digest)[:16]}")


def load_frames():
    """Load the real reference frames once (spots, programmes, dayparts, classifier)."""
    from kairos.data.classifier import ProgramClassifier
    from kairos.data.loaders import load_dayparts, load_programmes, load_spots

    return (
        load_spots(),
        load_programmes(),
        load_dayparts(),
        ProgramClassifier.from_yaml(),
    )


def load_effects_full(frames=None) -> pd.DataFrame:
    """Measure every break on the full month with the SHIPPED pipeline."""
    from kairos.model.measure import break_effects

    spots, programmes, dayparts, classifier = frames or load_frames()
    effects = break_effects(spots, programmes, dayparts, classifier)
    effects = effects.reset_index(drop=True)
    effects["break_start"] = pd.to_datetime(effects["break_start"])
    return effects


def coeff_log_params(coeff) -> tuple[float, float]:
    """Invert a shipped MeasuredCoefficient into (theta, posterior sd) in log space.

    The pipeline writes ci_low = exp(theta - 1.96*sd) - 1 and
    ci_high = exp(theta + 1.96*sd) - 1 with theta = log(1 + raw_delta), so
    log1p inverts the transform exactly. No reimplementation of the pooling.
    """
    theta = float(np.log1p(coeff.raw_delta))
    half95 = 0.5 * (float(np.log1p(coeff.ci_high)) - float(np.log1p(coeff.ci_low)))
    return theta, half95 / Z95


def wilson_interval(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (guards n == 0)."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def normal_cdf(x: np.ndarray | float) -> np.ndarray | float:
    from math import erf, sqrt

    if np.isscalar(x):
        return 0.5 * (1.0 + erf(float(x) / sqrt(2.0)))
    xs = np.asarray(x, dtype=float)
    return np.array([0.5 * (1.0 + erf(float(v) / sqrt(2.0))) for v in xs])


def write_results(name: str, payload: dict[str, object]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    target = OUT_DIR / name
    target.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    print(f"[out] wrote {target}")
    return target
