"""Pipeline determinism and gate seed-sensitivity study.

Part A -- DETERMINISM. The coefficient computation is claimed deterministic
(pure detrending + closed-form pooling, no sampling). This script proves or
refutes that by running the full computation twice in SEPARATE PROCESSES
(fresh interpreter, fresh hash seed, fresh data load) and byte-comparing the
canonical JSON payloads (coefficients + detail + diagnostics + both gates).
It then compares the recompute against the SHIPPED artifact
``models/tv_break_coefficients.json`` field by field, and re-hashes the source
workbooks against the artifact's recorded fingerprints. Nothing under models/
or output/ is written; the recompute goes to scripts/validation/out/.

Part B -- SEED SENSITIVITY OF THE GATES. The series gate and the
counter-programming gate each hold out a "deterministic random 20%" of breaks
with a hard-coded seed (42). A gate whose verdict depends on which 20% the
seed picks is not a gate. We monkeypatch the module seed constant (runtime
only; no product source is edited) across 20 seeds and report: the verdict
flip rate, the seed-noise band of the RMSE-improvement statistic, and how the
2% activation margin compares with that noise. The first-break gate has no
split; it is run twice to confirm bit-identical output.

Part C -- FIX CANDIDATES, EVALUATED. (1) 5-fold cross-validation of the same
genre-vs-series comparison (every break predicted exactly once; no arbitrary
20%); (2) the seed-averaged gate (mean improvement across the 20 splits with
its SE). Both are computed so the recommendation in the findings document
carries measured numbers, not taste.

Deterministic: fixed seed list 0..19 plus the shipped 42; K-fold partition
seeded with 42. Runtime ~2-3 minutes (dominated by two subprocess data loads
and the competitor-feature build).

Run:  /Users/home/.venvs/meridian/bin/python scripts/validation/determinism_seeds.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _valcommon import (  # noqa: E402
    ARTIFACT,
    OUT_DIR,
    ROOT,
    env_snapshot,
    load_frames,
    print_snapshot,
    write_results,
)

SEEDS = list(range(20))
SHIPPED_SEED = 42
KFOLD_K = 5
KFOLD_SEED = 42


# --------------------------------------------------------------------------
# Part A: determinism
# --------------------------------------------------------------------------

def emit_canonical(path: Path) -> None:
    """Compute the full coefficient payload from scratch and write canonical JSON.

    Runs in a fresh subprocess so two invocations share no interpreter state.
    Mirrors scripts/compute_measured_coefficients.py's flow (same functions in
    the same order) but writes to the given path and omits the wall-clock
    ``computed_at`` stamp, which is the one intentionally volatile field.
    """
    from kairos.model.measure import (
        between_cell_variance,
        break_effects,
        channel_coefficients,
        first_break_gate,
    )
    from kairos.model.series_gate import series_holdout_gate

    spots, programmes, dayparts, classifier = load_frames()
    effects = break_effects(spots, programmes, dayparts, classifier)
    coefficients = channel_coefficients(effects)
    payload = {
        "coefficients": {name: c.coefficient for name, c in coefficients.items()},
        "detail": {name: asdict(c) for name, c in coefficients.items()},
        "diagnostics": between_cell_variance(effects),
        "first_break": first_break_gate(effects),
        "series_gate": series_holdout_gate(effects),
        "n_effects": int(len(effects)),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_determinism() -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = [OUT_DIR / "recompute_run_a.json", OUT_DIR / "recompute_run_b.json"]
    for p in paths:
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--emit", str(p)],
            check=True, cwd=str(ROOT),
        )
    digests = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    byte_identical = digests[0] == digests[1]

    run_a = json.loads(paths[0].read_text(encoding="utf-8"))
    shipped = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    # Field-by-field against the shipped artifact.
    max_coeff_diff = 0.0
    exact = 0
    for name, value in shipped["coefficients"].items():
        diff = abs(run_a["coefficients"].get(name, float("nan")) - value)
        max_coeff_diff = max(max_coeff_diff, diff)
        exact += int(run_a["coefficients"].get(name) == value)
    detail_fields_equal = all(
        run_a["detail"][name][field] == shipped["detail"][name][field]
        for name in shipped["detail"]
        for field in ("coefficient", "raw_delta", "n", "ci_low", "ci_high")
    ) if set(shipped["detail"]) == set(run_a["detail"]) else False

    # Freshness: re-hash the source files the artifact claims it was built from.
    fingerprints_ok = {}
    for rel, digest in shipped.get("metadata", {}).get("source_fingerprints", {}).items():
        f = ROOT / rel
        fingerprints_ok[rel] = (
            hashlib.sha256(f.read_bytes()).hexdigest() == digest if f.exists() else None
        )

    gate_match = (
        run_a["series_gate"]["series_gate_holdout"]["genre_rmse"]
        == shipped["metadata"]["series_gate_holdout"]["genre_rmse"]
        and run_a["series_gate"]["series_gate_holdout"]["series_rmse"]
        == shipped["metadata"]["series_gate_holdout"]["series_rmse"]
    )
    fb_match = (
        run_a["first_break"]["first_break_p_value"]
        == shipped["metadata"]["first_break_p_value"]
    )
    tau2_match = (
        run_a["diagnostics"]["tau2"] == shipped["metadata"]["between_cell_variance_tau2"]
    )

    return {
        "two_process_byte_identical": byte_identical,
        "run_sha256": digests,
        "recompute_vs_shipped": {
            "n_cells_shipped": len(shipped["coefficients"]),
            "n_cells_recomputed": len(run_a["coefficients"]),
            "coefficients_exactly_equal": exact,
            "max_abs_coefficient_diff": max_coeff_diff,
            "detail_fields_exactly_equal": bool(detail_fields_equal),
            "tau2_exactly_equal": bool(tau2_match),
            "series_gate_rmse_exactly_equal": bool(gate_match),
            "first_break_p_exactly_equal": bool(fb_match),
        },
        "source_fingerprints_fresh": fingerprints_ok,
    }


# --------------------------------------------------------------------------
# Part B: gate seed sensitivity
# --------------------------------------------------------------------------

def series_gate_over_seeds(effects: pd.DataFrame) -> dict:
    import kairos.model.series_gate as sg

    original = sg._HOLDOUT_SEED
    rows = []
    try:
        for seed in [SHIPPED_SEED, *SEEDS]:
            sg._HOLDOUT_SEED = seed
            gate = sg.series_holdout_gate(effects)
            h = gate["series_gate_holdout"]
            improvement = (
                100.0 * (h["genre_rmse"] - h["series_rmse"]) / h["genre_rmse"]
                if h["genre_rmse"]
                else float("nan")
            )
            rows.append(
                {
                    "seed": seed,
                    "active": bool(gate["series_layer_active"]),
                    "genre_rmse": h["genre_rmse"],
                    "series_rmse": h["series_rmse"],
                    "improvement_pct": improvement,
                }
            )
    finally:
        sg._HOLDOUT_SEED = original
    return _summarize_seed_rows(rows, threshold_pct=2.0)


def counterprogramming_gate_over_seeds(effects_cp: pd.DataFrame) -> dict:
    import kairos.model.competitor_gate as cg

    original = cg._HOLDOUT_SEED
    rows = []
    try:
        for seed in [SHIPPED_SEED, *SEEDS]:
            cg._HOLDOUT_SEED = seed
            gate = cg.counterprogramming_holdout_gate(effects_cp)
            h = gate["counterprogramming_holdout"]
            improvement = (
                100.0 * (h["rmse_without"] - h["rmse_with"]) / h["rmse_without"]
                if h["rmse_without"]
                else float("nan")
            )
            rows.append(
                {
                    "seed": seed,
                    "active": bool(gate["counterprogramming_active"]),
                    "rmse_without": h["rmse_without"],
                    "rmse_with": h["rmse_with"],
                    "improvement_pct": improvement,
                }
            )
    finally:
        cg._HOLDOUT_SEED = original
    return _summarize_seed_rows(rows, threshold_pct=2.0)


def _summarize_seed_rows(rows: list[dict], threshold_pct: float) -> dict:
    shipped_row = rows[0]
    others = rows[1:]
    imps = np.array([r["improvement_pct"] for r in others], dtype=float)
    verdicts = [r["active"] for r in others]
    flips_vs_shipped = sum(v != shipped_row["active"] for v in verdicts)
    sign_flips = int(np.sum(imps > 0)), int(np.sum(imps <= 0))
    return {
        "shipped_seed": shipped_row,
        "n_seeds": len(others),
        "verdict_flip_rate_vs_shipped": flips_vs_shipped / len(others),
        "verdicts_active": int(np.sum(verdicts)),
        "improvement_pct_mean": float(np.mean(imps)),
        "improvement_pct_sd": float(np.std(imps, ddof=1)),
        "improvement_pct_min": float(np.min(imps)),
        "improvement_pct_max": float(np.max(imps)),
        "improvement_pct_se_of_mean": float(np.std(imps, ddof=1) / np.sqrt(len(imps))),
        "positive_improvement_seeds": sign_flips[0],
        "nonpositive_improvement_seeds": sign_flips[1],
        "threshold_pct": threshold_pct,
        "threshold_minus_mean_in_sd_units": float(
            (threshold_pct - np.mean(imps)) / np.std(imps, ddof=1)
        ),
        "rows": rows,
    }


def first_break_determinism(effects: pd.DataFrame) -> dict:
    from kairos.model.measure import first_break_gate

    a = first_break_gate(effects)
    b = first_break_gate(effects)
    return {"identical_across_two_runs": a == b, "gate": {k: a[k] for k in (
        "first_break_active", "first_break_p_value", "first_break_multiplier")}}


# --------------------------------------------------------------------------
# Part C: fix candidates
# --------------------------------------------------------------------------

def series_kfold(effects: pd.DataFrame, k: int = KFOLD_K, seed: int = KFOLD_SEED) -> dict:
    """5-fold CV of the identical genre-vs-series comparison: every break is
    predicted exactly once, removing the arbitrary-20% dependence. Evaluation
    logic mirrors the gate's own predictors (train-cell means, series means
    with genre fallback); this is a candidate replacement DESIGN, computed
    here for evidence, not shipped code."""
    from kairos.data.title_features import canonicalize_series

    work = effects[["channel_name", "log_effect", "title"]].copy().reset_index(drop=True)
    work["series_key"] = work["title"].map(canonicalize_series)
    n = len(work)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, k)

    se_genre, se_series = [], []
    fold_improvements = []
    for fold in folds:
        test_mask = np.zeros(n, dtype=bool)
        test_mask[fold] = True
        train = work[~test_mask]
        test = work[test_mask]
        genre_means = train.groupby("channel_name")["log_effect"].mean().to_dict()
        series_means = {
            (str(c), str(s)): float(g["log_effect"].mean())
            for (c, s), g in train.groupby(["channel_name", "series_key"])
            if s
        }
        global_mean = float(train["log_effect"].mean())
        y = test["log_effect"].to_numpy()
        yg = np.array([
            genre_means.get(str(r.channel_name), global_mean)
            for r in test.itertuples(index=False)
        ])
        ys = np.array([
            series_means.get(
                (str(r.channel_name), str(r.series_key)),
                genre_means.get(str(r.channel_name), global_mean),
            )
            for r in test.itertuples(index=False)
        ])
        se_genre.append((y - yg) ** 2)
        se_series.append((y - ys) ** 2)
        rg = float(np.sqrt(np.mean((y - yg) ** 2)))
        rs = float(np.sqrt(np.mean((y - ys) ** 2)))
        fold_improvements.append(100.0 * (rg - rs) / rg)

    rmse_genre = float(np.sqrt(np.mean(np.concatenate(se_genre))))
    rmse_series = float(np.sqrt(np.mean(np.concatenate(se_series))))
    return {
        "k": k,
        "rmse_genre_pooled": rmse_genre,
        "rmse_series_pooled": rmse_series,
        "improvement_pct_pooled": 100.0 * (rmse_genre - rmse_series) / rmse_genre,
        "fold_improvements_pct": fold_improvements,
        "fold_improvement_sd": float(np.std(fold_improvements, ddof=1)),
    }


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "--emit":
        emit_canonical(Path(sys.argv[2]))
        return

    t0 = time.time()
    snapshot = env_snapshot()
    print_snapshot(snapshot)

    print("\n=== PART A: DETERMINISM (two fresh processes + shipped artifact) ===")
    det = run_determinism()
    print(f"two-process byte-identical: {det['two_process_byte_identical']}")
    rvs = det["recompute_vs_shipped"]
    print(
        f"recompute vs shipped artifact: {rvs['coefficients_exactly_equal']}/"
        f"{rvs['n_cells_shipped']} coefficients exactly equal "
        f"(max abs diff {rvs['max_abs_coefficient_diff']:.3e}); detail fields equal: "
        f"{rvs['detail_fields_exactly_equal']}; tau2 equal: {rvs['tau2_exactly_equal']}; "
        f"series-gate RMSEs equal: {rvs['series_gate_rmse_exactly_equal']}; "
        f"first-break p equal: {rvs['first_break_p_exactly_equal']}"
    )
    print(f"source fingerprints fresh: {det['source_fingerprints_fresh']}")

    frames = load_frames()
    from kairos.model.measure import break_effects

    effects = break_effects(*frames)

    print("\n=== PART B: GATE SEED SENSITIVITY (20 seeds vs shipped seed 42) ===")
    series = series_gate_over_seeds(effects)
    print(
        f"series gate: shipped verdict active={series['shipped_seed']['active']} "
        f"(improvement {series['shipped_seed']['improvement_pct']:+.2f}%)"
    )
    print(
        f"  20-seed improvement: mean {series['improvement_pct_mean']:+.2f}%, "
        f"sd {series['improvement_pct_sd']:.2f}pp, range "
        f"[{series['improvement_pct_min']:+.2f}, {series['improvement_pct_max']:+.2f}]"
    )
    print(
        f"  verdict flip rate vs shipped: {series['verdict_flip_rate_vs_shipped']:.2f} "
        f"({series['verdicts_active']}/{series['n_seeds']} seeds would activate); "
        f"threshold sits {series['threshold_minus_mean_in_sd_units']:+.1f} sd above the mean"
    )

    t_feat = time.time()
    from kairos.model.competitor_model import measure_effects_with_competitors

    spots, programmes, dayparts, classifier = frames
    effects_cp = measure_effects_with_competitors(
        spots=spots, programmes=programmes, dayparts=dayparts, classifier=classifier
    )
    print(f"  [competitor features built in {time.time() - t_feat:.1f}s]")
    cp = counterprogramming_gate_over_seeds(effects_cp)
    print(
        f"counter-programming gate: shipped verdict active={cp['shipped_seed']['active']} "
        f"(improvement {cp['shipped_seed']['improvement_pct']:+.2f}%)"
    )
    print(
        f"  20-seed improvement: mean {cp['improvement_pct_mean']:+.2f}%, "
        f"sd {cp['improvement_pct_sd']:.2f}pp, range "
        f"[{cp['improvement_pct_min']:+.2f}, {cp['improvement_pct_max']:+.2f}]"
    )
    print(
        f"  verdict flip rate vs shipped: {cp['verdict_flip_rate_vs_shipped']:.2f} "
        f"({cp['verdicts_active']}/{cp['n_seeds']} seeds would activate)"
    )

    fb = first_break_determinism(effects)
    print(f"first-break gate identical across two runs: {fb['identical_across_two_runs']}")

    print("\n=== PART C: FIX CANDIDATES ===")
    kf = series_kfold(effects)
    print(
        f"series 5-fold CV: pooled improvement {kf['improvement_pct_pooled']:+.2f}% "
        f"(per-fold sd {kf['fold_improvement_sd']:.2f}pp; folds "
        f"{[round(v, 2) for v in kf['fold_improvements_pct']]})"
    )
    print(
        f"seed-averaged gate: series mean {series['improvement_pct_mean']:+.2f}% "
        f"+/- {series['improvement_pct_se_of_mean']:.2f} (SE over 20 seeds); "
        f"cp mean {cp['improvement_pct_mean']:+.2f}% +/- {cp['improvement_pct_se_of_mean']:.2f}"
    )

    write_results(
        "determinism_seeds.json",
        {
            "env": snapshot,
            "determinism": det,
            "series_gate_seeds": series,
            "counterprogramming_gate_seeds": cp,
            "first_break": fb,
            "series_kfold": kf,
            "runtime_seconds": round(time.time() - t0, 1),
        },
    )
    print(f"\n[done] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
