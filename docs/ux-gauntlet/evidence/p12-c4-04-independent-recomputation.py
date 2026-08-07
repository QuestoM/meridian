"""Independent re-computation of every figure P12 puts on screen.

Written by the critic, not by the builder. It shares nothing with
scripts/adopt_candidate_* except the loaders that build the measured effects,
which are the frozen data path both would have to use.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/home/Code/questo/meridian")
sys.path.insert(0, str(ROOT))

from kairos.data.classifier import ProgramClassifier
from kairos.data.loaders import load_dayparts, load_programmes, load_spots
from kairos.model.measure import break_effects

frame = break_effects(load_spots(), load_programmes(), load_dayparts(),
                      ProgramClassifier.from_yaml())
frame = frame.sort_values("break_start").reset_index(drop=True)
y = frame["log_effect"].to_numpy(dtype=float)
cells = frame["channel_name"].to_numpy()

print("breaks", len(y))
print("cells", len(set(cells)))
print("window", f"{frame['break_start'].min():%Y-%m-%d} to {frame['break_start'].max():%Y-%m-%d}")
print("target sd (numpy default ddof=0)", round(float(y.std()), 9))
print("target sd (ddof=1)", round(float(y.std(ddof=1)), 9))


def predict(coefficients):
    values = np.array([coefficients.get(c, np.nan) for c in cells], dtype=float)
    fallback = float(np.mean(list(coefficients.values()))) if coefficients else 0.0
    missing = int(np.isnan(values).sum())
    return np.nan_to_num(values, nan=fallback), missing


shipped = json.loads((ROOT / "models/tv_break_coefficients.json").read_text())
sc = shipped["coefficients"]
sp, smiss = predict(sc)
se = (y - sp) ** 2
srmse = float(np.sqrt(se.mean()))
print("shipped rmse", round(srmse, 9), "missing", smiss)

folds = [b for b in np.array_split(np.arange(len(y)), 5) if b.size]

rows = {}
for path in sorted((ROOT / "models/candidates").glob("tv_break_coefficients_*.json")):
    ident = path.stem.replace("tv_break_coefficients_", "", 1)
    payload = json.loads(path.read_text())
    cc = payload["coefficients"]
    cp, cmiss = predict(cc)
    ce = (y - cp) ** 2
    crmse = float(np.sqrt(ce.mean()))
    d = ce - se
    mean = float(d.mean())
    sem = float(d.std(ddof=1) / np.sqrt(len(d)))
    stat = mean / sem if sem else 0.0
    fold_moves = [float(np.sqrt(ce[b].mean()) - np.sqrt(se[b].mean())) for b in folds]
    disp = float(np.std(fold_moves, ddof=1))
    # coefficient delta
    keys = sorted(set(sc) | set(cc))
    moved = 0
    largest, largest_at = 0.0, None
    contrib = {}
    for k in keys:
        a, b = sc.get(k), cc.get(k)
        if a is None or b is None:
            continue
        if abs(b - a) > 1e-12:
            moved += 1
        if abs(b - a) > largest:
            largest, largest_at = abs(b - a), k
        mask = cells == k
        contrib[k] = float(d[mask].sum())
    net = sum(contrib.values())
    absolute = sum(abs(v) for v in contrib.values())
    cancelled = 1.0 - abs(net) / absolute if absolute else None
    rows[ident] = dict(rmse=round(crmse, 9), delta=round(crmse - srmse, 9),
                       stat=round(stat, 4), disp=round(disp, 9), missing=cmiss,
                       moved=moved, of=len([k for k in keys if k in sc and k in cc]),
                       largest=round(largest, 9), largest_at=largest_at,
                       cancelled=None if cancelled is None else round(cancelled, 6),
                       identical=bool(np.array_equal(cp, sp)))
    print(ident, json.dumps(rows[ident], ensure_ascii=False))

# baselines
total, count = y.sum(), len(y)
glob = (total - y) / (count - 1)
f2 = pd.DataFrame({"g": cells, "y": y})
sums = f2.groupby("g")["y"].transform("sum").to_numpy(dtype=float)
sizes = f2.groupby("g")["y"].transform("size").to_numpy(dtype=float)
inside = (sums - y) / (sizes - 1.0)
cellloo = np.where(sizes > 1.0, inside, glob)
print("global_mean_loo", round(float(np.sqrt(((y - glob) ** 2).mean())), 9))
print("cell_mean_loo", round(float(np.sqrt(((y - cellloo) ** 2).mean())), 9))

# cross-check the stored payload
stored = json.loads((ROOT / "models/releases/holdout_rescores.json").read_text())
print("--- stored vs recomputed ---")
print("stored shipped rmse", stored["shipped"]["rmse"], "mine", round(srmse, 9),
      "MATCH" if abs(stored["shipped"]["rmse"] - srmse) < 5e-10 else "DIFFER")
for row in stored["candidates"]:
    mine = rows[row["id"]]
    ok = (abs(row["rmse"] - mine["rmse"]) < 5e-10
          and abs(row["paired"]["paired_statistic"] - mine["stat"]) < 1e-3
          and abs(row["paired"]["fold_dispersion"] - mine["disp"]) < 5e-9
          and row["cell_deltas"]["summary"]["cells_moved"] == mine["moved"]
          and (row["cell_deltas"]["summary"]["cancelled_share"] is None
               or abs(row["cell_deltas"]["summary"]["cancelled_share"] - (mine["cancelled"] or 0)) < 1e-5))
    print(row["id"], "stored", row["rmse"], row["paired"]["paired_statistic"],
          row["cell_deltas"]["summary"]["cells_moved"],
          row["cell_deltas"]["summary"]["cancelled_share"],
          "| mine", mine["rmse"], mine["stat"], mine["moved"], mine["cancelled"],
          "|", "MATCH" if ok else "DIFFER")
print("stored target_sd", stored["evaluation"]["target_sd"])
print("stored baselines", {b["id"]: b["rmse"] for b in stored["baselines"]})
