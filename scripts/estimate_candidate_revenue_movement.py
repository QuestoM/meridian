"""Revenue movement if a candidate coefficients artifact were adopted.

Runs the exact recompute path the golden test pins (saved settings, saved
risk_lambda, saved operator_channel) three times IN MEMORY: once with the
shipped ``models/tv_break_coefficients.json`` and once per candidate under
``models/candidates/``, passing the candidate through the public
``impact_model`` parameter of :func:`kairos.export.schedule.build_weekly_schedule`.
Nothing is written: no output CSV, no artifact, no server. The deltas are the
honest answer to "what would adopting this candidate move".

    PYTHONUTF8=1 python scripts/estimate_candidate_revenue_movement.py
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kairos.export.schedule import DEFAULT_IMPACT_MODEL_PATH, build_weekly_schedule
from kairos.model.impact import load_impact_model
from kairos.optimize.pricing import OptimizerAssumptions

SETTINGS_PATH = ROOT / "data" / "kairos_settings.json"
CANDIDATES = {
    "spotclip": ROOT / "models" / "candidates" / "tv_break_coefficients_spotclip.json",
    "competitor": ROOT / "models" / "candidates" / "tv_break_coefficients_competitor.json",
}


def _build(impact_model=None):
    settings = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    frame = build_weekly_schedule(
        settings=settings,
        revenue_weight=settings["revenue_weight"] / 100.0,
        risk_lambda=settings["risk_lambda"],
        operator_channel=settings["operator_channel"],
        today=date.today(),
        impact_model=impact_model,
    )
    return {
        "revenue": float(frame["predicted_revenue"].sum()),
        "retention": float(frame["predicted_retention"].sum()),
        "breaks": int(frame["num_breaks"].sum()),
        "rows": len(frame),
    }


def main() -> None:
    base = _build()
    print(f"shipped: revenue {base['revenue']:,.0f}, retention-sum {base['retention']:,.1f}, "
          f"breaks {base['breaks']}, rows {base['rows']}")
    for name, path in CANDIDATES.items():
        if not path.exists():
            print(f"{name}: candidate missing at {path}")
            continue
        model = load_impact_model(
            DEFAULT_IMPACT_MODEL_PATH,
            assumptions=OptimizerAssumptions(),
            coefficients_path=path,
        )
        got = _build(impact_model=model)
        d_rev = got["revenue"] - base["revenue"]
        print(f"{name}: revenue {got['revenue']:,.0f} ({d_rev:+,.0f}, "
              f"{100*d_rev/base['revenue']:+.3f}%), retention-sum {got['retention']:,.1f} "
              f"({got['retention']-base['retention']:+,.1f}), breaks {got['breaks']} "
              f"({got['breaks']-base['breaks']:+d})")


if __name__ == "__main__":
    main()
