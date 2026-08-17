"""Measure the rating forecast's out-of-sample accuracy and write the artifact.

Walks the observation window forward (:mod:`kairos.model.forecast_backtest`):
each contiguous date block is forecast by a model refitted on the blocks before
it -- pooled base, all eight family gates, factor tables and the dispersion
behind the interval -- and scored against what was actually measured. Writes
``models/forecast_accuracy.json`` with the same discipline
``scripts/compute_audience_model.py`` uses: a ``computed_at`` stamp and a
``source_fingerprints`` map, so a later reader can tell whether the figures still
describe the data on disk.

Deterministic given the data. Only ``computed_at`` reads the clock; the per-fold
model stamps are derived from each fold's own training window.

This script CHANGES NO SHIPPED NUMBER. It measures the forecast, it does not
refit the shipped artifact, and it never writes ``models/audience_model.json``.

Run from the repo root:

    PYTHONUTF8=1 python scripts/compute_forecast_accuracy.py
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from kairos.data.loaders import REFERENCE_DIR, _resolve_reference_path
from kairos.model.audience_model import ARTIFACT_PATH as AUDIENCE_ARTIFACT
from kairos.model.forecast_backtest import DEFAULT_BLOCKS, walk_forward
from kairos.model.forecast_basis import DEFAULT_LEVEL
from kairos.observability.run_log import checksum_file
from kairos.optimize.event_pricing import DEFAULT_EVENTS_PATH

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "models" / "forecast_accuracy.json"

# The sources the measurement rests on: the spots history the folds are cut
# from, the deterministic calendar and the events store the features read, and
# the audience artifact the backtest takes its owned-channel scope from.
_CALENDAR_PATH = ROOT / "kairos" / "config" / "israel_calendar.csv"


def _source_fingerprints() -> dict[str, str]:
    """Relative POSIX path -> sha256 for every source that fed the measurement."""
    sources = (
        _resolve_reference_path(REFERENCE_DIR / "Spots.xlsx"),
        _CALENDAR_PATH,
        DEFAULT_EVENTS_PATH,
        AUDIENCE_ARTIFACT,
    )
    prints: dict[str, str] = {}
    for path in sources:
        digest = checksum_file(path)
        if digest is not None:
            prints[Path(path).relative_to(ROOT).as_posix()] = digest
    return prints


def _print_report(report: dict) -> None:
    overall = report.get("overall", {})
    verdict = report.get("verdict", {})
    if not overall.get("available"):
        print(f"  nothing scored: {overall.get('reason', report.get('reason'))}")
        return
    print(f"  scored {overall['n']} observations over "
          f"{report['n_folds_scored']} walk-forward folds")
    print(f"  points   model  MAE {overall['mae']:.4f}  RMSE {overall['rmse']:.4f}  "
          f"bias {overall['bias']:+.4f}")
    print(f"  points   history MAE {overall['historical_mae']:.4f}  "
          f"RMSE {overall['historical_rmse']:.4f}  bias {overall['historical_bias']:+.4f}")
    print(f"  log      model RMSE {overall['log_rmse']:.4f}  "
          f"history RMSE {overall['historical_log_rmse']:.4f}")
    mape = overall.get("mape")
    print(f"  MAPE     {mape if mape is None else f'{mape:.3f}%'} over "
          f"{overall['mape_n']} rows ({overall['mape_excluded_n']} excluded)")
    coverage = overall.get("interval_coverage")
    print(f"  coverage {coverage} at level {overall['interval_level']} "
          f"(mean width {overall.get('interval_mean_width')} points)")
    if verdict.get("available"):
        print(f"  VERDICT  {verdict['headline_en']}")
        if verdict.get("mechanism_note_en"):
            print(f"           {verdict['mechanism_note_en']}")
    for gap in report.get("gaps", []):
        print(f"  gap [{gap['kind']}] {gap['reason']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ARTIFACT_PATH),
                        help="Where to write the artifact (default: models/forecast_accuracy.json).")
    parser.add_argument("--blocks", type=int, default=DEFAULT_BLOCKS,
                        help=f"Walk-forward date blocks (default: {DEFAULT_BLOCKS}).")
    parser.add_argument("--level", type=float, default=DEFAULT_LEVEL,
                        help=f"Interval level whose coverage is scored (default: {DEFAULT_LEVEL}).")
    args = parser.parse_args()

    report = walk_forward(n_blocks=args.blocks, level=args.level)
    payload = {
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "source_fingerprints": _source_fingerprints(),
        **report,
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Wrote forecast accuracy to {target}")
    print(f"  computed_at: {payload['computed_at']}")
    _print_report(report)


if __name__ == "__main__":
    main()
