"""Prepare the real TV data and train the Meridian impact posterior.

This CLI is the owner-flagged entrypoint for fitting the retention-impact model.
It builds a Meridian ``InputData`` from the reference data with
:func:`kairos.model.prepare.build_meridian_input_data`, then hands it to
:func:`kairos.model.train.train_tv_break_model`. Sampling counts are reduced from
Meridian's defaults to stay CPU-tractable and are overridable on the command
line.

Training needs Python 3.11 or 3.12 plus tensorflow and google-meridian. If
:func:`kairos.model.train.can_train` reports the environment cannot train, this
prints a clear message and exits non-zero rather than fabricating a model.

Usage:
    python scripts/train_impact_model.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kairos.model.prepare import build_meridian_input_data  # noqa: E402
from kairos.model.train import can_train, train_tv_break_model  # noqa: E402

DEFAULT_OUTPUT_PATH = Path("models") / "tv_break_posterior.pkl"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    parser = argparse.ArgumentParser(description="Prepare TV data and train the Meridian impact model")
    parser.add_argument("--programmes", type=Path, default=None, help="path to Programmes.xlsx")
    parser.add_argument("--spots", type=Path, default=None, help="path to Spots.xlsx")
    parser.add_argument("--dayparts", type=Path, default=None, help="path to Dayparts.xlsx")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="where to save the posterior")
    parser.add_argument("--n-chains", type=int, default=2, help="MCMC chains (reduced for CPU)")
    parser.add_argument("--n-adapt", type=int, default=200, help="adaptation draws (reduced for CPU)")
    parser.add_argument("--n-burnin", type=int, default=200, help="burn-in draws (reduced for CPU)")
    parser.add_argument("--n-keep", type=int, default=300, help="kept draws (reduced for CPU)")
    args = parser.parse_args()

    if not can_train():
        print(
            "Cannot train the Meridian impact model in this environment.\n"
            "Training needs Python 3.11 or 3.12 with tensorflow and google-meridian installed.\n"
            "Install the training stack on a supported Python, then rerun this script."
        )
        return 1

    print("Building the Meridian InputData from the reference data ...")
    input_data = build_meridian_input_data(
        programmes_path=args.programmes,
        spots_path=args.spots,
        dayparts_path=args.dayparts,
    )

    print("Training the Meridian impact posterior (this is slow) ...")
    destination = train_tv_break_model(
        input_data,
        output_path=args.output,
        n_chains=args.n_chains,
        n_adapt=args.n_adapt,
        n_burnin=args.n_burnin,
        n_keep=args.n_keep,
    )
    print(f"Saved trained posterior to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
