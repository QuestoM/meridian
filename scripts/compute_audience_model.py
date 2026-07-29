"""Rebuild the audience (expected TVR) model artifact.

Builds the training frame from the multi-channel aired-spots history, fits
the pooled multiplicative base, re-measures every family gate on five
temporal folds (+2 percent held-out bar, self-activating each rebuild), and
writes ``models/audience_model.json``. Deterministic given the data: only the
``computed_at`` stamp reads the clock.

The artifact records verdict off with the honest reason for any family whose
source is absent or contrast-free in the window; nothing is forced on. The
activation flag stays with the operator (settings key
``audience_model_activation``, default off), so writing the artifact changes
no shipped number.

Run from the repo root:

    PYTHONUTF8=1 python scripts/compute_audience_model.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from kairos.data.loaders import REFERENCE_DIR, _resolve_reference_path
from kairos.model.audience_factors import FAMILIES
from kairos.model.audience_model import ARTIFACT_PATH, fit_audience_model
from kairos.observability.run_log import checksum_file
from kairos.optimize.event_pricing import DEFAULT_EVENTS_PATH

ROOT = Path(__file__).resolve().parents[1]

# The sources the model is measured from: the spots history (resolved through
# the same xlsx-then-uploaded-CSV fallback the loader uses), the deterministic
# Israeli calendar table, and the operator events store. A change to any of
# them is detectable as staleness by re-hashing; nothing else is claimed.
_CALENDAR_PATH = ROOT / "kairos" / "config" / "israel_calendar.csv"


def _source_fingerprints() -> dict[str, str]:
    """Relative POSIX path -> sha256 for every source that fed the fit."""
    sources = (
        _resolve_reference_path(REFERENCE_DIR / "Spots.xlsx"),
        _CALENDAR_PATH,
        DEFAULT_EVENTS_PATH,
    )
    prints: dict[str, str] = {}
    for path in sources:
        digest = checksum_file(path)
        if digest is not None:
            prints[path.relative_to(ROOT).as_posix()] = digest
    return prints


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(ARTIFACT_PATH),
        help="Where to write the artifact (default: models/audience_model.json).",
    )
    args = parser.parse_args()

    model = fit_audience_model(source_fingerprints=_source_fingerprints())
    written = model.write_artifact(Path(args.output))

    summary = model.base.summary()
    print(f"Wrote audience model to {written}")
    print(f"  computed_at: {model.computed_at}")
    print(
        f"  observations: {summary['n_observations']} across "
        f"{len(summary['channels'])} channels, {summary['n_series']} series"
    )
    print(f"  owned channel: {model.owned_channel or '(none configured)'}")
    print(f"  activation default: {model.activation_default} (operator settings flag decides)")
    for family in FAMILIES:
        gate = model.gates[family]
        print(f"  gate {family}: {gate['verdict']} ({gate['reason']})")


if __name__ == "__main__":
    main()
