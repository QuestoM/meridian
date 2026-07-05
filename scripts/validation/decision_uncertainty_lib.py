"""Shared harness for the decision-robustness review of the retention model.

Everything here runs the REAL repo engine on the REAL data, in memory only:

  * the decision process replayed is exactly the shipped one
    (:func:`kairos.service.run_scenario` with the saved dashboard settings:
    blend objective, revenue_weight, retention floor, breaks/hour, risk_lambda),
  * coefficient perturbations are injected by monkeypatching the loader seam
    (``kairos.service.load_impact_model``) with a real
    :class:`~kairos.model.impact.PosteriorImpactModel` built from drawn values,
    never by touching ``models/tv_break_coefficients.json`` on disk,
  * plans are priced in ILS with the product's own money model
    (:mod:`kairos.optimize.revenue_net`), never a reimplementation,
  * nothing writes ``output/weekly_break_schedule.csv`` (the shipped plan is
    only READ from it) and nothing calls ``write_weekly_schedule``.

Scope: the owned channel's representative day. The owned channel comes from the
saved settings' ``operator_channel``; the representative day is the channel-day
whose shipped gross revenue is closest to the channel's median across the saved
schedule (ties broken by the earliest date), a deterministic, defensible pick.

Determinism: coefficient draws use ``numpy.random.default_rng(seed)`` over the
artifact's cells in sorted-name order, so every script and test reproduces the
same draws bit-for-bit for a given (K, seed).
"""

from __future__ import annotations

import json
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from kairos.data.transform import build_segments_from_programmes
from kairos.data.loaders import load_programmes
from kairos.model.impact import PosteriorImpactModel, RetentionEstimate
from kairos.optimize._segment_math import _segment_revenue
from kairos.optimize._types import ProgramSegment
from kairos.optimize.pricing import OptimizerAssumptions, pricing_from_settings
from kairos.optimize.revenue_net import segment_retention_cost_ils
import kairos.service as kairos_service

SETTINGS_PATH = ROOT / "data" / "kairos_settings.json"
COEFFICIENTS_PATH = ROOT / "models" / "tv_break_coefficients.json"
SHIPPED_CSV_PATH = ROOT / "output" / "weekly_break_schedule.csv"
RESULTS_DIR = ROOT / "docs" / "model-validation" / "results"

# The artifact's credible intervals are 95% (theta +/- 1.96 * sd in log space,
# see kairos/model/measure.py), so the CI-implied sd in delta space is width/3.92.
_CI_Z = 1.96
# Sign-plausible range for a per-break retention delta: a break cannot raise
# retention (<= 0) and cannot shed more than the whole audience (>= -1).
_COEF_LOW, _COEF_HIGH = -1.0, 0.0


# --------------------------------------------------------------------------- #
# Context: the fixed scope everything in the review is computed on.
# --------------------------------------------------------------------------- #

@dataclass
class ReviewContext:
    """The fixed, deterministic scope for the whole review."""

    settings: dict[str, Any]
    channel: str
    day: str
    programmes: pd.DataFrame
    shipped_counts: dict[str, int]          # segment_id -> num_breaks from the saved CSV
    shipped_day_revenue: float              # gross ILS the saved CSV reports for the day
    detail: dict[str, dict[str, Any]]       # raw per-cell artifact detail (36 cells)
    cells: list[str]                        # sorted cell names (draw column order)
    metadata: dict[str, Any]                # artifact metadata (computed_at etc.)
    classifier: Any = field(repr=False, default=None)
    pricing: Any = field(repr=False, default=None)
    assumptions: OptimizerAssumptions = field(default_factory=OptimizerAssumptions)


def representative_day(csv: pd.DataFrame, channel: str) -> str:
    """The channel-day whose shipped gross revenue is closest to the median.

    Deterministic: distance to the channel's median day revenue, ties broken by
    the earliest date. This is the 'typical money day', not a cherry-pick.
    """
    own = csv[csv["channel"] == channel]
    if own.empty:
        raise ValueError(f"No rows for channel {channel!r} in the saved schedule")
    per_day = own.groupby("date")["predicted_revenue"].sum()
    median = per_day.median()
    ranked = (per_day - median).abs().sort_values(kind="mergesort")
    tied = ranked[ranked == ranked.iloc[0]].index
    return str(sorted(tied)[0])


def load_context(day: Optional[str] = None) -> ReviewContext:
    """Load settings, data, the shipped plan and the artifact, once."""
    settings = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    channel = str(settings.get("operator_channel", "") or "")
    if not channel:
        raise ValueError("settings carry no operator_channel; cannot pick the owned channel")

    csv = pd.read_csv(SHIPPED_CSV_PATH)
    chosen_day = day or representative_day(csv, channel)
    day_rows = csv[(csv["channel"] == channel) & (csv["date"] == chosen_day)]
    if day_rows.empty:
        raise ValueError(f"No shipped rows for {channel!r} on {chosen_day}")
    shipped_counts = {
        str(sid): int(k) for sid, k in zip(day_rows["segment_id"], day_rows["num_breaks"])
    }

    artifact = json.loads(COEFFICIENTS_PATH.read_text(encoding="utf-8"))
    detail = dict(artifact["detail"])
    cells = sorted(detail)

    programmes = load_programmes()

    # Mirror run_scenario's internals exactly (same seams, same defaults) so the
    # evaluation segments this harness builds are the ones the service would build.
    assumptions = kairos_service._apply_first_break_multiplier(OptimizerAssumptions())
    pricing = pricing_from_settings(settings)
    classifier = kairos_service._build_classifier()

    return ReviewContext(
        settings=settings,
        channel=channel,
        day=chosen_day,
        programmes=programmes,
        shipped_counts=shipped_counts,
        shipped_day_revenue=float(day_rows["predicted_revenue"].sum()),
        detail=detail,
        cells=cells,
        metadata=dict(artifact.get("metadata", {})),
        classifier=classifier,
        pricing=pricing,
        assumptions=assumptions,
    )


# --------------------------------------------------------------------------- #
# Coefficient draws and perturbed impact models.
# --------------------------------------------------------------------------- #

def cell_sd(detail_entry: Mapping[str, Any]) -> float:
    """CI-implied standard deviation of one cell's coefficient (delta space)."""
    return (float(detail_entry["ci_high"]) - float(detail_entry["ci_low"])) / (2.0 * _CI_Z)


def draw_coefficient_vectors(
    detail: Mapping[str, Mapping[str, Any]],
    k: int,
    seed: int = 42,
) -> list[dict[str, float]]:
    """K seeded draws of the full 36-cell coefficient vector.

    Each cell is drawn Normal(coefficient, CI-implied sd) independently and
    truncated (clipped) to the sign-plausible range [-1, 0]. With coefficients
    near -0.04 and sds near 0.01 the clip probability is ~Phi(-4) per cell, so
    clipping is numerically indistinguishable from exact truncation here.
    Cells are ordered by sorted name so draws are reproducible for a given seed.
    """
    cells = sorted(detail)
    means = np.array([float(detail[c]["coefficient"]) for c in cells])
    sds = np.array([cell_sd(detail[c]) for c in cells])
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((k, len(cells)))
    drawn = np.clip(means + sds * z, _COEF_LOW, _COEF_HIGH)
    return [dict(zip(cells, map(float, row))) for row in drawn]


def make_impact_model(
    cell_coefficients: Mapping[str, float],
    detail: Mapping[str, Mapping[str, Any]],
    *,
    degenerate_ci: bool = True,
    default: Optional[float] = None,
) -> PosteriorImpactModel:
    """A real PosteriorImpactModel carrying the given per-cell coefficients.

    ``degenerate_ci=True`` sets ci_low == ci_high == coefficient (a drawn 'truth'
    is known exactly in its world; with the shipped risk_lambda=0 the interval is
    decision-inert anyway). ``default`` is the coefficient for any unmeasured
    cell; it defaults to the engine's declared assumption, held FIXED across
    draws because it carries no measured interval to draw from (on the review
    scope every lookup hits a measured cell, asserted separately).
    """
    assumptions = OptimizerAssumptions()
    default_value = assumptions.retention_impact_per_break if default is None else float(default)
    estimates = {}
    for cell, value in cell_coefficients.items():
        base = detail.get(cell, {})
        n = int(base.get("n", 0))
        if degenerate_ci:
            lo = hi = float(value)
        else:
            lo, hi = float(base.get("ci_low", value)), float(base.get("ci_high", value))
        estimates[cell] = RetentionEstimate(
            coefficient=float(value), ci_low=lo, ci_high=hi, n=n,
            confidence=str(base.get("confidence", "low")) or "low",
        )
    return PosteriorImpactModel(
        dict(cell_coefficients),
        default=default_value,
        source="measured",
        detail=estimates,
        series={},
    )


def constant_model(mu: float, detail: Mapping[str, Mapping[str, Any]]) -> PosteriorImpactModel:
    """One global constant coefficient for every cell (and the default)."""
    return make_impact_model(
        {cell: float(mu) for cell in detail}, detail, degenerate_ci=True, default=float(mu),
    )


def pooled_mean(detail: Mapping[str, Mapping[str, Any]]) -> float:
    """The n-weighted mean coefficient across all measured cells."""
    total_n = sum(int(v["n"]) for v in detail.values())
    if total_n <= 0:
        raise ValueError("artifact carries no measured breaks")
    return sum(int(v["n"]) * float(v["coefficient"]) for v in detail.values()) / total_n


@contextmanager
def patched_impact(model: Any) -> Iterator[None]:
    """Point kairos.service's impact-model loader at ``model``, then restore.

    run_scenario resolves ``load_impact_model`` through its own module globals,
    so rebinding that name is the clean in-memory seam; the shipped artifact on
    disk is never touched.
    """
    original = kairos_service.load_impact_model
    kairos_service.load_impact_model = lambda path, *, assumptions=None, coefficients_path=None: model
    try:
        yield
    finally:
        kairos_service.load_impact_model = original


# --------------------------------------------------------------------------- #
# Re-optimization (the shipped decision process) and ILS evaluation.
# --------------------------------------------------------------------------- #

def reoptimize(
    ctx: ReviewContext,
    impact_model: Any = None,
    *,
    risk_lambda: Optional[float] = None,
    refine: bool = True,
) -> tuple[dict[str, int], dict[str, Any]]:
    """Run the shipped decision process on the review scope, in memory.

    This is :func:`kairos.service.run_scenario` with the saved settings (blend
    objective, saved revenue_weight / floor / breaks-per-hour), optionally with a
    perturbed impact model injected at the loader seam and/or a risk_lambda
    override. Returns (counts by segment_id, full payload).
    """
    risk = float(ctx.settings["risk_lambda"]) if risk_lambda is None else float(risk_lambda)

    def _run() -> dict[str, Any]:
        return kairos_service.run_scenario(
            revenue_weight=ctx.settings["revenue_weight"],
            retention_floor=ctx.settings["min_retention_floor"],
            max_breaks_per_hour=ctx.settings["max_breaks_per_hour"],
            risk_lambda=risk,
            channel=ctx.channel,
            day=ctx.day,
            programmes=ctx.programmes,
            refine=refine,
            settings=ctx.settings,
        )

    if impact_model is None:
        payload = _run()
    else:
        with patched_impact(impact_model):
            payload = _run()
    counts = {str(s["segment_id"]): int(s["num_breaks"]) for s in payload["segments"]}
    return counts, payload


def build_segments(ctx: ReviewContext, impact_model: Any) -> list[ProgramSegment]:
    """Build the review day's segments under ``impact_model``.

    Identical construction to run_scenario's internals (same classifier, pricing,
    assumptions), verified by the revenue-consistency assertion in
    :func:`evaluate_counts` callers.
    """
    return build_segments_from_programmes(
        ctx.programmes,
        ctx.classifier,
        ctx.pricing,
        assumptions=ctx.assumptions,
        impact_model=impact_model,
        channel=ctx.channel,
        day=ctx.day,
    )


def evaluate_counts(
    segments: Sequence[ProgramSegment],
    counts: Mapping[str, int],
) -> dict[str, float]:
    """Price a (segments, break-counts) plan in ILS with the product money model.

    gross = the optimizer's own revenue at these counts (per-break, retention-
    discounted); cost = the retention cost the revenue-net machinery charges
    (:func:`kairos.optimize.revenue_net.segment_retention_cost_ils`); net is the
    difference. Segments absent from ``counts`` carry zero breaks.
    """
    gross = 0.0
    cost = 0.0
    breaks = 0
    for segment in segments:
        k = int(counts.get(segment.segment_id, 0))
        breaks += k
        gross += _segment_revenue(segment, k)
        cost += segment_retention_cost_ils(segment, k)
    return {
        "gross_ils": gross,
        "retention_cost_ils": cost,
        "net_ils": gross - cost,
        "breaks": float(breaks),
    }


def class_means_at_standard(cell_coefficients: Mapping[str, float]) -> dict[str, float]:
    """The 4 effective per-pricing-class coefficients a plan actually decides with.

    Segment construction reads the model at the segment's pricing class, averaged
    over the three position buckets at the STANDARD length bucket (every
    programmes-path segment uses the default 120s break length). So exactly 12 of
    the 36 cells reach the plan, collapsed to these 4 numbers.
    """
    classes = ("News", "PrimeShow1", "PrimeShow2", "Other")
    positions = ("first", "middle", "last")
    return {
        cls: float(np.mean([cell_coefficients[f"{cls}_{pos}_standard"] for pos in positions]))
        for cls in classes
    }


def verify_segment_mapping(
    segments: Sequence[ProgramSegment],
    cell_coefficients: Mapping[str, float],
) -> None:
    """Assert every segment's coefficient equals one of the 4 class means.

    Guards two things at once: (a) the perturbation reached the segments through
    the real loader/transform path, and (b) no segment silently fell back to the
    declared default (which would mean an unmeasured cell on this scope).
    """
    expected = set(np.round(list(class_means_at_standard(cell_coefficients).values()), 12))
    for segment in segments:
        if round(segment.impact_coefficient, 12) not in expected:
            raise AssertionError(
                f"segment {segment.segment_id} coefficient {segment.impact_coefficient} "
                f"matches no expected class mean {sorted(expected)}"
            )


def hamming_share(
    counts_a: Mapping[str, int],
    counts_b: Mapping[str, int],
) -> tuple[float, int]:
    """(share, count) of segments whose break count differs between two plans."""
    keys = set(counts_a) | set(counts_b)
    changed = sum(1 for key in keys if int(counts_a.get(key, 0)) != int(counts_b.get(key, 0)))
    return (changed / len(keys) if keys else 0.0), changed


def assert_revenue_consistency(
    payload: Mapping[str, Any],
    evaluation: Mapping[str, float],
    tolerance: float = 0.05,
) -> None:
    """The harness-built segments must reprice the service's own plan to the cent.

    ``payload`` is the run_scenario result, ``evaluation`` the harness pricing of
    the SAME counts under the SAME coefficients. Any daylight between the two
    means the harness's segment construction drifted from the service internals
    (for example after a concurrent engine edit), so fail loudly.
    """
    reported = float(payload["summary"]["projected_revenue"])
    computed = float(evaluation["gross_ils"])
    if abs(reported - computed) > tolerance:
        raise AssertionError(
            f"harness gross {computed:.2f} != service gross {reported:.2f}; "
            "segment construction has drifted from kairos.service internals"
        )


# --------------------------------------------------------------------------- #
# Reporting helpers.
# --------------------------------------------------------------------------- #

def percentiles(values: Sequence[float], points: Sequence[int] = (10, 50, 90)) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    return {f"p{p}": float(np.percentile(arr, p)) for p in points}


def provenance() -> dict[str, Any]:
    """Repo state stamped into every result file (a peer edits the engine live)."""
    def _git(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True,
            ).stdout.strip()
        except Exception:
            return "unknown"

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_head": _git("rev-parse", "HEAD"),
        "kairos_dirty": _git("status", "--porcelain", "kairos", "kairos_api"),
        "coefficients_computed_at": None,  # filled by callers from ctx.metadata
        "python": sys.version.split()[0],
    }


def write_results(name: str, payload: Mapping[str, Any]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    target = RESULTS_DIR / name
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return target
