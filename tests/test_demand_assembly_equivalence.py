"""Equivalence test for the two advertiser-demand assembly folds.

The live optimize path folds advertiser demand, inventory awareness and delivery
pacing into per-segment placement weights inside
:func:`kairos.service._assemble_demand_weights`. The weekly-CSV export path
re-inlines the same fold inside
:func:`kairos.export.schedule.build_weekly_schedule`. Both converge on
:func:`kairos.optimize.demand.build_demand_weights`, but they assemble its inputs
with independent code (engine construction, inventory build, and the pacing
gate/knob threading). The production corpus is all-1.0, so a divergence in that
assembly would be invisible there.

This test drives a SYNTHETIC, deliberately NON-uniform signal through BOTH folds
and asserts byte-identical per-segment weight maps. It is the safety net for the
Phase-1 consolidation that will merge the two folds: if either fold changes its
result, the equivalence assertion goes red. A second test proves the assertion
is not vacuous by making the two folds see different pacing gating and confirming
their maps then differ.

The test uses only the standard library for patching (no pytest dependency) so it
runs under the project venv directly (``python tests/test_demand_assembly_equivalence.py``)
as well as being collectable by pytest.
"""
from __future__ import annotations

import contextlib
import json
import sys
import types
from datetime import date
from pathlib import Path
from unittest import mock

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import kairos.export.schedule as export_schedule  # noqa: E402
import kairos.service as service  # noqa: E402
from kairos.optimize._types import ProgramSegment  # noqa: E402
from kairos.optimize.advertiser_rules import AdvertiserRuleEngine  # noqa: E402

REFERENCE_DATE = date(2024, 11, 1)
SETTINGS_PATH = ROOT / "data" / "kairos_settings.json"

# Non-uniform synthetic signals keyed by programme genre. Values are chosen so
# the folded product spans the clamp band [0.25, 4.0] with entries both above and
# below 1.0, so all three signals (demand boost, inventory boost, two-sided
# pacing) leave a distinct footprint and no segment collapses to the trivial 1.0.
DEMAND_BY_GENRE = {"News": 1.5, "Movie": 2.0, "Sports": 3.0, "Kids": 1.0, "Drama": 1.25, "Reality": 1.75}
INVENTORY_BY_GENRE = {"News": 1.2, "Movie": 1.0, "Sports": 1.1, "Kids": 1.3, "Drama": 1.0, "Reality": 1.0}
PACING_BY_GENRE = {"News": 1.0, "Movie": 0.5, "Sports": 1.3, "Kids": 0.6, "Drama": 1.1, "Reality": 0.9}


def _settings() -> dict:
    """The saved dashboard settings (carries the pacing knobs both folds read)."""
    return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))


def _segments() -> list[ProgramSegment]:
    """A handful of synthetic segments, one per genre, spread across dayparts."""
    segs = []
    for i, genre in enumerate(DEMAND_BY_GENRE):
        hour = 6 + i * 3
        segs.append(
            ProgramSegment(
                segment_id=f"seg-{i}",
                channel="SYN",
                day="2024-11-01",
                start_seconds=float(hour * 3600),
                duration_seconds=1800.0,
                program_type=genre,
                baseline_tvr=10.0,
                cpp=100.0,
                program_title=f"show-{genre}",
            )
        )
    return segs


class _StubEngine:
    """Advertiser rule engine stub returning a non-uniform, genre-keyed demand."""

    def segment_demand(self, *, channel=None, genre=None, daypart=None, programme=None) -> float:
        return max(1.0, DEMAND_BY_GENRE.get(genre, 1.0))

    def pacing_overrides(self) -> dict:
        return {}


def _stub_inventory_weights(segments, pool):
    return {s.segment_id: INVENTORY_BY_GENRE.get(s.program_type, 1.0) for s in segments}


def _stub_pacing_weights(segments, campaigns, reference, *, daypart_of=None, advertiser_k_of=None, **knobs):
    return {s.segment_id: PACING_BY_GENRE.get(s.program_type, 1.0) for s in segments}


@contextlib.contextmanager
def _patched_signals():
    """Inject the synthetic signals into BOTH folds' shared dependencies.

    ``build_demand_weights`` (the fold under protection) is deliberately left
    real; only its inputs are stubbed, and identically for both folds.
    """
    stub = _StubEngine()
    with contextlib.ExitStack() as stack:
        # from_files is a classmethod on the shared class object, so one patch
        # feeds both the service fold and the export fold.
        stack.enter_context(
            mock.patch.object(AdvertiserRuleEngine, "from_files", classmethod(lambda cls, *a, **k: stub))
        )
        for mod in (service, export_schedule):
            stack.enter_context(mock.patch.object(mod, "build_inventory_weights", _stub_inventory_weights))
            stack.enter_context(mock.patch.object(mod, "load_inventory", lambda *a, **k: {}))
            # Non-empty so the pacing gate fires; the stub ignores the contents.
            stack.enter_context(mock.patch.object(mod, "load_campaigns", lambda *a, **k: ["campaign"]))
            stack.enter_context(mock.patch.object(mod, "build_pacing_weights", _stub_pacing_weights))
        yield stub


def _run_live_fold(segments, settings, today) -> dict:
    pacing_knobs = service._pacing_knobs_from_settings(settings)
    return service._assemble_demand_weights(list(segments), today=today, pacing_knobs=pacing_knobs)


def _run_csv_fold(segments, settings, today) -> dict:
    """Drive the real ``build_weekly_schedule`` inline fold and capture its output.

    The heavy pieces around the fold are stubbed so only the demand-assembly path
    runs: segment construction is replaced by the synthetic segments, the channel
    grid is a single channel-day, the posterior load is skipped, and
    ``optimize_breaks`` records the ``demand_weights`` the inline fold produced.
    """
    captured: dict = {}

    def _fake_optimize(segs, guardrails, **kwargs):
        captured["demand_weights"] = kwargs.get("demand_weights")
        return types.SimpleNamespace(segments=())

    with contextlib.ExitStack() as stack:
        stack.enter_context(
            mock.patch.object(export_schedule, "build_segments_from_programmes", lambda *a, **k: list(segments))
        )
        stack.enter_context(
            mock.patch.object(export_schedule, "_channel_days", lambda programmes: [("SYN", "2024-11-01")])
        )
        stack.enter_context(mock.patch.object(export_schedule, "load_impact_model", lambda *a, **k: None))
        stack.enter_context(mock.patch.object(export_schedule, "optimize_breaks", _fake_optimize))
        export_schedule.build_weekly_schedule(
            programmes=pd.DataFrame({"_": [0]}),
            settings=settings,
            today=today,
        )
    assert "demand_weights" in captured, "export fold never reached optimize_breaks"
    return captured["demand_weights"]


def test_demand_folds_equivalent():
    """Both folds must produce byte-identical per-segment weight maps."""
    segments = _segments()
    settings = _settings()

    with _patched_signals():
        live = _run_live_fold(segments, settings, REFERENCE_DATE)
        csv = _run_csv_fold(segments, settings, REFERENCE_DATE)

    # Byte-identical: same keys, same float values, same canonical serialization.
    assert json.dumps(live, sort_keys=True) == json.dumps(csv, sort_keys=True), (
        f"demand folds diverged\n live={live}\n csv={csv}"
    )
    assert live == csv
    assert set(live) == {s.segment_id for s in segments}

    # Guard: the synthetic signal is genuinely non-uniform and two-sided, so this
    # is not the trivial all-1.0 comparison that hides divergence in production.
    values = list(live.values())
    assert len({round(v, 6) for v in values}) > 1, f"expected non-uniform weights, got {live}"
    assert any(v < 1.0 for v in values), f"expected a sub-1.0 (pacing) weight, got {live}"
    assert any(v > 1.0 for v in values), f"expected an above-1.0 (demand/inventory) weight, got {live}"


def test_equivalence_detects_divergence():
    """The equivalence assertion is a real gate, not vacuous.

    Force the live fold to drop pacing (as a broken consolidation might) while the
    export fold keeps it, and confirm the two maps then differ. If this ever
    passed with equal maps, the equivalence test above would be blind to a real
    fold divergence.
    """
    segments = _segments()
    settings = _settings()

    with _patched_signals():
        live_knobs = dict(service._pacing_knobs_from_settings(settings), enabled=False)
        live = service._assemble_demand_weights(list(segments), today=REFERENCE_DATE, pacing_knobs=live_knobs)
        csv = _run_csv_fold(segments, settings, REFERENCE_DATE)

    assert live != csv, "equivalence test would be blind to a pacing-gate divergence"


def _main() -> int:
    failures = 0
    for name in ("test_demand_folds_equivalent", "test_equivalence_detects_divergence"):
        try:
            globals()[name]()
            print(f"PASS {name}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {name}: {exc}")
    print("OK" if not failures else f"{failures} FAILURE(S)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
