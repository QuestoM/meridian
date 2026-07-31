"""QA wave: dashboard component fixes verified against the real backend vocab.

These tests pin the cross-layer contracts this wave's component fixes rely on,
so a backend vocabulary change cannot silently reopen the bugs:

* ScheduleStalenessBanner must know every operator-facing input-group label the
  freshness engine can emit (it previously dropped the five extended groups and
  rendered a double-verb fallback sentence).
* The shared daypart label helper must cover every daypart key the engine's
  hour mapping can produce (plus the yield endpoint's "unclassified" fallback),
  because YieldView and ConstraintBuilder now localize through it.
* The dead helper exports removed from advertisers-helpers.js and
  surface-helpers.js must stay gone and unreferenced.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "tv-break-dashboard" / "src"


def _sources(pattern: str) -> list[Path]:
    """Every source under src/, at any depth, ignoring installed packages."""
    return sorted(p for p in SRC.rglob(pattern) if "node_modules" not in p.parts)


def _find(name: str) -> Path:
    """A component by basename, wherever the tree puts it.

    These contracts are about what a component says, not where it sits, and the
    components move between trees as surfaces are reorganised. Resolving by
    basename keeps the assertions pinned to the content. The uniqueness check is
    the point: two files with one name would make the assertion ambiguous.
    """
    matches = [p for p in _sources(name) if p.name == name]
    assert len(matches) == 1, f"expected exactly one {name} under src/, found {matches}"
    return matches[0]


def _read(name: str) -> str:
    return _find(name).read_text(encoding="utf-8")


def test_staleness_banner_covers_every_backend_group_label() -> None:
    """Every label schedule_freshness can put into `changed` maps in the banner."""
    from kairos.export.schedule_freshness import GROUP_LABELS

    source = _read("ScheduleStalenessBanner.jsx")
    for key, label in GROUP_LABELS.items():
        assert f"'{label}'" in source or f"{key}:" in source, (
            f"ScheduleStalenessBanner.jsx has no label mapping for backend group "
            f"{key!r} (emitted as {label!r}); it would fall back to verbatim text "
            f"instead of a bilingual label"
        )


def test_staleness_banner_passes_unknown_labels_through() -> None:
    """An unknown changed entry must render verbatim, never be dropped."""
    source = _read("ScheduleStalenessBanner.jsx")
    assert "changedLabels[key] || String(key" in source


def test_staleness_banner_has_no_double_verb_fallback() -> None:
    """The old empty-list fallback composed 'הקלט השתנה' + 'השתנו' (double verb)."""
    source = _read("ScheduleStalenessBanner.jsx")
    assert "הקלט השתנה" not in source
    assert "${changedPhrase} השתנו" not in source
    # The new agreement-free frame must be present for both list and empty cases.
    assert "חל שינוי ב${changedPhrase}" in source
    assert "חל שינוי בקלט הלוח" in source


def test_daypart_label_helper_covers_engine_keys() -> None:
    """surface-helpers daypartLabel covers every key daypart_for_hour can emit."""
    from kairos.data.dayparts import daypart_for_hour

    engine_keys = {daypart_for_hour(hour) for hour in range(24)}
    engine_keys.discard(None)
    assert engine_keys, "engine produced no daypart keys at all"
    source = _read("surface-helpers.js")
    for key in sorted(engine_keys) + ["unclassified"]:
        assert f"{key}:" in source, f"daypartLabel is missing engine key {key!r}"


def test_removed_dead_exports_stay_gone_and_unreferenced() -> None:
    """The dead helpers were deleted and nothing in src/ still references them."""
    removed = [
        "DAYPART_PRESETS",
        "chipOptions",
        "filterAdvertisers",
        "sortAdvertisers",
        "computeSummary",
        "collectDaypartTokens",
        "fetchJsonOrError",
    ]
    sources = {
        str(path.relative_to(SRC)): path.read_text(encoding="utf-8")
        for path in _sources("*.js") + _sources("*.jsx")
    }
    # The sweep must reach the whole tree. A flat glob here went vacuous the
    # moment the components moved into per-surface directories: it kept passing
    # while checking almost nothing, so the reach is asserted before the content.
    assert "shell/surface-helpers.js" in sources, (
        f"the sweep did not reach the moved sources; it found {sorted(sources)}"
    )
    for name in removed:
        hits = [file for file, text in sources.items() if name in text]
        assert hits == [], f"dead export {name!r} is still referenced in {hits}"


def test_freshness_changed_contract_still_emits_group_labels() -> None:
    """schedule_freshness reports GROUP_LABELS values, which the banner keys on."""
    from kairos.export import schedule_freshness as sf

    # The stale path maps internal keys through GROUP_LABELS; verify the mapping
    # the banner depends on exists for every ordered group.
    for key in sf._GROUP_ORDER:
        assert key in sf.GROUP_LABELS
