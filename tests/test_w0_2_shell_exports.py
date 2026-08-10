"""W0-2 shell exports that remain shared after the destination split."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "tv-break-dashboard" / "src"


def _read(relative: str) -> str:
    return (SRC / relative).read_text(encoding="utf-8")


def test_the_program_type_helper_delegates_to_the_one_shared_table() -> None:
    helpers = _read("shell/surface-helpers.js")
    labels = _read("shell/labels.js")

    assert "programTypeLabel as sharedProgramTypeLabel" in helpers
    assert "return sharedProgramTypeLabel(String(value ?? '').trim(), locale);" in helpers
    assert "PROGRAM_TYPE_LABELS_HE" not in helpers
    assert "export const PROGRAM_TYPE_LABELS_HE" in labels


def test_removed_dead_exports_stay_gone_and_unreferenced() -> None:
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
        path.relative_to(SRC).as_posix(): path.read_text(encoding="utf-8")
        for path in SRC.rglob("*")
        if path.suffix in {".js", ".jsx"}
    }
    for name in removed:
        hits = [path for path, source in sources.items() if name in source]
        assert hits == [], f"dead export {name!r} is still referenced in {hits}"
