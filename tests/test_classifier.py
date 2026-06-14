"""Unit tests for the Kairos program classifier.

Pure-algorithm tests (no LLM, no network): exercise the weighted-scoring
classifier against known Israeli TV titles, rerun detection, normalisation,
honest-unknown handling, and the legacy collapse mapping.
"""

from __future__ import annotations

import pytest

from kairos.data.classifier import Classification, ProgramClassifier


@pytest.fixture(scope="module")
def clf() -> ProgramClassifier:
    return ProgramClassifier.from_yaml()


@pytest.mark.parametrize(
    "title,expected",
    [
        ("חדשות 12 - מבזק", "News"),
        ("מהדורת חדשות הערב", "News"),
        ("ארץ נהדרת", "Comedy"),
        ("קופה ראשית", "Comedy"),
        ("מאסטר שף", "Reality"),
        ("הישרדות VIP", "Reality"),
        ("טהרן", "Drama"),
        ("המקור - עונה 24", "Documentary"),
        ("עובדה", "Documentary"),
        ("ישראל הבוקר", "Morning Program"),
        ("הפטריוטים", "Talk Show"),
        ("פסטיבל הפסנתר", "Music"),
        ("חרבות ברזל - שידור מיוחד", "Special Event"),
    ],
)
def test_known_titles(clf: ProgramClassifier, title: str, expected: str) -> None:
    assert clf.classify(title).category == expected


def test_specific_override_is_confident(clf: ProgramClassifier) -> None:
    result = clf.classify("ארץ נהדרת")
    assert result.rule == "specific"
    assert result.confidence == 1.0


def test_rerun_marker_detected_without_changing_genre(clf: ProgramClassifier) -> None:
    result = clf.classify("הפטריוטים ש.ח")
    assert result.is_rerun is True
    assert result.category == "Talk Show"


def test_unknown_title_is_honest_other(clf: ProgramClassifier) -> None:
    result = clf.classify("כותרת דמיונית שאינה קיימת בלוח")
    assert result.category == "Other"
    assert result.confidence == 0.0
    assert result.rule in {"fallback", "empty"}


def test_empty_and_nan_titles(clf: ProgramClassifier) -> None:
    for value in ("", "   ", None, "nan"):
        result = clf.classify(value)
        assert result.category == "Other"


def test_bracketed_spots_title_normalised(clf: ProgramClassifier) -> None:
    # Spots log titles arrive bracketed and sometimes joined with " * ".
    result = clf.classify("[חדשות 12 - מבזק * ישראל הבוקר]")
    assert result.category in {"News", "Morning Program"}
    assert result.confidence > 0.0


def test_legacy_mapping_collapse(clf: ProgramClassifier) -> None:
    assert clf.classify("חדשות 12").legacy_type == "News"
    assert clf.classify("הפטריוטים").legacy_type == "Other"  # Talk Show -> Other in legacy set
    assert clf.classify("מאסטר שף").legacy_type == "Reality"


def test_classify_series_shape(clf: ProgramClassifier) -> None:
    titles = ["חדשות 12", "ארץ נהדרת", "כותרת לא ידועה"]
    frame = clf.classify_series(titles)
    assert list(frame.columns) == ["title", "category", "confidence", "rule", "is_rerun", "legacy_type"]
    assert len(frame) == 3


def test_coverage_report_keys(clf: ProgramClassifier) -> None:
    report = clf.coverage_report(["חדשות 12", "ארץ נהדרת", "כותרת לא ידועה"])
    assert report["total"] == 3
    assert "by_category" in report
    assert "uncovered" in report
    assert report["covered"] + len(report["uncovered"]) == report["total"]


def test_classification_dataclass_is_frozen() -> None:
    result = Classification("News", 1.0, "specific", False, "News")
    with pytest.raises(Exception):
        result.category = "Drama"  # type: ignore[misc]
