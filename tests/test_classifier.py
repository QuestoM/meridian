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


# ------------------------------------------- reading the synopsis, and its limits
#
# MEASURED: the classifier falls to Other on 3.3% of data/Programmes.csv, the
# corpus it was tuned against, and on 40.8% of the pulled feed, because 156 of
# 417 historical titles resolve through a hand-written list taken from that same
# corpus. The feed carries a synopsis on 632 of 638 broadcasts; reading it takes
# the unknown rate to 21.6%. These pin the two ways of getting that wrong, both
# of which were measured on the real feed before the code was written.

def _classifier():
    from kairos.data.classifier import ProgramClassifier

    return ProgramClassifier.from_yaml()


def test_a_synopsis_rescues_a_title_that_says_nothing():
    c = _classifier()
    title = "לילה ישראלי (ש.ח.)"
    assert c.classify(title).category == "Other"
    verdict = c.classify(title, description="לקט קליפים ישראלים מכל הזמנים.")
    assert verdict.category == "Music"
    assert verdict.rule == "description"


def test_a_title_that_resolves_is_never_overruled_by_its_own_synopsis():
    """THE FIRST FAILURE MODE, MEASURED. Scoring title and description together
    changes the verdict on 39 of 378 titles the classifier already reads
    correctly: "מבזק חדשות" becomes Documentary because its synopsis says
    "דיווחי אקטואליה". A title that resolves is the better evidence."""
    c = _classifier()
    title = "מבזק חדשות"
    alone = c.classify(title)
    assert alone.category != "Other"
    with_synopsis = c.classify(
        title, description="עדכוני חדשות ודיווחי אקטואליה שוטפים בזמן אמת מהארץ ומהעולם.")
    assert with_synopsis.category == alone.category
    assert with_synopsis.rule == alone.rule


def test_a_programme_named_inside_a_synopsis_is_a_mention_and_not_an_identity():
    """THE SECOND FAILURE MODE, MEASURED. The specific rule matches a programme
    NAME anywhere in the text, and a synopsis names other programmes: a guest
    introduced as a star of "האח הגדול" made an unrelated broadcast Reality.
    Only the weighted keywords run on a description."""
    c = _classifier()
    verdict = c.classify(
        "חשיפה - הפרשה שאיש לא סיפר",
        description='תום חיימוב, כוכב "האח הגדול", מצא את אהבת חייו.')
    assert verdict.rule != "specific"


def test_the_verdict_says_the_genre_came_from_a_synopsis():
    """A genre read out of a synopsis is weaker evidence than one read out of a
    title, and a reader downstream has to be able to tell which they were given."""
    c = _classifier()
    verdict = c.classify("שם שלא אומר כלום", description="לקט קליפים ישראלים.")
    assert verdict.rule == "description"


def test_no_synopsis_leaves_every_verdict_exactly_where_it_was():
    """data/Programmes.csv has no Description column at all, and its 3.3% unknown
    rate must not move by one row."""
    c = _classifier()
    for title in ("מבזק חדשות", "מאסטר שף", "שם שלא אומר כלום", ""):
        assert c.classify(title) == c.classify(title, description=None)
        assert c.classify(title) == c.classify(title, description="")


def test_classify_series_takes_descriptions_positionally_and_tolerates_fewer():
    c = _classifier()
    frame = c.classify_series(
        ["לילה ישראלי", "מבזק חדשות", "שם אחר"],
        ["לקט קליפים ישראלים מכל הזמנים."],
    )
    assert len(frame) == 3
    assert frame["rule"].iloc[0] == "description"


def test_a_programme_name_in_a_synopsis_cannot_reach_its_category_by_any_path():
    """Blocking the `specific` rule was NOT enough, and the claim that it removed
    the class was wrong. 34 of the names in specific_programs are ALSO category
    keywords, so the same hijack came straight back through the keyword path and
    was live on the feed: an expose whose synopsis introduces a guest as a star
    of "האח הגדול" was classified Reality. The names are now removed from a
    synopsis before it is scored, which costs exactly one resolution out of 122
    and the one it costs is this false positive."""
    c = _classifier()
    verdict = c.classify(
        "חשיפה - הפרשה המטורפת של כוכב הריאליטי",
        description='תום חיימוב, כוכב "האח הגדול", מצא את אהבת חייו וניהל קרב משפטי.')
    assert verdict.category == "Other"


def test_a_programme_name_in_a_TITLE_is_still_its_identity():
    """The removal applies to synopses only. In a title the programme name is
    exactly what identifies it, and stripping there would break the specific rule
    that resolves 1,261 historical titles."""
    c = _classifier()
    assert c.classify("האח הגדול").category != "Other"


def test_the_cached_ai_classifier_accepts_a_description_like_the_base():
    """It documents itself as a drop-in and the base signature grew a keyword. A
    drop-in that raises TypeError on the new argument is a drop-in only until
    somebody uses it - dormant here because no override cache exists on most
    machines, which is how it went unnoticed."""
    from kairos.data.ai_classifier import CachedClassifier

    cached = CachedClassifier(_classifier(), {})
    verdict = cached.classify("לילה ישראלי", description="לקט קליפים ישראלים.")
    assert verdict.category == "Music"


def test_the_ai_queue_does_not_pay_to_be_told_what_the_synopsis_already_said():
    """Every title on this list becomes a model call."""
    from kairos.data.ai_classifier import unclassified_titles

    c = _classifier()
    titles = ["לילה ישראלי", "שם שבאמת אף אחד לא מזהה"]
    notes = ["לקט קליפים ישראלים מכל הזמנים.", "אין כאן שום רמז."]
    assert "לילה ישראלי" in unclassified_titles(titles, c)
    assert "לילה ישראלי" not in unclassified_titles(titles, c, notes)
