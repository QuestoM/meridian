"""The competitor's schedule, pulled instead of typed — and what must never break.

Every competitor input this engine had arrived as a file somebody typed. This
suite covers the feeder that replaces that, and it is written around the three
ways such a thing rots: a failed pull that looks successful, an identity that
stops being stable, and a cache that keeps answering after the world changed.

The fixture is the REAL captured week from Keshet (127 broadcasts, 70 titles,
53 series), not an invention, because every design decision here was made by
measuring that file and would be untestable against a tidier one.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from kairos.model import keshet_enrich as enrich
from kairos.model import keshet_epg as epg
from kairos.model import keshet_refresh as refresh

CAPTURE = Path("/Users/home/Code/experiments/ads/artifacts/kway-capture/keshet-epg.json")


@pytest.fixture()
def payload():
    if not CAPTURE.exists():
        pytest.skip("the captured Keshet week is not on this machine")
    return json.loads(CAPTURE.read_text(encoding="utf-8"))


@pytest.fixture()
def rows(payload):
    out, _ = epg.to_contract_rows(payload, channel="קשת 12")
    return out


# ----------------------------------------------------------- the conversion

def test_the_whole_published_week_survives_the_conversion(payload, rows):
    _, status = epg.to_contract_rows(payload, channel="קשת 12")
    assert status["records_in"] == len(rows), "broadcasts were lost in conversion"
    assert status["dropped"] == [], f"records were dropped: {status['dropped']}"
    assert status["days"], "no broadcast days came through"


def test_the_existing_competitor_loader_reads_it_unchanged(rows, tmp_path):
    """The point of the whole exercise: feed the contract, do not replace it."""
    from kairos.model.future_epg import load_future_competitor_epg

    target = epg.write_contract_csv(rows, tmp_path / "CompetitorProgrammes.csv")
    frame, status = load_future_competitor_epg(target)
    assert status["present"] is True, status["reason"]
    assert frame is not None and len(frame) == len(rows)
    assert status["channels"] == ["קשת 12"]


def test_a_broadcast_running_past_midnight_ends_after_it_started(payload):
    """DisplayEndTime is a wall clock with no date, so it is never trusted.

    A programme starting 23:40 and running 40 minutes ends at 00:20 the NEXT
    day. Taking the published end clock would place its end before its start and
    hand the model a negative-length broadcast.
    """
    late = copy.deepcopy(payload)
    record = late["json"]["data"]["programs"][0]
    record["StartTime"] = "18/08/2026 23:40:00"
    record["DurationMs"] = 40 * 60 * 1000
    record["DisplayEndTime"] = "00:20"
    out, _ = epg.to_contract_rows(late, channel="ק")
    row = next(r for r in out if r["Start time"] == "23:40:00")
    assert row["Duration"] == 2400
    assert row["End time"] == "00:20:00"


def test_a_payload_with_no_programmes_raises_instead_of_reading_as_empty():
    """'The rival airs nothing' and 'we could not read it' must never match."""
    with pytest.raises(epg.EpgShapeError):
        epg.programmes_of({"json": {"data": {}}})


def test_a_record_that_cannot_be_placed_in_time_is_counted_not_hidden(payload):
    broken = copy.deepcopy(payload)
    broken["json"]["data"]["programs"][3]["StartTime"] = "not a date"
    broken["json"]["data"]["programs"][3]["Date"] = ""
    _, status = epg.to_contract_rows(broken, channel="ק")
    assert len(status["dropped"]) == 1
    assert "unreadable start clock" in status["dropped"][0]["reason"]


# ------------------------------------------------------------- the identity

def test_the_series_and_the_episode_both_come_free_from_the_title():
    series, episode = enrich.series_of("רוקדים עם כוכבים – ששת הגדולים בדו קרב")
    assert series == "רוקדים עם כוכבים"
    assert episode == "ששת הגדולים בדו קרב"


def test_a_programme_with_no_episode_is_its_own_series():
    series, episode = enrich.series_of("מבזק חדשות")
    assert series == "מבזק חדשות"
    assert episode == ""


def test_retitled_episodes_collapse_to_one_series_so_they_are_paid_for_once(rows):
    """Measured: 5 series in one week retitle themselves every episode.

    Keying on the full title means paying for every episode of every returning
    show, forever. That is the failure this identity exists to prevent, so the
    collapse is asserted on the real week rather than described.
    """
    titles = {r["Title"] for r in rows}
    series = {enrich.series_of(t)[0] for t in titles}
    assert len(series) < len(titles), "no series collapsed; the identity is doing nothing"
    dancing = [t for t in titles if t.startswith("רוקדים עם כוכבים")]
    assert len(dancing) > 1, "the fixture no longer exercises retitled episodes"
    assert len({enrich.series_of(t)[0] for t in dancing}) == 1


# -------------------------------------------------------------- the memory

def _classifier():
    from kairos.data.classifier import ProgramClassifier

    return ProgramClassifier.from_yaml()


def test_a_second_week_of_the_same_programmes_asks_nothing(rows, tmp_path):
    memory = enrich.SeriesMemory(tmp_path / "memory.json")
    classify = _classifier().classify
    first = enrich.plan(rows, classify=classify, memory=memory)
    for series in first["ask"]:
        memory.put(series, str(next(
            r for r in rows if enrich.series_of(r["Title"])[0] == series)["Description"]),
            {"category": "Reality", "season": None, "episode": None, "confidence": "high"})
    memory.save()
    again = enrich.plan(rows, classify=classify, memory=memory)
    assert again["calls_needed"] == 0, (
        f"a repeat week still costs {again['calls_needed']} calls")


def test_a_series_whose_description_changed_is_asked_again(rows, tmp_path):
    """Six titles in one real week carried more than one description.

    A presenter change under a stable name is exactly when a remembered reading
    goes wrong, and exactly when a name-keyed cache would never notice.
    """
    memory = enrich.SeriesMemory(tmp_path / "memory.json")
    classify = _classifier().classify
    target = next(r for r in rows if enrich.series_of(r["Title"])[0] == "חשיפה")
    memory.put("חשיפה", target["Description"],
               {"category": "Documentary", "season": None, "episode": None, "confidence": "high"})
    memory.save()
    assert enrich.plan(rows, classify=classify, memory=memory)["stale"] == []

    drifted = copy.deepcopy(rows)
    for row in drifted:
        if enrich.series_of(row["Title"])[0] == "חשיפה":
            row["Description"] = "סדרת תחקירים חדשה בהגשת אילנה דיין על מערכת הביטחון"
    plan = enrich.plan(drifted, classify=classify, memory=memory)
    assert "חשיפה" in plan["stale"]
    assert "חשיפה" in plan["ask"]


def test_punctuation_churn_in_a_description_is_not_a_change_of_meaning(rows, tmp_path):
    memory = enrich.SeriesMemory(tmp_path / "memory.json")
    row = next(r for r in rows if r["Description"])
    series = enrich.series_of(row["Title"])[0]
    memory.put(series, row["Description"],
               {"category": "News", "season": None, "episode": None, "confidence": "high"})
    _, state = memory.get(series, row["Description"] + "  !!  ")
    assert state == "fresh", "cosmetic churn would re-buy the same answer"


def test_the_model_may_not_invent_a_category_outside_the_engines_own_list():
    from kairos.data.classifier import ProgramClassifier

    categories = list(ProgramClassifier.from_yaml()._priority.keys())
    schema = enrich.enrichment_schema(categories)
    allowed = schema["properties"]["category"]["enum"]
    assert set(categories) <= set(allowed)
    assert enrich.UNFITTABLE in allowed, "there is no honest way to say 'none of these'"
    assert schema["additionalProperties"] is False


def test_an_unresolved_series_carries_no_category_rather_than_a_guess(rows, tmp_path):
    """Downstream knows how to treat unknown. It cannot un-believe wrong."""
    memory = enrich.SeriesMemory(tmp_path / "memory.json")
    out, status = enrich.enrich(
        rows, classify=_classifier().classify,
        categories=["News", "Reality"], memory=memory, call=None)
    unresolved = [r for r in out if r["CategorySource"] == "unresolved"]
    assert unresolved, "the fixture no longer exercises the no-provider path"
    assert all(r["Category"] == "" for r in unresolved)
    assert all(r["Season"] is None and r["Episode"] is None for r in unresolved)
    assert status["asked"] == 0


# ------------------------------------------------------------- the refresh

def test_a_failed_pull_keeps_the_schedule_and_says_it_is_stale(payload, tmp_path):
    target = tmp_path / "CompetitorProgrammes.csv"
    refresh.refresh(fetch=lambda: payload, channel="ק", target=target)
    before = len(refresh._read_rows(target))

    def explode():
        raise ConnectionError("no session")

    result = refresh.refresh(fetch=explode, channel="ק", target=target)
    assert result["refreshed"] is False
    assert result["kept_rows"] == before
    assert len(refresh._read_rows(target)) == before, "a failed pull erased the schedule"
    assert "לא רוענן" in refresh.headline(result, "he")


def test_an_empty_pull_never_erases_a_schedule_we_had(payload, tmp_path):
    """No publication ever claimed the rival airs nothing."""
    target = tmp_path / "CompetitorProgrammes.csv"
    refresh.refresh(fetch=lambda: payload, channel="ק", target=target)
    before = len(refresh._read_rows(target))
    result = refresh.refresh(
        fetch=lambda: {"json": {"data": {"programs": []}}}, channel="ק", target=target)
    assert result["refreshed"] is False
    assert len(refresh._read_rows(target)) == before


def test_a_programme_pushed_later_reads_as_a_move_not_a_deletion(payload, tmp_path):
    """The single most decision-relevant change a rival can make."""
    target = tmp_path / "CompetitorProgrammes.csv"
    refresh.refresh(fetch=lambda: payload, channel="ק", target=target)

    later = copy.deepcopy(payload)
    record = next(p for p in later["json"]["data"]["programs"]
                  if p["DisplayStartTime"].startswith("21"))
    day, clock = record["StartTime"].split(" ")
    hour, minute, second = clock.split(":")
    record["StartTime"] = f"{day} {int(hour) + 1:02d}:{minute}:{second}"

    result = refresh.refresh(fetch=lambda: later, channel="ק", target=target)
    changes = result["changes"]
    assert len(changes["moved"]) == 1, changes
    assert not changes["added"] and not changes["removed"], (
        "a move was reported as a deletion plus an addition")


def test_an_unchanged_week_reports_quiet(payload, tmp_path):
    target = tmp_path / "CompetitorProgrammes.csv"
    refresh.refresh(fetch=lambda: payload, channel="ק", target=target)
    result = refresh.refresh(fetch=lambda: payload, channel="ק", target=target)
    assert result["changes"]["quiet"] is True
    assert result["changes"]["changes"] == 0


def test_a_slot_the_broadcaster_finally_named_reads_as_announced(payload, tmp_path):
    """'פרטים יפורסמו בהמשך' becoming a real title is news of a different kind
    from a programme appearing where none was scheduled."""
    target = tmp_path / "CompetitorProgrammes.csv"
    base = copy.deepcopy(payload)
    base["json"]["data"]["programs"][10]["ProgramName"] = "פרטים יפורסמו בהמשך"
    refresh.refresh(fetch=lambda: base, channel="ק", target=target)

    named = copy.deepcopy(base)
    named["json"]["data"]["programs"][10]["ProgramName"] = "הסרט הגדול של החג"
    result = refresh.refresh(fetch=lambda: named, channel="ק", target=target)
    assert len(result["changes"]["announced"]) == 1
    assert not result["changes"]["added"]


# ------------------------------------------------------- the channel's name

def test_the_engines_own_spelling_is_what_gets_written():
    from kairos.data.loaders import CHANNELS

    for probe in ("קשת 12", " קשת  12 ", "קשת12", "‏קשת 12", "קשת"):
        assert epg.resolve_channel(probe, CHANNELS) == "קשת 12", probe


def test_a_channel_this_engine_has_no_history_for_is_refused():
    """The failure this prevents is silent, which is why it is refused loudly.

    A schedule filed under an unknown name loads cleanly, validates cleanly and
    contributes exactly zero: future_epg gives an unmatched rival 0.0 strength,
    correctly, and nothing downstream can tell a channel with no history from a
    channel whose name was mistyped.
    """
    from kairos.data.loaders import CHANNELS

    with pytest.raises(epg.UnknownChannel) as caught:
        epg.resolve_channel("ערוץ 12", CHANNELS)
    assert "קשת 12" in str(caught.value), "the refusal does not say what is allowed"


def test_a_name_that_could_be_two_channels_is_refused_rather_than_guessed():
    with pytest.raises(epg.UnknownChannel) as caught:
        epg.resolve_channel("ערוץ", ("ערוץ 12", "ערוץ 13"))
    assert "ערוץ 12" in str(caught.value) and "ערוץ 13" in str(caught.value)


def test_with_no_registry_at_all_nothing_is_attributed():
    with pytest.raises(epg.UnknownChannel):
        epg.resolve_channel("קשת 12", ())


# --------------------------------------------------------------- the feed

def test_the_feed_refuses_to_file_a_rival_under_the_operators_own_channel(tmp_path):
    """Both names are real, both resolve, and the result would be silence.

    counterprogramming_features_for_window drops the operator's channel from
    the rival list. A competitor schedule stamped with it therefore leaves NO
    rivals, returns None, and every counter-programming adjustment becomes
    exactly zero — with a valid file on disk and nothing to see.
    """
    from kairos.model import keshet_feed

    result = keshet_feed.pull(channel="רשת 13", operator_channel="רשת 13",
                              target=tmp_path / "x.csv")
    assert result["refreshed"] is False
    assert "own channel" in result["reason"]
    assert not (tmp_path / "x.csv").exists(), "a schedule was written anyway"


def test_the_feed_stops_on_an_unknown_channel_before_it_signs_in(tmp_path):
    from kairos.model import keshet_feed

    result = keshet_feed.pull(channel="Channel 12", operator_channel="",
                              target=tmp_path / "x.csv")
    assert result["refreshed"] is False
    assert result["needs_human"] is True
    assert not (tmp_path / "x.csv").exists()


def test_a_source_that_cannot_be_read_is_stale_and_never_quiet(monkeypatch, tmp_path):
    """'We could not read it' must never arrive as 'the rival changed nothing'.

    This used to be about a missing signed-in session, which was the only way
    Keshet could fail. It is not any more — no channel needs a credential — but
    the property it was protecting is the same one and outlives the reason.
    """
    from kairos.model import keshet_feed

    def broken(channel, days):
        raise ConnectionError("the publication is down")

    monkeypatch.setattr(keshet_feed, "_fetchers", lambda: {
        "mako": (broken, lambda p, *, channel: ([], {})),
        "freetv": (broken, lambda p, *, channel: ([], {})),
    })
    result = keshet_feed.pull(channel="קשת 12", operator_channel="",
                              target=tmp_path / "x.csv")
    assert result["refreshed"] is False
    assert "לא רוענן" in keshet_feed.headline(result, "he")
    assert "קשת 12" in keshet_feed.headline(result, "he"), "the line does not say which rival"


# -------------------------------------------------- one file, every rival

def _channel_rows(path, channel):
    return [r for r in refresh._read_rows(path) if r["Channel"] == channel]


def test_pulling_one_rival_never_erases_another(payload, tmp_path):
    """The whole point of a shared file, and the whole risk of one.

    The optimizer wants the competitive lineup in one place, which the contract
    has always allowed. A refresh that wrote the file wholesale would have made
    the second channel's pull a silent deletion of the first.
    """
    target = tmp_path / "CompetitorProgrammes.csv"
    refresh.refresh(fetch=lambda: payload, channel="קשת 12", target=target)
    keshet = len(_channel_rows(target, "קשת 12"))
    assert keshet, "the first channel was not written at all"

    refresh.refresh(fetch=lambda: payload, channel="כאן 11", target=target)
    assert len(_channel_rows(target, "קשת 12")) == keshet, "the first rival was erased"
    assert len(_channel_rows(target, "כאן 11")) == keshet
    assert len(refresh._read_rows(target)) == keshet * 2


def test_the_existing_loader_reads_every_channel_out_of_the_one_file(payload, tmp_path):
    from kairos.model.future_epg import load_future_competitor_epg

    target = tmp_path / "CompetitorProgrammes.csv"
    for channel in ("קשת 12", "כאן 11", "עכשיו 14"):
        refresh.refresh(fetch=lambda: payload, channel=channel, target=target)
    frame, status = load_future_competitor_epg(target)
    assert status["present"] is True
    assert status["channels"] == ["כאן 11", "עכשיו 14", "קשת 12"]


def test_a_change_on_one_channel_is_not_reported_against_another(payload, tmp_path):
    """The diff is per channel, or a rival that never moved reads as rebuilt."""
    import copy

    target = tmp_path / "CompetitorProgrammes.csv"
    refresh.refresh(fetch=lambda: payload, channel="קשת 12", target=target)
    refresh.refresh(fetch=lambda: payload, channel="כאן 11", target=target)

    moved = copy.deepcopy(payload)
    record = next(p for p in moved["json"]["data"]["programs"]
                  if p["DisplayStartTime"].startswith("21"))
    day, clock = record["StartTime"].split(" ")
    hour, minute, second = clock.split(":")
    record["StartTime"] = f"{day} {int(hour) + 1:02d}:{minute}:{second}"

    result = refresh.refresh(fetch=lambda: moved, channel="קשת 12", target=target)
    assert len(result["changes"]["moved"]) == 1
    assert result["changes"]["changes"] == 1, "another channel's rows entered the diff"

    quiet = refresh.refresh(fetch=lambda: payload, channel="כאן 11", target=target)
    assert quiet["changes"]["quiet"] is True, "a channel that did not move reported changes"


def test_each_channel_carries_its_own_age(payload, tmp_path):
    """A file's modified time cannot answer "how old is THIS rival's schedule".

    Refreshing one channel touches the file. Without a per-channel stamp, a
    channel nobody has pulled for a week reads as one minute old, which is the
    silent staleness this module exists to prevent.
    """
    from datetime import datetime, timedelta, timezone

    target = tmp_path / "CompetitorProgrammes.csv"
    long_ago = datetime(2026, 8, 1, tzinfo=timezone.utc)
    refresh.refresh(fetch=lambda: payload, channel="קשת 12", target=target, now=long_ago)
    refresh.refresh(fetch=lambda: payload, channel="כאן 11", target=target,
                    now=long_ago + timedelta(days=7))

    stamps = refresh.read_freshness(target)
    assert set(stamps) == {"קשת 12", "כאן 11"}

    def explode():
        raise ConnectionError("no session")

    stale = refresh.refresh(fetch=explode, channel="קשת 12", target=target,
                            now=long_ago + timedelta(days=7, hours=1))
    assert stale["stale_hours"] == pytest.approx(169.0, abs=0.2), (
        "the untouched channel borrowed the other channel's freshness")


def test_a_failed_pull_says_which_channel_it_kept(payload, tmp_path):
    target = tmp_path / "CompetitorProgrammes.csv"
    refresh.refresh(fetch=lambda: payload, channel="קשת 12", target=target)
    refresh.refresh(fetch=lambda: payload, channel="כאן 11", target=target)

    def explode():
        raise ConnectionError("no session")

    result = refresh.refresh(fetch=explode, channel="קשת 12", target=target)
    assert result["channel"] == "קשת 12"
    assert result["kept_rows"] < result["kept_rows_in_file"], (
        "the count for one channel is being reported as the whole file")


def test_one_rivals_source_failing_does_not_cost_the_others(payload, tmp_path):
    """Measured live, then pinned: with Keshet's session gone, כאן 11 and
    עכשיו 14 refreshed normally, Keshet said so and named the one human step,
    and all 300 of its rows survived. A lineup where one source can take the
    others down with it is a lineup that stops being pulled the first time a
    credential expires."""
    target = tmp_path / "CompetitorProgrammes.csv"
    refresh.refresh(fetch=lambda: payload, channel="קשת 12", target=target)
    kept = len(_channel_rows(target, "קשת 12"))

    def dead():
        raise ConnectionError("no session")

    broken = refresh.refresh(fetch=dead, channel="קשת 12", target=target)
    fine = refresh.refresh(fetch=lambda: payload, channel="כאן 11", target=target)

    assert broken["refreshed"] is False
    assert fine["refreshed"] is True
    assert len(_channel_rows(target, "קשת 12")) == kept, "the failed channel lost its rows"
    assert len(_channel_rows(target, "כאן 11")) == kept


def test_the_channel_that_failed_keeps_its_older_stamp(payload, tmp_path):
    """The other half of the same property: refreshing one channel must not
    make a channel that failed look freshly pulled."""
    from datetime import datetime, timedelta, timezone

    target = tmp_path / "CompetitorProgrammes.csv"
    first = datetime(2026, 8, 1, tzinfo=timezone.utc)
    refresh.refresh(fetch=lambda: payload, channel="קשת 12", target=target, now=first)

    def dead():
        raise ConnectionError("no session")

    later = first + timedelta(days=2)
    refresh.refresh(fetch=dead, channel="קשת 12", target=target, now=later)
    refresh.refresh(fetch=lambda: payload, channel="כאן 11", target=target, now=later)

    stamps = refresh.read_freshness(target)
    assert stamps["קשת 12"].startswith("2026-08-01"), "a failed pull moved the stamp"
    assert stamps["כאן 11"].startswith("2026-08-03")


# ------------------------------------------------- more than one way in

def test_the_second_source_is_used_when_the_first_cannot_be_read(payload, tmp_path, monkeypatch):
    """Keshet has two independent publications. One being down is not an outage.

    The order is a preference and not a last resort: the first source that
    returns a usable schedule wins, and what was tried before it is reported
    rather than hidden.
    """
    from kairos.model import keshet_feed

    def broken(channel, days):
        raise ConnectionError("the publication is down")

    monkeypatch.setattr(keshet_feed, "_fetchers", lambda: {
        "mako": (broken, lambda p, *, channel: (_ for _ in ()).throw(AssertionError)),
        "freetv": (lambda channel, days: payload,
                   lambda p, *, channel: epg.to_contract_rows(p, channel=channel)),
    })
    result = keshet_feed.pull(channel="קשת 12", operator_channel="",
                              target=tmp_path / "x.csv")
    assert result["refreshed"] is True
    assert result["source"] == "freetv"
    assert result["attempts"][0]["source"] == "mako", "the first attempt was not reported"


def test_every_source_failing_still_says_what_was_tried(tmp_path, monkeypatch):
    from kairos.model import keshet_feed

    def broken(channel, days):
        raise ConnectionError("the publication is down")

    monkeypatch.setattr(keshet_feed, "_fetchers", lambda: {
        "mako": (broken, lambda p, *, channel: ([], {})),
        "freetv": (broken, lambda p, *, channel: ([], {})),
    })
    result = keshet_feed.pull(channel="קשת 12", operator_channel="",
                              target=tmp_path / "x.csv")
    assert result["refreshed"] is False
    assert [a["source"] for a in result["attempts"]] == ["mako", "freetv"]


def test_no_rival_needs_a_credential(monkeypatch):
    """The daily pull must not depend on a session that can expire unwatched.

    Every channel's FIRST source is one that answers without an account. Kway
    stays in the catalogue and stays working; nothing daily reaches for it.
    """
    from kairos.model import keshet_feed

    for channel, sources in keshet_feed.SOURCES.items():
        assert sources, channel
        assert sources[0] in ("mako", "freetv"), (
            f"{channel} leads with {sources[0]}, which needs a credential")
