"""Tests for the advertiser agreements module.

They prove three things: the real header-only reference stub yields no
agreements and therefore no violations, a satisfied must-air commitment passes,
and an unmet one is reported as a typed violation.
"""

from __future__ import annotations

from dataclasses import dataclass

from kairos.optimize.agreements import (
    CPP,
    FIX,
    MUST_AIR,
    AdvertiserAgreement,
    AgreementConstraint,
    AgreementViolation,
    DEFAULT_AGREEMENTS_PATH,
    agreement_violations,
    load_agreements,
)


# A minimal stand-in for kairos.optimize.optimizer.BreakPlacement. The real
# placement does not carry a daypart, so the synthetic one only adds the fields
# the agreement checks actually read (channel and an optional daypart).
@dataclass(frozen=True)
class FakePlacement:
    channel: str
    daypart: str | None = None


@dataclass(frozen=True)
class FakeResult:
    placements: tuple
    segments: tuple = ()


def _result(*placements: FakePlacement) -> FakeResult:
    return FakeResult(placements=tuple(placements))


# The real reference stub: headers only, no rows.

def test_real_reference_file_has_only_headers() -> None:
    # The committed reference file is the header-only stub, so it must yield
    # zero agreements with no exception and no invented rows.
    agreements = load_agreements(DEFAULT_AGREEMENTS_PATH)
    assert agreements == []


def test_no_agreements_means_no_violations() -> None:
    result = _result(FakePlacement(channel="קשת 12", daypart="prime"))
    assert agreement_violations(result, []) == []


def test_load_missing_file_returns_empty(tmp_path) -> None:
    missing = tmp_path / "nope.csv"
    assert load_agreements(missing) == []


def test_load_header_only_file_returns_empty(tmp_path) -> None:
    path = tmp_path / "AdvertiserAgreements.csv"
    path.write_text(
        "AdvertiserName,CampaignName,AgreementType,ProgramTitle,Date_From,"
        "Date_To,DayOfWeek,PositionInBreak,Value1,Value2\n",
        encoding="utf-8",
    )
    assert load_agreements(path) == []


# Parsing real-shaped rows.

def test_load_parses_fix_and_cpp_rows(tmp_path) -> None:
    path = tmp_path / "AdvertiserAgreements.csv"
    path.write_text(
        "AdvertiserName,AgreementType,Channel,Daypart,MinBreaks,Commitment\n"
        "Tnuva,FIX,קשת 12,prime,2,must_air\n"
        "Osem,CPP,רשת 13,daytime,0,\n",
        encoding="utf-8",
    )
    agreements = load_agreements(path)
    assert [a.advertiser for a in agreements] == ["Tnuva", "Osem"]

    tnuva, osem = agreements
    assert tnuva.agreement_type == FIX
    assert tnuva.is_fixed_price is True
    assert len(tnuva.must_air_commitments) == 1
    assert tnuva.must_air_commitments[0].min_breaks == 2

    assert osem.agreement_type == CPP
    assert osem.is_fixed_price is False
    assert osem.must_air_commitments == ()


def test_row_without_advertiser_is_skipped(tmp_path) -> None:
    path = tmp_path / "AdvertiserAgreements.csv"
    path.write_text(
        "AdvertiserName,AgreementType\n"
        ",FIX\n"
        "Strauss,CPP\n",
        encoding="utf-8",
    )
    agreements = load_agreements(path)
    assert [a.advertiser for a in agreements] == ["Strauss"]


# Must-air checking against a result.

def _must_air_agreement(min_breaks: int, daypart: str = "prime", channel: str = "קשת 12") -> AdvertiserAgreement:
    return AdvertiserAgreement(
        advertiser="Tnuva",
        agreement_type=FIX,
        channel=channel,
        daypart=daypart,
        constraints=(AgreementConstraint(
            kind=MUST_AIR, daypart=daypart, channel=channel, min_breaks=min_breaks,
        ),),
    )


def test_satisfied_must_air_has_no_violation() -> None:
    agreement = _must_air_agreement(min_breaks=2)
    result = _result(
        FakePlacement(channel="קשת 12", daypart="prime"),
        FakePlacement(channel="קשת 12", daypart="prime"),
        FakePlacement(channel="רשת 13", daypart="prime"),  # other channel, ignored
    )
    assert agreement_violations(result, [agreement]) == []


def test_violated_must_air_is_reported() -> None:
    agreement = _must_air_agreement(min_breaks=3)
    result = _result(
        FakePlacement(channel="קשת 12", daypart="prime"),
        FakePlacement(channel="קשת 12", daypart="daytime"),  # wrong daypart
    )
    violations = agreement_violations(result, [agreement])
    assert len(violations) == 1
    violation = violations[0]
    assert isinstance(violation, AgreementViolation)
    assert violation.code == "must_air"
    assert violation.observed == 1
    assert violation.limit == 3
    assert "Tnuva" in violation.scope


def test_exclusivity_constraint_is_not_checked() -> None:
    # An exclusivity commitment is recorded but not checkable, so it never
    # produces a violation here even when the schedule could breach it.
    from kairos.optimize.agreements import EXCLUSIVITY

    agreement = AdvertiserAgreement(
        advertiser="Coca-Cola",
        agreement_type=CPP,
        constraints=(AgreementConstraint(
            kind=EXCLUSIVITY, daypart="prime", checkable=False,
        ),),
    )
    result = _result(FakePlacement(channel="כאן 11", daypart="prime"))
    assert agreement_violations(result, [agreement]) == []


def test_commitment_without_daypart_counts_every_break() -> None:
    agreement = AdvertiserAgreement(
        advertiser="Elite",
        agreement_type=FIX,
        constraints=(AgreementConstraint(kind=MUST_AIR, min_breaks=2),),
    )
    result = _result(
        FakePlacement(channel="קשת 12"),
        FakePlacement(channel="רשת 13"),
    )
    assert agreement_violations(result, [agreement]) == []
